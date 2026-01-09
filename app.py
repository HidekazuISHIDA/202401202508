import json
from pathlib import Path
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

ARR_MODEL_PATH = MODELS_DIR / "model_A_timeseries.json"
SVC_MODEL_PATH = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH = MODELS_DIR / "model_A_waittime_30min.json"
WAITP90_MODEL_PATH = MODELS_DIR / "model_A_waittime_p90_30min.json"

ARR_COLS_PATH = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH = MODELS_DIR / "wait_calibration.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
SLOT_MINUTES = 30.0

WEATHER_CATS = ["晴", "曇", "雨", "雪"]

# ---------------- holiday ----------------
def _load_holidays() -> set:
    if not HOLIDAY_CSV_PATH.exists():
        return set()
    df = pd.read_csv(HOLIDAY_CSV_PATH, encoding="utf-8", engine="python")
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "日付"]:
            col = c
            break
    if col is None:
        col = df.columns[0]
    s = pd.to_datetime(df[col], errors="coerce").dropna().dt.date
    return set(s.tolist())

HOLIDAYS = _load_holidays()

def is_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    if d in HOLIDAYS:
        return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return True
    return False

def normalize_weather(text: str) -> str:
    s = str(text) if text is not None else ""
    if "雪" in s: return "雪"
    if "雨" in s: return "雨"
    if "曇" in s: return "曇"
    if "晴" in s: return "晴"
    return "曇"

def slot_id(ts: datetime) -> int:
    minutes = ts.hour * 60 + ts.minute
    base = 8 * 60
    return int((minutes - base) // 30)

def baseline_key(ts: datetime) -> str:
    return f"{ts.month}_{ts.weekday()}_{slot_id(ts)}"

def in_peak(ts: datetime) -> bool:
    # 8:30〜11:00
    h, m = ts.hour, ts.minute
    after = (h > 8) or (h == 8 and m >= 30)
    before = (h < 11) or (h == 11 and m == 0)
    return after and before

@st.cache_resource
def load_assets():
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    arr_bst = xgb.Booster(); arr_bst.load_model(str(ARR_MODEL_PATH))
    svc_bst = xgb.Booster(); svc_bst.load_model(str(SVC_MODEL_PATH))
    wait_bst = xgb.Booster(); wait_bst.load_model(str(WAIT_MODEL_PATH))
    wp90_bst = xgb.Booster(); wp90_bst.load_model(str(WAITP90_MODEL_PATH))

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    return arr_bst, arr_cols, svc_bst, wait_bst, wp90_bst, multi_cols, baseline, calib

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _predict_booster(booster: xgb.Booster, cols, df: pd.DataFrame) -> float:
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

def get_base(baseline, kind: str, key: str, stat="median", default=0.0):
    try:
        return float(baseline.get(kind, {}).get(key, {}).get(stat, default))
    except Exception:
        return float(default)

def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))

def simulate_one_day(target_date: date, total_outpatient_count: int, weather: str) -> pd.DataFrame:
    arr_bst, arr_cols, svc_bst, wait_bst, wp90_bst, multi_cols, baseline, calib = load_assets()

    is_h = is_holiday(target_date)
    prev = target_date - timedelta(days=1)
    is_prev_h = is_holiday(prev)

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min").to_pydatetime().tolist()
    slots = [t for t in slots if t.time() != end.time()]  # 18:00除外

    # state
    queue_start = 0.0

    # lags
    arr_lag_30=arr_lag_60=arr_lag_90=0.0
    svc_lag_30=svc_lag_60=svc_lag_90=0.0

    wcat = normalize_weather(weather)

    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    alpha_base = float(calib.get("alpha_base", 0.60))
    alpha_peak = float(calib.get("alpha_peak", 0.25))

    results = []
    for ts in slots:
        key = baseline_key(ts)

        # ---------- arrivals (log1p) ----------
        af = _make_zero_df(arr_cols)
        for col, val in [
            ("hour", ts.hour), ("minute", ts.minute),
            ("月", ts.month),
            ("週回数", int((ts.day - 1)//7 + 1)),
            ("前日祝日フラグ", int(is_prev_h)),
            ("total_outpatient_count", int(total_outpatient_count)),
            ("is_holiday", int(is_h)),
        ]:
            if col in af.columns:
                af.loc[0, col] = val

        dc = f"dayofweek_{ts.weekday()}"
        if dc in af.columns: af.loc[0, dc] = 1
        wc = f"天気カテゴリ_{wcat}"
        if wc in af.columns: af.loc[0, wc] = 1
        if "雨フラグ" in af.columns: af.loc[0, "雨フラグ"] = 1 if wcat=="雨" else 0
        if "雪フラグ" in af.columns: af.loc[0, "雪フラグ"] = 1 if wcat=="雪" else 0

        for col, val in [
            ("arr_lag_30", arr_lag_30), ("arr_lag_60", arr_lag_60), ("arr_lag_90", arr_lag_90),
            ("arr_roll_60", (arr_lag_30+arr_lag_60)/2.0),
            ("queue_at_start_truth", queue_start),
            ("queue_at_start_of_slot", queue_start),
        ]:
            if col in af.columns:
                af.loc[0, col] = val

        arr_log = _predict_booster(arr_bst, arr_cols, af)
        arr_pred = max(0.0, float(np.expm1(arr_log)))

        arr_med = get_base(baseline, "arr", key, "median", default=arr_pred)
        # arrivalsは軽くbaseline寄せ（過学習のブレ対策）
        arr = 0.85*arr_pred + 0.15*arr_med

        # ---------- multi features ----------
        mf = _make_zero_df(multi_cols)
        for col, val in [
            ("hour", ts.hour), ("minute", ts.minute),
            ("月", ts.month),
            ("週回数", int((ts.day - 1)//7 + 1)),
            ("前日祝日フラグ", int(is_prev_h)),
            ("total_outpatient_count", int(total_outpatient_count)),
            ("is_holiday", int(is_h)),
            ("reception_count", arr),
            ("queue_at_start_truth", queue_start),
            ("queue_at_start_of_slot", queue_start),
            ("svc_lag_30", svc_lag_30), ("svc_lag_60", svc_lag_60), ("svc_lag_90", svc_lag_90),
            ("svc_roll_60", (svc_lag_30+svc_lag_60)/2.0),
        ]:
            if col in mf.columns:
                mf.loc[0, col] = val

        dc2 = f"dayofweek_{ts.weekday()}"
        if dc2 in mf.columns: mf.loc[0, dc2] = 1
        wc2 = f"天気カテゴリ_{wcat}"
        if wc2 in mf.columns: mf.loc[0, wc2] = 1
        if "雨フラグ" in mf.columns: mf.loc[0, "雨フラグ"] = 1 if wcat=="雨" else 0
        if "雪フラグ" in mf.columns: mf.loc[0, "雪フラグ"] = 1 if wcat=="雪" else 0

        # ---------- service (RESIDUAL, then restore) ----------
        svc_res = _predict_booster(svc_bst, multi_cols, mf)  # residual in log-space
        svc_base_med = get_base(baseline, "svc", key, "median", default=0.0)
        svc_log = float(np.log1p(max(0.0, svc_base_med)) + svc_res)
        svc_pred = max(0.0, float(np.expm1(svc_log)))

        # 崩壊防止：baselineへ強く寄せ、さらにp95で上限
        svc_p95 = get_base(baseline, "svc", key, "p95", default=max(svc_base_med, svc_pred))
        svc_p05 = max(0.0, 0.6*svc_base_med)

        svc = 0.35*svc_pred + 0.65*svc_base_med
        svc = clip(svc, svc_p05, svc_p95)

        # 物理上限：その枠で処理できるのは「今いる+入る」まで
        svc = min(svc, queue_start + arr)

        # ---------- queue update ----------
        queue_end = max(0.0, queue_start + arr - svc)

        # ---------- wait mean/p90 ----------
        wm_log = _predict_booster(wait_bst, multi_cols, mf)
        wp_log = _predict_booster(wp90_bst, multi_cols, mf)
        wait_model = max(0.0, float(np.expm1(wm_log)))
        waitp90_model = max(0.0, float(np.expm1(wp_log)))

        # physics wait (calibrated)
        wait_phy = (queue_start + 0.5*arr) / max(svc, 1.0) * SLOT_MINUTES
        wait_phy = a*wait_phy + b

        alpha = alpha_peak if in_peak(ts) else alpha_base
        wait_med = get_base(baseline, "wait_mean", key, "median", default=wait_model)
        wait_p95 = get_base(baseline, "wait_mean", key, "p95", default=max(wait_model, wait_phy))

        wait_mean = alpha*wait_model + (1-alpha)*wait_phy
        wait_mean = 0.85*wait_mean + 0.15*wait_med
        wait_mean = clip(wait_mean, 0.0, wait_p95)

        wp95 = get_base(baseline, "wait_p90", key, "p95", default=max(waitp90_model, wait_mean))
        wait_p90 = 0.70*waitp90_model + 0.30*max(wait_phy, wait_mean)
        wait_p90 = clip(wait_p90, wait_mean, wp95)

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(round(arr)),
            "予測処理数(呼出数)": int(round(svc)),
            "予測待ち人数_開始(人)": int(round(queue_start)),
            "予測待ち人数_終了(人)": int(round(queue_end)),
            "予測平均待ち時間(分)": int(round(wait_mean)),
            "予測混雑時待ち時間_p90(分)": int(round(wait_p90)),
        })

        # update state
        arr_lag_90, arr_lag_60, arr_lag_30 = arr_lag_60, arr_lag_30, arr
        svc_lag_90, svc_lag_60, svc_lag_30 = svc_lag_60, svc_lag_30, svc
        queue_start = queue_end

    return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="A病院 採血 待ち人数・待ち時間 予測", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（最終版）")
    st.caption("service残差モデル + 保存則キュー + wait(モデル/物理/baselineブレンド)")

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気（簡易）", ["晴", "曇", "雨", "雪"], index=0)
        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.subheader("必要ファイル（models/）")
        st.write("- model_A_timeseries.json")
        st.write("- columns_A_timeseries.json")
        st.write("- model_A_service_30min.json")
        st.write("- model_A_waittime_30min.json")
        st.write("- model_A_waittime_p90_30min.json")
        st.write("- columns_A_multi_30min.json")
        st.write("- baseline_tables_mds.json")
        st.write("- wait_calibration.json")

    required = [
        ARR_MODEL_PATH, ARR_COLS_PATH,
        SVC_MODEL_PATH,
        WAIT_MODEL_PATH, WAITP90_MODEL_PATH,
        MULTI_COLS_PATH,
        BASELINE_PATH, CALIB_PATH
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("models/ に必要ファイルが不足しています:\n\n" + "\n".join(missing))
        st.stop()

    if run:
        with st.spinner("計算中..."):
            df = simulate_one_day(target, int(total_out), str(weather))
        st.success(f"{target} の予測が完了しました。")

        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.subheader("結果テーブル")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("CSVダウンロード", data=csv, file_name=f"A_predict_{target}.csv", mime="text/csv")

        with c2:
            st.subheader("可視化")
            st.line_chart(df.set_index("時間帯")[["予測平均待ち時間(分)", "予測混雑時待ち時間_p90(分)"]])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数_開始(人)"]])

if __name__ == "__main__":
    main()
