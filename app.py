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

ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.json"
SVC_MODEL_PATH   = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"
WAITP90_MODEL_PATH = MODELS_DIR / "model_A_waittime_p90_30min.json"

ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH    = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
SLOT_MINUTES = 30.0

WEATHER_CATS = ["晴","曇","雨","雪"]

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

    arr_bst = xgb.Booster()
    arr_bst.load_model(str(ARR_MODEL_PATH))

    svc_bst = xgb.Booster()
    svc_bst.load_model(str(SVC_MODEL_PATH))

    wait_bst = xgb.Booster()
    wait_bst.load_model(str(WAIT_MODEL_PATH))

    waitp90_bst = xgb.Booster()
    waitp90_bst.load_model(str(WAITP90_MODEL_PATH))

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))

    return arr_bst, arr_cols, svc_bst, wait_bst, waitp90_bst, multi_cols, baseline, calib

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _predict_booster(booster: xgb.Booster, cols, df: pd.DataFrame) -> float:
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

def baseline_key(ts: datetime):
    slot_id = int((((ts.hour*60 + ts.minute) - (8*60)) // 30))
    return f"{ts.month}_{ts.weekday()}_{slot_id}"

def get_baseline(baseline, kind: str, key: str, stat="median", default=0.0):
    try:
        return float(baseline.get(kind, {}).get(key, {}).get(stat, default))
    except Exception:
        return float(default)

def clip_by_baseline(x: float, baseline, kind: str, key: str, lo=0.0, hi_stat="p95"):
    hi = get_baseline(baseline, kind, key, stat=hi_stat, default=max(1.0, x))
    return float(np.clip(x, lo, hi))

def simulate_one_day(target_date: date, total_outpatient_count: int, weather: str) -> pd.DataFrame:
    arr_bst, arr_cols, svc_bst, wait_bst, waitp90_bst, multi_cols, baseline, calib = load_assets()

    is_h = is_holiday(target_date)
    prev = target_date - timedelta(days=1)
    is_prev_h = is_holiday(prev)

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    time_slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")
    time_slots = [t.to_pydatetime() for t in time_slots if t.to_pydatetime().time() != end.time()]  # 18:00除外

    # state
    queue_start = 0.0
    # lags
    arr_lag_30 = 0.0
    arr_lag_60 = 0.0
    arr_lag_90 = 0.0
    svc_lag_30 = 0.0
    svc_lag_60 = 0.0
    svc_lag_90 = 0.0

    results = []
    wcat = normalize_weather(weather)

    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    alpha_base = float(calib.get("alpha_base", 0.60))
    alpha_peak = float(calib.get("alpha_peak", 0.25))

    for ts in time_slots:
        key = baseline_key(ts)

        # ---------- arrivals features ----------
        af = _make_zero_df(arr_cols)
        if "hour" in af.columns: af.loc[0, "hour"] = ts.hour
        if "minute" in af.columns: af.loc[0, "minute"] = ts.minute
        if "月" in af.columns: af.loc[0, "月"] = ts.month
        if "週回数" in af.columns: af.loc[0, "週回数"] = int((ts.day - 1)//7 + 1)
        if "前日祝日フラグ" in af.columns: af.loc[0, "前日祝日フラグ"] = int(is_prev_h)
        if "total_outpatient_count" in af.columns: af.loc[0, "total_outpatient_count"] = int(total_outpatient_count)
        if "is_holiday" in af.columns: af.loc[0, "is_holiday"] = int(is_h)

        if "雨フラグ" in af.columns: af.loc[0, "雨フラグ"] = 1 if wcat=="雨" else 0
        if "雪フラグ" in af.columns: af.loc[0, "雪フラグ"] = 1 if wcat=="雪" else 0
        wc = f"天気カテゴリ_{wcat}"
        if wc in af.columns: af.loc[0, wc] = 1

        dc = f"dayofweek_{ts.weekday()}"
        if dc in af.columns: af.loc[0, dc] = 1

        if "arr_lag_30" in af.columns: af.loc[0, "arr_lag_30"] = arr_lag_30
        if "arr_lag_60" in af.columns: af.loc[0, "arr_lag_60"] = arr_lag_60
        if "arr_lag_90" in af.columns: af.loc[0, "arr_lag_90"] = arr_lag_90
        if "arr_roll_60" in af.columns: af.loc[0, "arr_roll_60"] = (arr_lag_30 + arr_lag_60)/2.0

        # queue_start_truthは学習で使っているので、appでは state を投入
        if "queue_at_start_truth" in af.columns: af.loc[0, "queue_at_start_truth"] = queue_start
        if "cum_arrivals_sofar" in af.columns:
            # ここは「予測ベース」累積（追加入力なし）
            # ただし学習時と同じ列名を満たすための近似
            # （列が無ければ何もしない）
            pass

        pred_arr = _predict_booster(arr_bst, arr_cols, af)
        arr = max(0.0, float(pred_arr))
        # baselineで弱く安定化（極端な日を抑えたい場合）
        arr_base = get_baseline(baseline, "arr", key, "median", default=arr)
        arr = 0.85*arr + 0.15*arr_base

        # ---------- multi features for service/wait ----------
        mf = _make_zero_df(multi_cols)
        for col, val in [
            ("hour", ts.hour),
            ("minute", ts.minute),
            ("月", ts.month),
            ("週回数", int((ts.day - 1)//7 + 1)),
            ("前日祝日フラグ", int(is_prev_h)),
            ("total_outpatient_count", int(total_outpatient_count)),
            ("is_holiday", int(is_h)),
        ]:
            if col in mf.columns:
                mf.loc[0, col] = val

        if "雨フラグ" in mf.columns: mf.loc[0, "雨フラグ"] = 1 if wcat=="雨" else 0
        if "雪フラグ" in mf.columns: mf.loc[0, "雪フラグ"] = 1 if wcat=="雪" else 0
        wc2 = f"天気カテゴリ_{wcat}"
        if wc2 in mf.columns: mf.loc[0, wc2] = 1
        dc2 = f"dayofweek_{ts.weekday()}"
        if dc2 in mf.columns: mf.loc[0, dc2] = 1

        # state/inputs
        if "queue_at_start_truth" in mf.columns: mf.loc[0, "queue_at_start_truth"] = queue_start
        if "queue_at_start_of_slot" in mf.columns: mf.loc[0, "queue_at_start_of_slot"] = queue_start
        if "reception_count" in mf.columns: mf.loc[0, "reception_count"] = arr  # 予測受付数を投入

        # lags
        if "svc_lag_30" in mf.columns: mf.loc[0, "svc_lag_30"] = svc_lag_30
        if "svc_lag_60" in mf.columns: mf.loc[0, "svc_lag_60"] = svc_lag_60
        if "svc_lag_90" in mf.columns: mf.loc[0, "svc_lag_90"] = svc_lag_90
        if "svc_roll_60" in mf.columns: mf.loc[0, "svc_roll_60"] = (svc_lag_30 + svc_lag_60)/2.0

        # ---------- service prediction (log1p) ----------
        svc_log = _predict_booster(svc_bst, multi_cols, mf)
        svc = float(np.expm1(svc_log))
        svc = max(0.0, svc)

        # clip service by baseline p95（過大処理→待ち消失を防止）
        svc = clip_by_baseline(svc, baseline, "svc", key, lo=0.0, hi_stat="p95")

        # 物理的上限：処理は「今いる＋入ってくる」以上は起きにくい
        svc = min(svc, queue_start + arr)

        # ---------- queue update (conservation) ----------
        queue_end = max(0.0, queue_start + arr - svc)

        # ---------- wait: model (log1p -> minutes) ----------
        wm_log = _predict_booster(wait_bst, multi_cols, mf)
        wp_log = _predict_booster(waitp90_bst, multi_cols, mf)
        wait_model = float(np.expm1(wm_log))
        wait_p90_model = float(np.expm1(wp_log))

        # ---------- wait: physics (KEY FIX: +0.5*arrivals) ----------
        wait_phy = (queue_start + 0.5*arr) / max(svc, 1.0) * SLOT_MINUTES
        wait_phy_cal = a*wait_phy + b

        # ---------- blend weights ----------
        alpha = alpha_peak if in_peak(ts) else alpha_base

        # ---------- baseline guard ----------
        wait_base_median = get_baseline(baseline, "wait_mean", key, "median", default=wait_model)
        wait_base_p95    = get_baseline(baseline, "wait_mean", key, "p95", default=max(wait_model, wait_phy_cal))

        # mean wait: blend(model, physics) then gently pull toward baseline median
        wait_mean = alpha*wait_model + (1.0-alpha)*wait_phy_cal
        wait_mean = 0.85*wait_mean + 0.15*wait_base_median
        wait_mean = float(np.clip(wait_mean, 0.0, wait_base_p95))  # 上限はp95で暴れ防止

        # p90 wait: model主だが、物理を下限として混雑時に落ちすぎないように
        wait_p90 = 0.70*wait_p90_model + 0.30*max(wait_phy_cal, wait_mean)
        wait_p90_base = get_baseline(baseline, "wait_p90", key, "p95", default=max(wait_p90, wait_mean))
        wait_p90 = float(np.clip(wait_p90, wait_mean, wait_p90_base))

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(round(arr)),
            "予測処理数(呼出数)": int(round(svc)),
            "予測待ち人数(人)": int(round(queue_start)),
            "予測平均待ち時間(分)": int(round(wait_mean)),
            "予測混雑時待ち時間_p90(分)": int(round(wait_p90)),
        })

        # update lags/state for next slot
        arr_lag_90, arr_lag_60, arr_lag_30 = arr_lag_60, arr_lag_30, arr
        svc_lag_90, svc_lag_60, svc_lag_30 = svc_lag_60, svc_lag_30, svc
        queue_start = queue_end

    return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="A病院 採血 待ち人数・待ち時間 予測", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（最終版：arrivals + service + wait mean/p90 + physics補正）")

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

    missing = []
    required = [
        ARR_MODEL_PATH, SVC_MODEL_PATH, WAIT_MODEL_PATH, WAITP90_MODEL_PATH,
        ARR_COLS_PATH, MULTI_COLS_PATH, BASELINE_PATH, CALIB_PATH
    ]
    for p in required:
        if not p.exists():
            missing.append(p.name)
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
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

    st.divider()
    st.caption("※ 祝日判定は data/syukujitsu.csv を参照（なければ土日・年末年始のみ）")

if __name__ == "__main__":
    main()
