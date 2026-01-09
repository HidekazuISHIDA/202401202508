import json
from pathlib import Path
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

# -------------------------
# Required files
# -------------------------
ARR_MODEL_PATH  = MODELS_DIR / "model_A_timeseries.json"
ARR_COLS_PATH   = MODELS_DIR / "columns_A_timeseries.json"

SVC_MODEL_PATH  = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH = MODELS_DIR / "model_A_waittime_30min.json"
MULTI_COLS_PATH = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH   = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH      = MODELS_DIR / "wait_calibration.json"

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
INCLUDE_CLOSE = False  # 18:00枠は除外（08:00〜17:30）

WEATHER_CATS = ["晴", "曇", "雨", "雪"]

def _slot_id(ts: datetime) -> int:
    minutes = ts.hour * 60 + ts.minute
    return int((minutes - 8 * 60) // 30)

def _baseline_key(ts: datetime) -> str:
    # month_dayofweek_slot
    return f"{ts.month}_{ts.weekday()}_{_slot_id(ts)}"

def _normalize_weather(w: str) -> str:
    s = str(w) if w is not None else ""
    if "雪" in s: return "雪"
    if "雨" in s: return "雨"
    if "曇" in s: return "曇"
    if "晴" in s: return "晴"
    return "曇"

def _is_holiday_like(d: date) -> bool:
    # Streamlit側は追加入力なしの方針なので最低限
    if d.weekday() >= 5:
        return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return True
    return False

def _week_of_month(d: date) -> int:
    return int((d.day - 1) // 7 + 1)

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "O":
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(0)
    return out

def _predict_log1p(booster: xgb.Booster, cols, row_df: pd.DataFrame) -> float:
    X = _safe_numeric_df(row_df[cols])
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

@st.cache_resource
def load_artifacts():
    # columns
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    # models
    arr_booster = xgb.Booster()
    arr_booster.load_model(str(ARR_MODEL_PATH))

    svc_booster = xgb.Booster()
    svc_booster.load_model(str(SVC_MODEL_PATH))

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAIT_MODEL_PATH))

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))

    return arr_booster, arr_cols, svc_booster, wait_booster, multi_cols, baseline, calib

def simulate_one_day(target_date: date, total_outpatient_count: int, weather: str) -> pd.DataFrame:
    arr_booster, arr_cols, svc_booster, wait_booster, multi_cols, baseline, calib = load_artifacts()

    is_h = int(_is_holiday_like(target_date))
    prev = target_date - timedelta(days=1)
    is_prev_h = int(_is_holiday_like(prev))

    # time slots
    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    time_slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")

    if not INCLUDE_CLOSE:
        time_slots = [t for t in time_slots if not (t.hour == CLOSE_HOUR and t.minute == 0)]

    # state
    queue = 0.0
    lags_arr = [0.0, 0.0, 0.0]  # 30/60/90
    lags_svc = [0.0, 0.0, 0.0]
    cum_arr = 0.0
    cum_svc = 0.0

    wcat = _normalize_weather(weather)
    results = []

    for ts in time_slots:
        ts_dt = ts.to_pydatetime()
        key = _baseline_key(ts_dt)

        # baseline medians
        arr_base = float(baseline.get("arr_median", {}).get(key, 0.0))
        svc_base = float(baseline.get("svc_median", {}).get(key, 0.0))
        wait_base = float(baseline.get("wait_median", {}).get(key, 0.0))

        # ----- arrivals features -----
        af = _make_zero_df(arr_cols)
        if "hour" in af.columns: af.loc[0, "hour"] = ts_dt.hour
        if "minute" in af.columns: af.loc[0, "minute"] = ts_dt.minute
        if "月" in af.columns: af.loc[0, "月"] = ts_dt.month
        if "週回数" in af.columns: af.loc[0, "週回数"] = _week_of_month(target_date)
        if "is_holiday" in af.columns: af.loc[0, "is_holiday"] = is_h
        if "前日祝日フラグ" in af.columns: af.loc[0, "前日祝日フラグ"] = is_prev_h
        if "total_outpatient_count" in af.columns: af.loc[0, "total_outpatient_count"] = int(total_outpatient_count)

        # weather flags
        if "雨フラグ" in af.columns: af.loc[0, "雨フラグ"] = 1 if wcat == "雨" else 0
        if "雪フラグ" in af.columns: af.loc[0, "雪フラグ"] = 1 if wcat == "雪" else 0
        for cat in WEATHER_CATS:
            col = f"天気カテゴリ_{cat}"
            if col in af.columns:
                af.loc[0, col] = 1 if wcat == cat else 0

        # dow one-hot
        dow = ts_dt.weekday()
        dcol = f"dayofweek_{dow}"
        if dcol in af.columns: af.loc[0, dcol] = 1

        # slot flags
        if "is_first_slot" in af.columns:  af.loc[0, "is_first_slot"] = 1 if (ts_dt.hour==8 and ts_dt.minute==0) else 0
        if "is_second_slot" in af.columns: af.loc[0, "is_second_slot"] = 1 if (ts_dt.hour==8 and ts_dt.minute==30) else 0

        # lags / rolling / cumulative / baseline / queue at start
        if "arr_lag_30" in af.columns: af.loc[0, "arr_lag_30"] = lags_arr[0]
        if "arr_lag_60" in af.columns: af.loc[0, "arr_lag_60"] = lags_arr[1]
        if "arr_lag_90" in af.columns: af.loc[0, "arr_lag_90"] = lags_arr[2]
        if "arr_roll_60" in af.columns: af.loc[0, "arr_roll_60"] = (lags_arr[0] + lags_arr[1]) / 2.0
        if "cum_arrivals" in af.columns: af.loc[0, "cum_arrivals"] = cum_arr
        if "queue_at_start_truth" in af.columns: af.loc[0, "queue_at_start_truth"] = queue  # training nameを流用
        if "arr_base" in af.columns: af.loc[0, "arr_base"] = arr_base
        if "svc_base" in af.columns: af.loc[0, "svc_base"] = svc_base
        if "wait_base" in af.columns: af.loc[0, "wait_base"] = wait_base

        pred_log_arr = _predict_log1p(arr_booster, arr_cols, af)
        pred_arr = max(0.0, float(np.expm1(pred_log_arr)))
        # baseline anchor（極端な暴走防止）：基準の1/3〜3倍に軽く制限
        pred_arr = float(np.clip(pred_arr, arr_base*0.33, max(arr_base*3.0, 5.0)))
        pred_arr_i = int(round(pred_arr))

        # ----- service features (multi) -----
        mf = _make_zero_df(multi_cols)
        if "hour" in mf.columns: mf.loc[0, "hour"] = ts_dt.hour
        if "minute" in mf.columns: mf.loc[0, "minute"] = ts_dt.minute
        if "月" in mf.columns: mf.loc[0, "月"] = ts_dt.month
        if "週回数" in mf.columns: mf.loc[0, "週回数"] = _week_of_month(target_date)
        if "is_holiday" in mf.columns: mf.loc[0, "is_holiday"] = is_h
        if "前日祝日フラグ" in mf.columns: mf.loc[0, "前日祝日フラグ"] = is_prev_h
        if "total_outpatient_count" in mf.columns: mf.loc[0, "total_outpatient_count"] = int(total_outpatient_count)

        if "雨フラグ" in mf.columns: mf.loc[0, "雨フラグ"] = 1 if wcat == "雨" else 0
        if "雪フラグ" in mf.columns: mf.loc[0, "雪フラグ"] = 1 if wcat == "雪" else 0
        for cat in WEATHER_CATS:
            col = f"天気カテゴリ_{cat}"
            if col in mf.columns:
                mf.loc[0, col] = 1 if wcat == cat else 0

        dcol2 = f"dayofweek_{dow}"
        if dcol2 in mf.columns: mf.loc[0, dcol2] = 1

        if "is_first_slot" in mf.columns:  mf.loc[0, "is_first_slot"] = 1 if (ts_dt.hour==8 and ts_dt.minute==0) else 0
        if "is_second_slot" in mf.columns: mf.loc[0, "is_second_slot"] = 1 if (ts_dt.hour==8 and ts_dt.minute==30) else 0

        # states (queue/cum/lag) + baseline
        if "queue_at_start_truth" in mf.columns: mf.loc[0, "queue_at_start_truth"] = queue
        if "cum_arrivals" in mf.columns: mf.loc[0, "cum_arrivals"] = cum_arr
        if "cum_service" in mf.columns: mf.loc[0, "cum_service"] = cum_svc

        if "arr_lag_30" in mf.columns: mf.loc[0, "arr_lag_30"] = lags_arr[0]
        if "arr_lag_60" in mf.columns: mf.loc[0, "arr_lag_60"] = lags_arr[1]
        if "arr_lag_90" in mf.columns: mf.loc[0, "arr_lag_90"] = lags_arr[2]
        if "arr_roll_60" in mf.columns: mf.loc[0, "arr_roll_60"] = (lags_arr[0] + lags_arr[1]) / 2.0

        if "svc_lag_30" in mf.columns: mf.loc[0, "svc_lag_30"] = lags_svc[0]
        if "svc_lag_60" in mf.columns: mf.loc[0, "svc_lag_60"] = lags_svc[1]
        if "svc_lag_90" in mf.columns: mf.loc[0, "svc_lag_90"] = lags_svc[2]
        if "svc_roll_60" in mf.columns: mf.loc[0, "svc_roll_60"] = (lags_svc[0] + lags_svc[1]) / 2.0

        if "arr_base" in mf.columns: mf.loc[0, "arr_base"] = arr_base
        if "svc_base" in mf.columns: mf.loc[0, "svc_base"] = svc_base
        if "wait_base" in mf.columns: mf.loc[0, "wait_base"] = wait_base

        # optional: current predicted arrivals can be a feature
        if "reception_count" in mf.columns: mf.loc[0, "reception_count"] = pred_arr_i

        pred_log_svc = _predict_log1p(svc_booster, multi_cols, mf)
        pred_svc = max(0.0, float(np.expm1(pred_log_svc)))
        pred_svc = float(np.clip(pred_svc, svc_base*0.33, max(svc_base*3.0, 5.0)))
        pred_svc_i = int(round(pred_svc))

        # ----- queue update (conservation) -----
        queue_next = max(0.0, queue + pred_arr_i - pred_svc_i)

        # ----- wait prediction -----
        # ML wait
        if "reception_count" in mf.columns:
            mf.loc[0, "reception_count"] = pred_arr_i
        if "call_count" in mf.columns:
            mf.loc[0, "call_count"] = pred_svc_i

        pred_log_wait = _predict_log1p(wait_booster, multi_cols, mf)
        wait_ml = max(0.0, float(np.expm1(pred_log_wait)))

        # physics (queue/service) with calibration
        a = float(calib.get("a", 1.0))
        b = float(calib.get("b", 0.0))
        alpha = float(calib.get("alpha", 0.65))
        floor_ratio = float(calib.get("floor_ratio", 0.70))

        svc_eff = max(1.0, float(pred_svc_i))
        wait_phy_raw = (float(queue) + 0.5*float(pred_arr_i)) / svc_eff * 30.0
        wait_phy = max(0.0, a*wait_phy_raw + b)

        # blend + floor
        wait_blend = alpha*wait_ml + (1.0-alpha)*wait_phy
        wait_floor = floor_ratio * wait_phy
        wait_pred = max(wait_blend, wait_floor)

        results.append({
            "時間帯": ts_dt.strftime("%H:%M"),
            "予測受付数": int(pred_arr_i),
            "予測呼出数": int(pred_svc_i),
            "予測待ち人数(人)": int(round(queue_next)),
            "予測平均待ち時間(分)": int(round(wait_pred)),
        })

        # update states
        queue = queue_next
        cum_arr += pred_arr_i
        cum_svc += pred_svc_i

        lags_arr = [float(pred_arr_i), lags_arr[0], lags_arr[1]]
        lags_svc = [float(pred_svc_i), lags_svc[0], lags_svc[1]]

    return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="A病院 採血 待ち人数・待ち時間 予測（最終版）", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（最終版・壊れない構造）")
    st.caption("曜日×時間帯のベースライン＋保存則キュー＋wait物理下限（短すぎ防止）")

    required = [
        ARR_MODEL_PATH, ARR_COLS_PATH,
        SVC_MODEL_PATH, WAIT_MODEL_PATH, MULTI_COLS_PATH,
        BASELINE_PATH, CALIB_PATH
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("models/ に必要ファイルが不足しています:\n\n" + "\n".join(missing))
        st.stop()

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気（簡易）", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=0)
        run = st.button("予測実行", type="primary")

        st.divider()
        st.subheader("読込ファイル")
        st.write("arrivals:", ARR_MODEL_PATH.name)
        st.write("service :", SVC_MODEL_PATH.name)
        st.write("wait    :", WAIT_MODEL_PATH.name)
        st.write("baseline:", BASELINE_PATH.name)
        st.write("calib   :", CALIB_PATH.name)

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
            st.line_chart(df.set_index("時間帯")[["予測平均待ち時間(分)"]])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

    st.divider()
    st.caption("※ 本アプリは“短すぎ崩壊”を避けるため、待ち時間に物理下限（queue/service）を適用しています。")

if __name__ == "__main__":
    main()
