import json
from pathlib import Path
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# =========================================================
# Paths
# =========================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.json"
ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"

SVC_MODEL_PATH   = MODELS_DIR / "model_A_service_30min.json"       # ★ service model
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"

MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH    = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"   # optional

# =========================================================
# Simulation config
# =========================================================
OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
SLOT_MIN = 30.0

# wait clip (safety)
WAIT_MAX = 180.0

# baseline blending (no extra user input)
# - For early slots, rely more on baseline
BASELINE_BLEND_EARLY = 0.45   # model weight in early slots
BASELINE_BLEND_NORMAL = 0.75  # model weight in normal slots
EARLY_SLOT_MAX = 4            # 08:00,08:30,09:00,09:30 -> slot_id 0..3

# service minimum to avoid crazy physics wait
SERVICE_MIN = 1.0

# =========================================================
# Holiday
# =========================================================
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

def week_of_month(d: date) -> int:
    return int((d.day - 1) // 7 + 1)

# =========================================================
# Weather normalize
# =========================================================
WEATHER_CATS = ["晴", "曇", "雨", "雪"]

def normalize_weather(w: str) -> str:
    s = str(w) if w is not None else ""
    if "雪" in s:
        return "雪"
    if "雨" in s:
        return "雨"
    if "曇" in s:
        return "曇"
    if "晴" in s:
        return "晴"
    return "曇"

# =========================================================
# Model loading
# =========================================================
@st.cache_resource
def load_assets():
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    booster_arr = xgb.Booster()
    booster_arr.load_model(str(ARR_MODEL_PATH))

    booster_svc = xgb.Booster()
    booster_svc.load_model(str(SVC_MODEL_PATH))

    booster_wait = xgb.Booster()
    booster_wait.load_model(str(WAIT_MODEL_PATH))

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))

    # calib keys: a, b, blend_alpha
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    blend_alpha = float(calib.get("blend_alpha", 0.65))

    return booster_arr, arr_cols, booster_svc, booster_wait, multi_cols, baseline, (a, b, blend_alpha)

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce all columns to numeric where possible (safety)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def _predict_booster(booster: xgb.Booster, cols, df: pd.DataFrame) -> float:
    X = df[cols].copy()
    X = _coerce_numeric(X)
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

# =========================================================
# Baseline table lookup
# =========================================================
def baseline_key(month: int, dow: int, slot_id: int) -> str:
    return f"{int(month)}_{int(dow)}_{int(slot_id)}"

def get_baseline(baseline_tables: dict, month: int, dow: int, slot_id: int):
    key = baseline_key(month, dow, slot_id)
    arr = baseline_tables.get("arr", {}).get(key, {})
    svc = baseline_tables.get("svc", {}).get(key, {})
    wat = baseline_tables.get("wait", {}).get(key, {})
    # median fallback 0
    return (
        float(arr.get("median", 0.0)),
        float(svc.get("median", 0.0)),
        float(wat.get("median", 0.0)),
    )

def blend_with_baseline(model_value: float, base_value: float, slot_id: int) -> float:
    # Early slots rely more on baseline
    if slot_id < EARLY_SLOT_MAX:
        w_model = BASELINE_BLEND_EARLY
    else:
        w_model = BASELINE_BLEND_NORMAL
    return w_model * float(model_value) + (1.0 - w_model) * float(base_value)

# =========================================================
# Feature builders
# =========================================================
def add_common_time_features(dfrow: pd.DataFrame, ts: datetime, target_date: date):
    d = target_date
    dow = ts.weekday()
    month = ts.month
    minute = ts.minute
    hour = ts.hour
    slot_id = int(((hour * 60 + minute) - (OPEN_HOUR * 60)) // FREQ_MIN)

    # Some training features
    if "hour" in dfrow.columns: dfrow.loc[0, "hour"] = int(hour)
    if "minute" in dfrow.columns: dfrow.loc[0, "minute"] = int(minute)
    if "dow" in dfrow.columns: dfrow.loc[0, "dow"] = int(dow)
    if "month" in dfrow.columns: dfrow.loc[0, "month"] = int(month)
    if "slot_id" in dfrow.columns: dfrow.loc[0, "slot_id"] = int(slot_id)

    # Japanese calendar columns (may exist)
    if "月" in dfrow.columns: dfrow.loc[0, "月"] = int(month)
    if "週回数" in dfrow.columns: dfrow.loc[0, "週回数"] = int(week_of_month(d))

    # one-hot dayofweek_*
    for k in range(7):
        c = f"dayofweek_{k}"
        if c in dfrow.columns:
            dfrow.loc[0, c] = 1 if k == dow else 0

    # early slot flags
    if "is_first_slot" in dfrow.columns:
        dfrow.loc[0, "is_first_slot"] = 1 if (hour == 8 and minute == 0) else 0
    if "is_second_slot" in dfrow.columns:
        dfrow.loc[0, "is_second_slot"] = 1 if (hour == 8 and minute == 30) else 0

    return slot_id, month, dow

def add_calendar_features(dfrow: pd.DataFrame, target_date: date):
    is_h = is_holiday(target_date)
    prev = target_date - timedelta(days=1)
    is_prev_h = is_holiday(prev)

    if "is_holiday" in dfrow.columns: dfrow.loc[0, "is_holiday"] = int(is_h)
    if "前日祝日フラグ" in dfrow.columns: dfrow.loc[0, "前日祝日フラグ"] = int(is_prev_h)

def add_weather_features(dfrow: pd.DataFrame, weather_label: str):
    w = normalize_weather(weather_label)
    # flags
    if "雨フラグ" in dfrow.columns: dfrow.loc[0, "雨フラグ"] = 1 if w == "雨" else 0
    if "雪フラグ" in dfrow.columns: dfrow.loc[0, "雪フラグ"] = 1 if w == "雪" else 0

    # category dummies
    for cat in WEATHER_CATS:
        c = f"天気カテゴリ_{cat}"
        if c in dfrow.columns:
            dfrow.loc[0, c] = 1 if w == cat else 0

    # numeric weather columns (not provided by user):
    # We keep them as 0; model is trained with them, but user doesn't want extra input.
    for c in ["降水量", "平均気温", "最高気温", "最低気温", "平均湿度", "平均風速"]:
        if c in dfrow.columns:
            dfrow.loc[0, c] = 0.0

def add_outpatient(dfrow: pd.DataFrame, total_outpatient_count: int):
    if "total_outpatient_count" in dfrow.columns:
        dfrow.loc[0, "total_outpatient_count"] = int(total_outpatient_count)

def add_lag_features_arr(dfrow: pd.DataFrame, arr_lags: dict):
    # arr_lags keys: arr_lag_30, arr_lag_60, arr_lag_90
    for k, v in arr_lags.items():
        if k in dfrow.columns:
            dfrow.loc[0, k] = float(v)
    if "arr_roll_60" in dfrow.columns:
        dfrow.loc[0, "arr_roll_60"] = (float(arr_lags["arr_lag_30"]) + float(arr_lags["arr_lag_60"])) / 2.0

def add_lag_features_svc(dfrow: pd.DataFrame, svc_lags: dict):
    for k, v in svc_lags.items():
        if k in dfrow.columns:
            dfrow.loc[0, k] = float(v)
    if "svc_roll_60" in dfrow.columns:
        dfrow.loc[0, "svc_roll_60"] = (float(svc_lags["svc_lag_30"]) + float(svc_lags["svc_lag_60"])) / 2.0

def add_queue_feature(dfrow: pd.DataFrame, queue_at_start: float):
    # training used queue_at_start_truth; we provide predicted queue here
    if "queue_at_start_truth" in dfrow.columns:
        dfrow.loc[0, "queue_at_start_truth"] = float(queue_at_start)

# =========================================================
# Simulation (one day)
# =========================================================
def simulate_one_day(target_date: date, total_outpatient_count: int, weather: str) -> pd.DataFrame:
    booster_arr, arr_cols, booster_svc, booster_wait, multi_cols, baseline, calib = load_assets()
    a, b, blend_alpha = calib

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    time_slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")

    # Drop 18:00 slot (train did 08:00-17:30)
    if len(time_slots) > 0 and time_slots[-1].hour == CLOSE_HOUR and time_slots[-1].minute == 0:
        time_slots = time_slots[:-1]

    # state
    arr_lags = {"arr_lag_30": 0.0, "arr_lag_60": 0.0, "arr_lag_90": 0.0}
    svc_lags = {"svc_lag_30": 0.0, "svc_lag_60": 0.0, "svc_lag_90": 0.0}

    queue_at_start = 0.0

    rows = []
    for ts in time_slots:
        ts_dt = ts.to_pydatetime()

        # --- baseline medians for this slot ---
        slot_id = int(((ts_dt.hour * 60 + ts_dt.minute) - (OPEN_HOUR * 60)) // FREQ_MIN)
        mo = ts_dt.month
        dow = ts_dt.weekday()
        base_arr, base_svc, base_wait = get_baseline(baseline, mo, dow, slot_id)

        # =====================================================
        # 1) arrivals (reception_count)
        # =====================================================
        af = _make_zero_df(arr_cols)
        _, _, _ = add_common_time_features(af, ts_dt, target_date)
        add_calendar_features(af, target_date)
        add_outpatient(af, total_outpatient_count)
        add_weather_features(af, weather)
        add_lag_features_arr(af, arr_lags)
        add_queue_feature(af, queue_at_start)  # safe if not in columns

        pred_arr = _predict_booster(booster_arr, arr_cols, af)
        pred_arr = max(0.0, pred_arr)
        pred_arr = blend_with_baseline(pred_arr, base_arr, slot_id)  # baseline regularization
        arr_i = int(round(pred_arr))
        if arr_i < 0: arr_i = 0

        # =====================================================
        # 2) service (call_count)  ※ staff capacity proxy
        # =====================================================
        sf = _make_zero_df(multi_cols)
        _, _, _ = add_common_time_features(sf, ts_dt, target_date)
        add_calendar_features(sf, target_date)
        add_outpatient(sf, total_outpatient_count)
        add_weather_features(sf, weather)
        add_lag_features_arr(sf, arr_lags)
        add_lag_features_svc(sf, svc_lags)
        add_queue_feature(sf, queue_at_start)

        pred_svc = _predict_booster(booster_svc, multi_cols, sf)
        pred_svc = max(0.0, pred_svc)
        pred_svc = blend_with_baseline(pred_svc, base_svc, slot_id)
        svc_i = int(round(pred_svc))
        if svc_i < 0: svc_i = 0

        # =====================================================
        # 3) queue (conservation)
        # =====================================================
        queue_end = max(0.0, float(queue_at_start) + float(arr_i) - float(svc_i))

        # =====================================================
        # 4) wait time
        #   - wait model
        #   - physics wait ~ queue / service * slot
        #   - calibrated and blended
        # =====================================================
        wf = _make_zero_df(multi_cols)
        _, _, _ = add_common_time_features(wf, ts_dt, target_date)
        add_calendar_features(wf, target_date)
        add_outpatient(wf, total_outpatient_count)
        add_weather_features(wf, weather)
        add_lag_features_arr(wf, arr_lags)
        add_lag_features_svc(wf, svc_lags)
        add_queue_feature(wf, queue_at_start)

        # wait model prediction
        pred_wait_model = _predict_booster(booster_wait, multi_cols, wf)
        pred_wait_model = max(0.0, pred_wait_model)

        # physics wait (calibrated)
        svc_for_phy = max(SERVICE_MIN, float(svc_i))
        wait_phy = (float(queue_at_start) / svc_for_phy) * SLOT_MIN
        wait_cal = a * wait_phy + b
        wait_cal = max(0.0, wait_cal)

        # baseline regularization on wait too (helps peak underestimation)
        pred_wait_model = blend_with_baseline(pred_wait_model, base_wait, slot_id)

        # blend model & physics
        wait_final = blend_alpha * pred_wait_model + (1.0 - blend_alpha) * wait_cal
        wait_final = float(np.clip(wait_final, 0.0, WAIT_MAX))
        wait_i = int(round(wait_final))

        rows.append({
            "時間帯": ts_dt.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測処理数(呼出数)": int(svc_i),
            "予測待ち人数(人)": int(round(queue_at_start)),
            "予測平均待ち時間(分)": int(wait_i),
        })

        # update lags and state for next slot
        arr_lags = {"arr_lag_30": float(arr_i), "arr_lag_60": float(arr_lags["arr_lag_30"]), "arr_lag_90": float(arr_lags["arr_lag_60"])}
        svc_lags = {"svc_lag_30": float(svc_i), "svc_lag_60": float(svc_lags["svc_lag_30"]), "svc_lag_90": float(svc_lags["svc_lag_60"])}

        queue_at_start = queue_end

    return pd.DataFrame(rows)

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="A病院 採血 待ち人数・待ち時間 予測", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（最終版：arrivals + service + queue + wait）")
    st.caption("入力追加なし（予測日/外来数/簡易天気）。待ち時間はML + 物理(保存則)ブレンド。")

    # File check
    required = [
        ARR_MODEL_PATH, ARR_COLS_PATH,
        SVC_MODEL_PATH, WAIT_MODEL_PATH,
        MULTI_COLS_PATH,
        BASELINE_PATH, CALIB_PATH,
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error(
            "models/ に必要ファイルが不足しています:\n\n" +
            "\n".join([f"- {m}" for m in missing]) +
            "\n\n※ A_models_bundle.zip を展開して models/ に配置してください。"
        )
        st.stop()

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気（簡易）", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=0)
        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.subheader("読み込みモデル")
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
            st.download_button(
                "CSVダウンロード",
                data=csv,
                file_name=f"A_predict_{target}.csv",
                mime="text/csv"
            )

        with c2:
            st.subheader("可視化")
            st.line_chart(df.set_index("時間帯")[["予測平均待ち時間(分)"]])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

        st.divider()
        st.caption("※ 祝日判定は data/syukujitsu.csv があればそれを使用（なければ土日・年末年始のみ）")

    st.divider()
    st.caption("vFinal: baseline(中央値)で初期枠を正則化 / 保存則キュー / wait=ML+physics(calibrated)")

if __name__ == "__main__":
    main()
