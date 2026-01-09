import json
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# =========================================================
# Paths
# =========================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

COUNT_MODEL_PATH = MODELS_DIR / "model_A_timeseries.json"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"
QUEUE_MODEL_PATH = MODELS_DIR / "model_A_queue_30min.json"

COUNT_COLS_PATH  = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

# =========================================================
# Fixed definitions (must match training)
# =========================================================
OPEN_HOUR = 8
OPEN_MIN  = 0
# close slot excluded -> last slot is 17:30
LAST_HOUR = 17
LAST_MIN  = 30
FREQ_MIN  = 30

WEATHER_CATS = ["晴", "曇", "雨", "雪"]

# If you used these numeric weather columns in training,
# provide defaults and optional UI input
DEFAULT_WEATHER_NUM = {
    "降水量": 0.0,
    "平均気温": 15.0,
    "最高気温": 18.0,
    "最低気温": 12.0,
    "平均湿度": 60.0,
    "平均風速": 2.0,
}

# =========================================================
# Holidays
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
    # weekend
    if d.weekday() >= 5:
        return True
    # national holidays from CSV
    if d in HOLIDAYS:
        return True
    # year-end / new-year
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return True
    return False

def week_of_month(d: date) -> int:
    return int((d.day - 1) // 7 + 1)

def normalize_weather_cat(s: str) -> str:
    s = str(s) if s is not None else ""
    # fixed 4 cats
    if "雪" in s:
        return "雪"
    if "雨" in s:
        return "雨"
    if "曇" in s:
        return "曇"
    if "晴" in s:
        return "晴"
    # fallback
    return "曇"

# =========================================================
# Model loader
# =========================================================
@st.cache_resource
def load_models_and_columns():
    # columns
    count_cols = json.loads(COUNT_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    # boosters
    count_booster = xgb.Booster()
    count_booster.load_model(str(COUNT_MODEL_PATH))

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAIT_MODEL_PATH))

    queue_booster = xgb.Booster()
    queue_booster.load_model(str(QUEUE_MODEL_PATH))

    return count_booster, count_cols, wait_booster, queue_booster, multi_cols

# =========================================================
# Feature building utilities
# =========================================================
def make_empty_row(cols: list[str]) -> pd.DataFrame:
    # Ensure all columns exist; fill with 0
    return pd.DataFrame({c: [0] for c in cols})

def set_if_exists(df: pd.DataFrame, col: str, value):
    if col in df.columns:
        df.loc[0, col] = value

def add_fixed_calendar_features(df: pd.DataFrame, ts: datetime, target_date: date, total_out: int,
                                weather_cat: str, rain_flag: int, snow_flag: int,
                                weather_nums: dict,
                                queue_at_start: int,
                                lags: dict):
    """
    Fill df with features that may or may not exist in df.columns.
    """
    # time
    set_if_exists(df, "hour", int(ts.hour))
    set_if_exists(df, "minute", int(ts.minute))

    # month / week-of-month
    set_if_exists(df, "月", int(ts.month))
    set_if_exists(df, "週回数", int(week_of_month(target_date)))

    # holiday
    is_h = int(is_holiday(target_date))
    prev = target_date - timedelta(days=1)
    is_prev_h = int(is_holiday(prev))
    set_if_exists(df, "is_holiday", is_h)
    set_if_exists(df, "前日祝日フラグ", is_prev_h)

    # outpatient count
    set_if_exists(df, "total_outpatient_count", int(total_out))

    # weekday fixed one-hot dayofweek_0..6
    dow = ts.weekday()  # 0=Mon
    for k in range(7):
        set_if_exists(df, f"dayofweek_{k}", 1 if dow == k else 0)

    # slot flags
    is_first = int(ts.hour == 8 and ts.minute == 0)
    is_second = int(ts.hour == 8 and ts.minute == 30)
    set_if_exists(df, "is_first_slot", is_first)
    set_if_exists(df, "is_second_slot", is_second)

    # rain/snow
    set_if_exists(df, "雨フラグ", int(rain_flag))
    set_if_exists(df, "雪フラグ", int(snow_flag))

    # weather cat one-hot (fixed)
    for cat in WEATHER_CATS:
        set_if_exists(df, f"天気カテゴリ_{cat}", 1 if weather_cat == cat else 0)

    # numeric weather if present in columns
    for k, v in weather_nums.items():
        set_if_exists(df, k, float(v))

    # queue_at_start_of_slot if present
    set_if_exists(df, "queue_at_start_of_slot", int(queue_at_start))

    # lags / rolling
    for k, v in lags.items():
        set_if_exists(df, k, float(v))
    if "rolling_mean_60min" in df.columns:
        df.loc[0, "rolling_mean_60min"] = float((lags.get("lag_30min", 0.0) + lags.get("lag_60min", 0.0)) / 2.0)

def predict_booster(booster: xgb.Booster, cols: list[str], df_row: pd.DataFrame) -> float:
    """
    Robust prediction: ensure all required columns exist & numeric.
    """
    X = df_row.copy()
    # align columns (missing -> 0)
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]

    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    X = X.fillna(0)

    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

# =========================================================
# Simulation
# =========================================================
def simulate_one_day(target_date: date,
                     total_outpatient_count: int,
                     weather_choice: str,
                     weather_nums: dict) -> pd.DataFrame:
    count_booster, count_cols, wait_booster, queue_booster, multi_cols = load_models_and_columns()

    # normalize weather
    wcat = normalize_weather_cat(weather_choice)
    rain_flag = 1 if wcat == "雨" else 0
    snow_flag = 1 if wcat == "雪" else 0

    # build timeslots: 08:00 -> 17:30
    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, OPEN_MIN)
    end = datetime(target_date.year, target_date.month, target_date.day, LAST_HOUR, LAST_MIN)
    time_slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")

    # state
    lags = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0}
    queue_at_start = 0

    results = []
    for ts in time_slots:
        # ---------------------------
        # (1) predict reception_count
        # ---------------------------
        cf = make_empty_row(count_cols)

        add_fixed_calendar_features(
            cf, ts.to_pydatetime(), target_date,
            total_outpatient_count,
            weather_cat=wcat,
            rain_flag=rain_flag,
            snow_flag=snow_flag,
            weather_nums=weather_nums,
            queue_at_start=queue_at_start,  # if count model expects it, provide
            lags=lags
        )

        pred_reception = predict_booster(count_booster, count_cols, cf)
        pred_reception_i = max(0, int(round(pred_reception)))

        # ---------------------------
        # (2) predict queue / wait
        # ---------------------------
        mf = make_empty_row(multi_cols)

        add_fixed_calendar_features(
            mf, ts.to_pydatetime(), target_date,
            total_outpatient_count,
            weather_cat=wcat,
            rain_flag=rain_flag,
            snow_flag=snow_flag,
            weather_nums=weather_nums,
            queue_at_start=queue_at_start,
            lags=lags
        )

        # multi models usually require reception_count
        set_if_exists(mf, "reception_count", int(pred_reception_i))

        pred_queue = predict_booster(queue_booster, multi_cols, mf)
        pred_wait  = predict_booster(wait_booster,  multi_cols, mf)

        pred_queue_i = max(0, int(round(pred_queue)))
        pred_wait_i  = max(0, int(round(pred_wait)))

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": pred_reception_i,
            "予測待ち人数(人)": pred_queue_i,
            "予測待ち時間(分)": pred_wait_i,
        })

        # update state for next slot
        lags = {
            "lag_30min": float(pred_reception_i),
            "lag_60min": float(lags["lag_30min"]),
            "lag_90min": float(lags["lag_60min"]),
        }
        queue_at_start = pred_queue_i

    return pd.DataFrame(results)

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="A病院 採血 混雑予測", layout="wide")
    st.title("🏥 A病院 採血 混雑予測（受付数・待ち人数・待ち時間）")
    st.caption("※ Boosterモデル（xgboost）を使用。学習と同じ特徴量定義で推論します。")

    # file checks
    required = [COUNT_MODEL_PATH, WAIT_MODEL_PATH, QUEUE_MODEL_PATH, COUNT_COLS_PATH, MULTI_COLS_PATH]
    missing = [p for p in required if not p.exists()]
    if missing:
        st.error("必要ファイルが不足しています。`models/` に以下を配置してください：\n\n" +
                 "\n".join([f"- {p.name}" for p in missing]))
        st.stop()

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数（total_outpatient_count）", min_value=0, value=1200, step=10)

        weather_choice = st.selectbox("天気カテゴリ（正規化：晴/曇/雨/雪）", WEATHER_CATS, index=0)

        with st.expander("気象の詳細入力（任意：学習に数値気象を入れている場合は推奨）", expanded=False):
            st.caption("学習で「降水量/気温/湿度/風速」を使っている場合、ここを入れると精度が安定します。")
            rain = st.number_input("降水量(mm)", value=float(DEFAULT_WEATHER_NUM["降水量"]), step=0.1)
            tavg = st.number_input("平均気温(℃)", value=float(DEFAULT_WEATHER_NUM["平均気温"]), step=0.1)
            tmax = st.number_input("最高気温(℃)", value=float(DEFAULT_WEATHER_NUM["最高気温"]), step=0.1)
            tmin = st.number_input("最低気温(℃)", value=float(DEFAULT_WEATHER_NUM["最低気温"]), step=0.1)
            hum  = st.number_input("平均湿度(%)", value=float(DEFAULT_WEATHER_NUM["平均湿度"]), step=1.0)
            wind = st.number_input("平均風速(m/s)", value=float(DEFAULT_WEATHER_NUM["平均風速"]), step=0.1)

        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.subheader("モデル/ファイル")
        st.write("受付数:", COUNT_MODEL_PATH.name)
        st.write("待ち時間:", WAIT_MODEL_PATH.name)
        st.write("待ち人数:", QUEUE_MODEL_PATH.name)

    weather_nums = {
        "降水量": float(rain),
        "平均気温": float(tavg),
        "最高気温": float(tmax),
        "最低気温": float(tmin),
        "平均湿度": float(hum),
        "平均風速": float(wind),
    }

    # preview model columns
    with st.expander("デバッグ：モデルの使用列（確認用）", expanded=False):
        _, count_cols, _, _, multi_cols = load_models_and_columns()
        st.write("count cols:", len(count_cols))
        st.write(count_cols)
        st.write("multi cols:", len(multi_cols))
        st.write(multi_cols)

    if run:
        with st.spinner("計算中..."):
            df = simulate_one_day(target, int(total_out), str(weather_choice), weather_nums)

        st.success(f"{target} の予測が完了しました。")

        c1, c2 = st.columns([2, 3], gap="large")

        with c1:
            st.subheader("結果テーブル")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("CSVダウンロード", data=csv, file_name=f"A_predict_{target}.csv", mime="text/csv")

        with c2:
            st.subheader("可視化")
            st.line_chart(df.set_index("時間帯")[["予測待ち時間(分)"]])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

    st.divider()
    st.caption("※ 祝日判定は data/syukujitsu.csv（任意）を参照。無ければ土日・年末年始のみ。")
    st.caption("※ 学習で数値気象特徴（降水量/気温/湿度/風速）を使っている場合、サイドバーの詳細入力を推奨。")

if __name__ == "__main__":
    main()
