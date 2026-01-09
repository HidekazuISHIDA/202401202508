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

ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.json"       # arrivals (reception_count)
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"    # wait (by reception slot)
SVC_MODEL_PATH   = MODELS_DIR / "model_A_queue_30min.json"       # ★service (call_count) - name kept for compatibility

ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"          # a,b,blend_alpha (optional)

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

# =========================================================
# Fixed defs (must match training)
# =========================================================
OPEN_HOUR, OPEN_MIN = 8, 0
LAST_HOUR, LAST_MIN = 17, 30       # 18:00 excluded
FREQ_MIN = 30

WEATHER_CATS = ["晴", "曇", "雨", "雪"]

DEFAULT_WEATHER_NUM = {
    "降水量": 0.0,
    "平均気温": 15.0,
    "最高気温": 18.0,
    "最低気温": 12.0,
    "平均湿度": 60.0,
    "平均風速": 2.0,
}

# Safety clips (adjust to your site)
CLIP_ARR_MAX = 200
CLIP_SVC_MAX = 200
CLIP_QUEUE_MAX = 500
CLIP_WAIT_MAX = 240  # minutes

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
    if d.weekday() >= 5:
        return True
    if d in HOLIDAYS:
        return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return True
    return False

def week_of_month(d: date) -> int:
    return int((d.day - 1) // 7 + 1)

def normalize_weather_cat(s: str) -> str:
    s = str(s) if s is not None else ""
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
# Load models/cols
# =========================================================
@st.cache_resource
def load_assets():
    # columns
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    # boosters
    arr_booster = xgb.Booster()
    arr_booster.load_model(str(ARR_MODEL_PATH))

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAIT_MODEL_PATH))

    svc_booster = xgb.Booster()
    svc_booster.load_model(str(SVC_MODEL_PATH))

    # calibration (optional)
    calib = {"a": 1.0, "b": 0.0, "blend_alpha": 0.65}
    if CALIB_PATH.exists():
        try:
            calib.update(json.loads(CALIB_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass

    return arr_booster, arr_cols, wait_booster, svc_booster, multi_cols, calib

# =========================================================
# Feature helpers
# =========================================================
def make_row(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [0] for c in cols})

def set_if_exists(df: pd.DataFrame, col: str, value):
    if col in df.columns:
        df.loc[0, col] = value

def fill_common_features(
    df: pd.DataFrame,
    ts: datetime,
    target_date: date,
    total_out: int,
    weather_cat: str,
    weather_nums: dict,
    queue_at_start: int,
    lags_arr: dict,
    lags_svc: dict,
):
    # time
    set_if_exists(df, "hour", int(ts.hour))
    set_if_exists(df, "minute", int(ts.minute))

    # month/week
    set_if_exists(df, "月", int(ts.month))
    set_if_exists(df, "週回数", int(week_of_month(target_date)))

    # holiday
    set_if_exists(df, "is_holiday", int(is_holiday(target_date)))
    prev = target_date - timedelta(days=1)
    set_if_exists(df, "前日祝日フラグ", int(is_holiday(prev)))

    # outpatient count
    set_if_exists(df, "total_outpatient_count", int(total_out))

    # weekday fixed one-hot
    dow = ts.weekday()
    for k in range(7):
        set_if_exists(df, f"dayofweek_{k}", 1 if dow == k else 0)

    # slot flags
    set_if_exists(df, "is_first_slot", int(ts.hour == 8 and ts.minute == 0))
    set_if_exists(df, "is_second_slot", int(ts.hour == 8 and ts.minute == 30))

    # rain/snow flags
    set_if_exists(df, "雨フラグ", 1 if weather_cat == "雨" else 0)
    set_if_exists(df, "雪フラグ", 1 if weather_cat == "雪" else 0)

    # weather cat one-hot
    for cat in WEATHER_CATS:
        set_if_exists(df, f"天気カテゴリ_{cat}", 1 if weather_cat == cat else 0)

    # numeric weather
    for k, v in weather_nums.items():
        set_if_exists(df, k, float(v))

    # queue at start (used by service/wait models)
    set_if_exists(df, "queue_at_start_truth", int(queue_at_start))
    set_if_exists(df, "queue_at_start_of_slot", int(queue_at_start))  # backward compat if present
    set_if_exists(df, "queue_at_start", int(queue_at_start))          # just in case

    # arrivals lags (support multiple naming variants)
    for k, v in lags_arr.items():
        set_if_exists(df, k, float(v))
    set_if_exists(df, "rolling_mean_60min", float((lags_arr.get("lag_30min", 0.0) + lags_arr.get("lag_60min", 0.0)) / 2.0))
    set_if_exists(df, "arr_roll_60", float((lags_arr.get("arr_lag_30", 0.0) + lags_arr.get("arr_lag_60", 0.0)) / 2.0))

    # service lags
    for k, v in lags_svc.items():
        set_if_exists(df, k, float(v))
    set_if_exists(df, "svc_roll_60", float((lags_svc.get("svc_lag_30", 0.0) + lags_svc.get("svc_lag_60", 0.0)) / 2.0))

def predict(booster: xgb.Booster, cols: list[str], row_df: pd.DataFrame) -> float:
    X = row_df.copy()
    # add missing
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]

    # numeric coercion
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    X = X.fillna(0)

    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

# =========================================================
# Simulation (arrivals + services -> queue -> wait blend)
# =========================================================
def simulate_one_day(target_date: date, total_out: int, weather_choice: str, weather_nums: dict) -> pd.DataFrame:
    arr_booster, arr_cols, wait_booster, svc_booster, multi_cols, calib = load_assets()

    wcat = normalize_weather_cat(weather_choice)

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, OPEN_MIN)
    end   = datetime(target_date.year, target_date.month, target_date.day, LAST_HOUR, LAST_MIN)
    slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")

    # state
    queue = 0.0

    # lags (support both old and new names)
    lags_arr = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0,
                "arr_lag_30": 0.0, "arr_lag_60": 0.0, "arr_lag_90": 0.0}
    lags_svc = {"svc_lag_30": 0.0, "svc_lag_60": 0.0, "svc_lag_90": 0.0}

    results = []
    for ts in slots:
        ts_py = ts.to_pydatetime()
        q0 = float(queue)

        # ---------------------------
        # 1) arrivals prediction
        # ---------------------------
        ar = make_row(arr_cols)
        fill_common_features(ar, ts_py, target_date, total_out, wcat, weather_nums, int(q0), lags_arr, lags_svc)

        arr_pred = predict(arr_booster, arr_cols, ar)
        arr_i = int(np.clip(np.round(arr_pred), 0, CLIP_ARR_MAX))

        # update arrivals lag state
        lags_arr = {
            "lag_30min": float(arr_i),
            "lag_60min": float(lags_arr["lag_30min"]),
            "lag_90min": float(lags_arr["lag_60min"]),
            "arr_lag_30": float(arr_i),
            "arr_lag_60": float(lags_arr["arr_lag_30"]),
            "arr_lag_90": float(lags_arr["arr_lag_60"]),
        }

        # ---------------------------
        # 2) service prediction (call_count)
        # ---------------------------
        mf = make_row(multi_cols)
        fill_common_features(mf, ts_py, target_date, total_out, wcat, weather_nums, int(q0), lags_arr, lags_svc)

        # some models may include arrival/service current counts as features
        set_if_exists(mf, "reception_count", int(arr_i))

        svc_pred = predict(svc_booster, multi_cols, mf)
        svc_i = int(np.clip(np.round(svc_pred), 0, CLIP_SVC_MAX))

        # update service lag state
        lags_svc = {
            "svc_lag_30": float(svc_i),
            "svc_lag_60": float(lags_svc["svc_lag_30"]),
            "svc_lag_90": float(lags_svc["svc_lag_60"]),
        }

        # ---------------------------
        # 3) queue update by conservation
        # ---------------------------
        queue = max(0.0, q0 + float(arr_i) - float(svc_i))
        queue = float(min(queue, CLIP_QUEUE_MAX))

        # ---------------------------
        # 4) wait prediction (model) + physics-calibrated wait
        # ---------------------------
        # model wait (by reception slot)
        wf = make_row(multi_cols)
        fill_common_features(wf, ts_py, target_date, total_out, wcat, weather_nums, int(q0), lags_arr, lags_svc)
        set_if_exists(wf, "reception_count", int(arr_i))
        wait_model_pred = predict(wait_booster, multi_cols, wf)

        # physics wait proxy: (queue_at_start / service_rate) * slot_minutes
        slot_minutes = 30.0
        eps = 1e-6
        wait_phy = (q0 / max(float(svc_i), eps)) * slot_minutes
        wait_phy_cal = calib.get("a", 1.0) * wait_phy + calib.get("b", 0.0)

        alpha = float(calib.get("blend_alpha", 0.65))  # model weight
        wait_blend = alpha * float(wait_model_pred) + (1.0 - alpha) * float(wait_phy_cal)

        wait_i = int(np.clip(np.round(wait_blend), 0, CLIP_WAIT_MAX))

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測処理数(人)": int(svc_i),
            "予測待ち人数(人)": int(round(queue)),
            "予測待ち時間(分)": int(wait_i),
            "参考:待ち時間_model": float(wait_model_pred),
            "参考:待ち時間_phy_cal": float(wait_phy_cal),
        })

    return pd.DataFrame(results)

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="A病院 採血 混雑予測（最終版）", layout="wide")
    st.title("🏥 A病院 採血 混雑予測（最終版）")
    st.caption("受付数(arrivals) + 処理数(service) → 保存則で待ち人数(queue) → 待ち時間は model と physics をブレンド")

    required = [ARR_MODEL_PATH, WAIT_MODEL_PATH, SVC_MODEL_PATH, ARR_COLS_PATH, MULTI_COLS_PATH]
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

        with st.expander("気象の詳細入力（推奨）", expanded=False):
            st.caption("学習で数値気象を使っている場合、ここを入れると精度が安定します。")
            rain = st.number_input("降水量(mm)", value=float(DEFAULT_WEATHER_NUM["降水量"]), step=0.1)
            tavg = st.number_input("平均気温(℃)", value=float(DEFAULT_WEATHER_NUM["平均気温"]), step=0.1)
            tmax = st.number_input("最高気温(℃)", value=float(DEFAULT_WEATHER_NUM["最高気温"]), step=0.1)
            tmin = st.number_input("最低気温(℃)", value=float(DEFAULT_WEATHER_NUM["最低気温"]), step=0.1)
            hum  = st.number_input("平均湿度(%)", value=float(DEFAULT_WEATHER_NUM["平均湿度"]), step=1.0)
            wind = st.number_input("平均風速(m/s)", value=float(DEFAULT_WEATHER_NUM["平均風速"]), step=0.1)

        with st.expander("上限設定（運用調整）", expanded=False):
            st.caption("極端値が出る場合は上限を調整してください（コード上の定数）。")
            st.write(f"受付数上限: {CLIP_ARR_MAX}, 処理数上限: {CLIP_SVC_MAX}, キュー上限: {CLIP_QUEUE_MAX}, 待ち時間上限: {CLIP_WAIT_MAX}")

        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.subheader("モデル/ファイル")
        st.write("arrivals:", ARR_MODEL_PATH.name)
        st.write("service:", SVC_MODEL_PATH.name, "（※ファイル名は互換のため queue_30min のまま）")
        st.write("wait:", WAIT_MODEL_PATH.name)
        if CALIB_PATH.exists():
            st.success("wait_calibration.json を検出（ブレンド補正あり）")
        else:
            st.warning("wait_calibration.json がありません（ブレンドはデフォルト）")

    weather_nums = {
        "降水量": float(rain),
        "平均気温": float(tavg),
        "最高気温": float(tmax),
        "最低気温": float(tmin),
        "平均湿度": float(hum),
        "平均風速": float(wind),
    }

    with st.expander("デバッグ：モデル列（確認用）", expanded=False):
        arr_booster, arr_cols, wait_booster, svc_booster, multi_cols, calib = load_assets()
        st.write("arr_cols:", len(arr_cols))
        st.write(arr_cols)
        st.write("multi_cols:", len(multi_cols))
        st.write(multi_cols)
        st.write("calibration:", calib)

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
            st.line_chart(df.set_index("時間帯")[["予測受付数", "予測処理数(人)"]])

        st.info("参考列（wait_model / wait_phy_cal）もテーブルに含めています。混雑時間帯で model が短く、phy が長い等の診断に使えます。")

    st.divider()
    st.caption("※ queueは回帰で予測せず、保存則で更新（壊れにくい構造）。")
    st.caption("※ 待ち時間はモデル予測と物理ベース推定をブレンド（wait_calibration.jsonで自動較正）。")
    st.caption("※ 祝日CSV（data/syukujitsu.csv）があれば土日以外も休日扱いできます。")

if __name__ == "__main__":
    main()
