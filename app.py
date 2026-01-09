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

# ===== filenames (match your training outputs) =====
ARR_MODEL_PATH  = MODELS_DIR / "model_A_timeseries.json"
ARR_COLS_PATH   = MODELS_DIR / "columns_A_timeseries.json"

SVC_MODEL_PATH  = MODELS_DIR / "model_A_service_30min.json"   # ★ service model
WAIT_MEAN_PATH  = MODELS_DIR / "model_A_waittime_30min.json"  # mean wait
WAIT_P90_PATH   = MODELS_DIR / "model_A_waittime_p90_30min.json"  # optional but recommended

MULTI_COLS_PATH = MODELS_DIR / "columns_A_multi_30min.json"
BASELINE_PATH   = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH      = MODELS_DIR / "wait_calibration.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"  # optional

# ===== time config =====
OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
SLOT_MIN = 30.0

# ===== safety bounds =====
WAIT_MAX = 180.0
SERVICE_MIN = 0.0   # allow 0, but we will stabilize by fallback/cap
ARR_MAX = 300        # safety only (won't matter usually)

WEATHER_CATS = ["晴", "曇", "雨", "雪"]

# ===== peak window =====
PEAK_START = (8, 30)
PEAK_END   = (11, 0)   # inclusive

# ===== congestion factor clamp (weakened) =====
CONGESTION_CLAMP = (0.90, 1.20)   # ★ too wide will explode; keep narrow


# -------------------------
# Holiday
# -------------------------
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

def normalize_weather(w: str) -> str:
    s = str(w) if w is not None else ""
    if "雪" in s: return "雪"
    if "雨" in s: return "雨"
    if "曇" in s: return "曇"
    if "晴" in s: return "晴"
    return "曇"

def slot_id_from_ts(ts: datetime) -> int:
    return int(((ts.hour * 60 + ts.minute) - (OPEN_HOUR * 60)) // FREQ_MIN)

def baseline_key(month: int, dow: int, slot_id: int) -> str:
    return f"{int(month)}_{int(dow)}_{int(slot_id)}"

def in_peak(ts: datetime) -> bool:
    h, m = ts.hour, ts.minute
    after_start = (h > PEAK_START[0]) or (h == PEAK_START[0] and m >= PEAK_START[1])
    before_end  = (h < PEAK_END[0]) or (h == PEAK_END[0] and m <= PEAK_END[1])
    return after_start and before_end


# -------------------------
# Load assets
# -------------------------
@st.cache_resource
def load_assets():
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    bst_arr = xgb.Booster()
    bst_arr.load_model(str(ARR_MODEL_PATH))

    bst_svc = xgb.Booster()
    bst_svc.load_model(str(SVC_MODEL_PATH))

    bst_wm = xgb.Booster()
    bst_wm.load_model(str(WAIT_MEAN_PATH))

    bst_wp90 = None
    if WAIT_P90_PATH.exists():
        bst_wp90 = xgb.Booster()
        bst_wp90.load_model(str(WAIT_P90_PATH))

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    blend_alpha = float(calib.get("blend_alpha", 0.65))

    return bst_arr, arr_cols, bst_svc, bst_wm, bst_wp90, multi_cols, baseline, (a, b, blend_alpha)

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def _predict(bst: xgb.Booster, cols, dfrow: pd.DataFrame) -> float:
    X = _coerce_numeric(dfrow[cols].copy())
    d = xgb.DMatrix(X, feature_names=list(cols))
    return float(bst.predict(d)[0])

def get_baseline(baseline: dict, month: int, dow: int, slot_id: int):
    k = baseline_key(month, dow, slot_id)
    arr = float(baseline.get("arr", {}).get(k, {}).get("median", 0.0))
    svc = float(baseline.get("svc", {}).get(k, {}).get("median", 0.0))
    wm  = float(baseline.get("wait_mean", {}).get(k, {}).get("median", 0.0))
    wp  = float(baseline.get("wait_p90", {}).get(k, {}).get("median", 0.0))
    return arr, svc, wm, wp

def add_common(dfrow: pd.DataFrame, ts: datetime, target_date: date):
    dow = ts.weekday()
    month = ts.month
    sid = slot_id_from_ts(ts)

    if "hour" in dfrow.columns: dfrow.loc[0, "hour"] = int(ts.hour)
    if "minute" in dfrow.columns: dfrow.loc[0, "minute"] = int(ts.minute)
    if "dow" in dfrow.columns: dfrow.loc[0, "dow"] = int(dow)
    if "month" in dfrow.columns: dfrow.loc[0, "month"] = int(month)
    if "slot_id" in dfrow.columns: dfrow.loc[0, "slot_id"] = int(sid)

    if "月" in dfrow.columns: dfrow.loc[0, "月"] = int(month)
    if "週回数" in dfrow.columns: dfrow.loc[0, "週回数"] = int(week_of_month(target_date))

    for k in range(7):
        c = f"dayofweek_{k}"
        if c in dfrow.columns:
            dfrow.loc[0, c] = 1 if k == dow else 0

    if "is_first_slot" in dfrow.columns:
        dfrow.loc[0, "is_first_slot"] = 1 if (ts.hour == 8 and ts.minute == 0) else 0
    if "is_second_slot" in dfrow.columns:
        dfrow.loc[0, "is_second_slot"] = 1 if (ts.hour == 8 and ts.minute == 30) else 0

    return sid, month, dow

def add_calendar(dfrow: pd.DataFrame, d: date):
    if "is_holiday" in dfrow.columns:
        dfrow.loc[0, "is_holiday"] = int(is_holiday(d))
    prev = d - timedelta(days=1)
    if "前日祝日フラグ" in dfrow.columns:
        dfrow.loc[0, "前日祝日フラグ"] = int(is_holiday(prev))

def add_weather(dfrow: pd.DataFrame, w: str):
    w = normalize_weather(w)
    if "雨フラグ" in dfrow.columns: dfrow.loc[0, "雨フラグ"] = 1 if w == "雨" else 0
    if "雪フラグ" in dfrow.columns: dfrow.loc[0, "雪フラグ"] = 1 if w == "雪" else 0
    for cat in WEATHER_CATS:
        c = f"天気カテゴリ_{cat}"
        if c in dfrow.columns:
            dfrow.loc[0, c] = 1 if w == cat else 0

def add_outpatient(dfrow: pd.DataFrame, total_out: int):
    if "total_outpatient_count" in dfrow.columns:
        dfrow.loc[0, "total_outpatient_count"] = int(total_out)

def add_state(dfrow: pd.DataFrame, queue0: float, cum_arr: float):
    # training had these columns; inference uses predicted values
    if "queue_at_start_truth" in dfrow.columns:
        dfrow.loc[0, "queue_at_start_truth"] = float(queue0)
    if "cum_arrivals_sofar" in dfrow.columns:
        dfrow.loc[0, "cum_arrivals_sofar"] = float(cum_arr)

def add_lags(dfrow: pd.DataFrame, lags: dict, roll60_name: str):
    for k, v in lags.items():
        if k in dfrow.columns:
            dfrow.loc[0, k] = float(v)
    if roll60_name in dfrow.columns:
        # expects keys *_lag_30 and *_lag_60
        k30 = [k for k in lags.keys() if k.endswith("lag_30")][0]
        k60 = [k for k in lags.keys() if k.endswith("lag_60")][0]
        dfrow.loc[0, roll60_name] = (float(lags[k30]) + float(lags[k60])) / 2.0


# -------------------------
# Simulation
# -------------------------
def simulate_one_day(target_date: date, total_out: int, weather: str) -> pd.DataFrame:
    bst_arr, arr_cols, bst_svc, bst_wm, bst_wp90, multi_cols, baseline, calib = load_assets()
    a, b, blend_alpha = calib

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min").to_pydatetime().tolist()
    # drop 18:00
    if slots and slots[-1].hour == CLOSE_HOUR and slots[-1].minute == 0:
        slots = slots[:-1]

    arr_lags = {"arr_lag_30": 0.0, "arr_lag_60": 0.0, "arr_lag_90": 0.0}
    svc_lags = {"svc_lag_30": 0.0, "svc_lag_60": 0.0, "svc_lag_90": 0.0}

    queue0 = 0.0
    cum_arr = 0.0

    # --- congestion factor from first 2 slots vs baseline
    congestion_factor = 1.0
    first_two = []

    rows = []

    for ts in slots:
        sid, month, dow = slot_id_from_ts(ts), ts.month, ts.weekday()
        base_arr, base_svc, base_wm, base_wp = get_baseline(baseline, month, dow, sid)

        # ========== arrivals ==========
        af = _make_zero_df(arr_cols)
        add_common(af, ts, target_date)
        add_calendar(af, target_date)
        add_outpatient(af, total_out)
        add_weather(af, weather)
        add_state(af, queue0, cum_arr)
        add_lags(af, arr_lags, "arr_roll_60")

        pred_arr = max(0.0, _predict(bst_arr, arr_cols, af))
        arr_i = int(round(pred_arr))
        arr_i = int(np.clip(arr_i, 0, ARR_MAX))

        # estimate congestion factor after slot0+slot1
        if sid in (0, 1):
            first_two.append((arr_i, max(1.0, base_arr)))
            if len(first_two) == 2:
                ratio = (first_two[0][0] + first_two[1][0]) / (first_two[0][1] + first_two[1][1])
                congestion_factor = float(np.clip(ratio, CONGESTION_CLAMP[0], CONGESTION_CLAMP[1]))

        # apply mild congestion to arrivals in peak
        if in_peak(ts):
            arr_i = int(round(arr_i * congestion_factor))

        # ========== service ==========
        sf = _make_zero_df(multi_cols)
        add_common(sf, ts, target_date)
        add_calendar(sf, target_date)
        add_outpatient(sf, total_out)
        add_weather(sf, weather)
        add_state(sf, queue0, cum_arr)
        add_lags(sf, arr_lags, "arr_roll_60")
        add_lags(sf, svc_lags, "svc_roll_60")

        pred_svc = max(0.0, _predict(bst_svc, multi_cols, sf))
        svc_i = int(round(pred_svc))
        svc_i = max(int(SERVICE_MIN), svc_i)

        # ★IMPORTANT FIX:
        # service cannot exceed available people (queue0 + arrivals)
        available = int(round(queue0)) + int(arr_i)
        svc_i = min(svc_i, max(0, available))

        # peak congestion: effective service slightly lower (mild)
        if in_peak(ts):
            svc_i = int(round(svc_i / congestion_factor))
            svc_i = min(svc_i, max(0, available))

        # optional fallback: if service becomes 0 while there are people, use baseline
        if svc_i == 0 and available > 0:
            svc_i = int(round(min(available, max(1.0, base_svc))))

        # ========== queue (conservation) ==========
        queue_end = max(0.0, queue0 + float(arr_i) - float(svc_i))

        # ========== wait mean ==========
        wf = _make_zero_df(multi_cols)
        add_common(wf, ts, target_date)
        add_calendar(wf, target_date)
        add_outpatient(wf, total_out)
        add_weather(wf, weather)
        add_state(wf, queue0, cum_arr)
        add_lags(wf, arr_lags, "arr_roll_60")
        add_lags(wf, svc_lags, "svc_roll_60")

        wait_model = max(0.0, _predict(bst_wm, multi_cols, wf))

        # physics wait (queue / service)
        svc_for_phy = max(1.0, float(svc_i))
        wait_phy = (float(queue0) / svc_for_phy) * SLOT_MIN
        wait_phy_cal = max(0.0, a * wait_phy + b)

        wait_mean = blend_alpha * wait_model + (1.0 - blend_alpha) * wait_phy_cal

        # mild congestion push in peak
        if in_peak(ts):
            wait_mean *= congestion_factor

        wait_mean = float(np.clip(wait_mean, 0.0, WAIT_MAX))

        # ========== wait p90 ==========
        wait_p90 = np.nan
        if bst_wp90 is not None:
            wp = max(0.0, _predict(bst_wp90, multi_cols, wf))
            if in_peak(ts):
                wp *= congestion_factor
            wait_p90 = float(np.clip(wp, 0.0, WAIT_MAX))

        rows.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測処理数(呼出数)": int(svc_i),
            "予測待ち人数(人)": int(round(queue0)),
            "予測平均待ち時間(分)": int(round(wait_mean)),
            "予測混雑時待ち時間_p90(分)": (int(round(wait_p90)) if not np.isnan(wait_p90) else np.nan),
        })

        # update states
        cum_arr += float(arr_i)
        arr_lags = {
            "arr_lag_30": float(arr_i),
            "arr_lag_60": float(arr_lags["arr_lag_30"]),
            "arr_lag_90": float(arr_lags["arr_lag_60"]),
        }
        svc_lags = {
            "svc_lag_30": float(svc_i),
            "svc_lag_60": float(svc_lags["svc_lag_30"]),
            "svc_lag_90": float(svc_lags["svc_lag_60"]),
        }
        queue0 = queue_end

    return pd.DataFrame(rows), congestion_factor


# -------------------------
# UI
# -------------------------
def main():
    st.set_page_config(page_title="A病院 採血 待ち予測（最終版・安定化）", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（最終版・安定化）")
    st.caption("追加入力なし。service予測を保存則でクリップして破綻を防ぎ、ピーク（8:30–11:00）は朝9時までの混雑係数で軽く補正します。")

    required = [
        ARR_MODEL_PATH, ARR_COLS_PATH,
        SVC_MODEL_PATH, WAIT_MEAN_PATH,
        MULTI_COLS_PATH, BASELINE_PATH, CALIB_PATH
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("models/ に必要ファイルが不足しています:\n\n" + "\n".join([f"- {m}" for m in missing]))
        st.stop()

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気（簡易）", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=0)
        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.subheader("モデル")
        st.write("arrivals:", ARR_MODEL_PATH.name)
        st.write("service :", SVC_MODEL_PATH.name)
        st.write("wait(mean):", WAIT_MEAN_PATH.name)
        st.write("wait(p90):", WAIT_P90_PATH.name if WAIT_P90_PATH.exists() else "（未配置）")
        st.write("baseline:", BASELINE_PATH.name)
        st.write("calib:", CALIB_PATH.name)

    if run:
        with st.spinner("計算中..."):
            df, cf = simulate_one_day(target, int(total_out), str(weather))
        st.success(f"{target} の予測が完了しました。混雑係数（朝9時推定）: {cf:.2f}")

        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.subheader("結果テーブル")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("CSVダウンロード", data=csv, file_name=f"A_predict_{target}.csv", mime="text/csv")

        with c2:
            st.subheader("可視化")
            cols = ["予測平均待ち時間(分)"]
            if df["予測混雑時待ち時間_p90(分)"].notna().any():
                cols.append("予測混雑時待ち時間_p90(分)")
            st.line_chart(df.set_index("時間帯")[cols])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

    st.divider()
    st.caption("※ 祝日判定は data/syukujitsu.csv があれば参照（なければ土日・年末年始のみ）")

if __name__ == "__main__":
    main()
