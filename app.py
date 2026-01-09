import json
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

ARR_MODEL_PATH  = MODELS_DIR / "model_A_timeseries.json"
SVC_MODEL_PATH  = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH = MODELS_DIR / "model_A_waittime_30min.json"

ARR_COLS_PATH   = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH   = MODELS_DIR / "baseline_tables_mds.json"   # month-dow-slot
CALIB_PATH      = MODELS_DIR / "wait_calibration.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR, OPEN_MIN = 8, 0
LAST_HOUR, LAST_MIN = 17, 30
FREQ_MIN = 30

WEATHER_CATS = ["晴", "曇", "雨", "雪"]

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

def is_holiday(d: date) -> int:
    if d.weekday() >= 5:
        return 1
    if d in HOLIDAYS:
        return 1
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return 1
    return 0

def normalize_weather_cat(s: str) -> str:
    s = str(s) if s is not None else ""
    if "雪" in s: return "雪"
    if "雨" in s: return "雨"
    if "曇" in s: return "曇"
    if "晴" in s: return "晴"
    return "曇"

def slot_id(h: int, m: int) -> int:
    return int(((h*60+m) - (OPEN_HOUR*60+OPEN_MIN)) // FREQ_MIN)

def is_peak_window(h: int, m: int) -> int:
    t = h*60+m
    return int((8*60+30) <= t <= (11*60+0))

def cyc_sin_cos(x: float, period: float):
    ang = 2*np.pi*x/period
    return float(np.sin(ang)), float(np.cos(ang))

@st.cache_resource
def load_assets():
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    arr_booster = xgb.Booster()
    arr_booster.load_model(str(ARR_MODEL_PATH))

    svc_booster = xgb.Booster()
    svc_booster.load_model(str(SVC_MODEL_PATH))

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAIT_MODEL_PATH))

    baselines = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else {}
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8")) if CALIB_PATH.exists() else {}

    # clips
    clips = calib.get("clips", {})
    clip_arr_max = int(clips.get("arr_max", 220))
    clip_svc_max = int(clips.get("svc_max", 220))
    clip_queue_max = int(clips.get("queue_max", 600))
    clip_wait_max = int(calib.get("clip_wait_max", 240))

    return arr_booster, arr_cols, svc_booster, wait_booster, multi_cols, baselines, calib, clip_arr_max, clip_svc_max, clip_queue_max, clip_wait_max

def make_row(cols):
    return pd.DataFrame({c: [0] for c in cols})

def set_if(df, col, val):
    if col in df.columns:
        df.loc[0, col] = val

def predict(booster: xgb.Booster, cols, row: pd.DataFrame) -> float:
    X = row.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    X = X.fillna(0)
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    return float(booster.predict(dmat)[0])

def baseline_get_mds(baselines: dict, kind: str, month: int, dow: int, slot: int):
    key = f"{int(month)}_{int(dow)}_{int(slot)}"
    return baselines.get(kind, {}).get(key, None)

def blend_with_baseline(raw: float, base: dict, w_model: float, clip_min=None, clip_max=None):
    x = float(raw)
    if base is not None:
        med = float(base.get("median", 0.0))
        x = w_model * x + (1.0 - w_model) * med
        p10 = base.get("p10", None)
        p90 = base.get("p90", None)
        if p10 is not None and p90 is not None:
            x = float(np.clip(x, float(p10), float(p90)))
    if clip_min is not None or clip_max is not None:
        x = float(np.clip(x, -np.inf if clip_min is None else clip_min, np.inf if clip_max is None else clip_max))
    return x

def fill_common(row: pd.DataFrame, ts: datetime, target: date, total_out: int, wcat: str, weather_nums: dict,
                queue_at_start: float, arr_lags: dict, svc_lags: dict):
    h, m = ts.hour, ts.minute
    dow = ts.weekday()
    month = ts.month

    set_if(row, "hour", h)
    set_if(row, "minute", m)
    set_if(row, "month", month)
    set_if(row, "月", month)

    set_if(row, "週回数", int((ts.day-1)//7 + 1))
    set_if(row, "is_holiday", int(is_holiday(target)))
    set_if(row, "前日祝日フラグ", int(is_holiday(target - timedelta(days=1))))

    doy = ts.timetuple().tm_yday
    ds, dc = cyc_sin_cos(doy, 365.25)
    set_if(row, "doy_sin", ds)
    set_if(row, "doy_cos", dc)

    mod = h*60 + m
    tsin, tcos = cyc_sin_cos(mod, 24*60)
    set_if(row, "tod_sin", tsin)
    set_if(row, "tod_cos", tcos)

    for k in range(7):
        set_if(row, f"dayofweek_{k}", 1 if dow == k else 0)

    set_if(row, "is_peak_window", int(is_peak_window(h,m)))
    set_if(row, "slot_id", int(slot_id(h,m)))

    set_if(row, "total_outpatient_count", int(total_out))

    set_if(row, "雨フラグ", 1 if wcat=="雨" else 0)
    set_if(row, "雪フラグ", 1 if wcat=="雪" else 0)
    for cat in WEATHER_CATS:
        set_if(row, f"天気カテゴリ_{cat}", 1 if wcat==cat else 0)

    for k,v in weather_nums.items():
        set_if(row, k, float(v))

    set_if(row, "queue_at_start_truth", float(queue_at_start))
    set_if(row, "queue_at_start_of_slot", float(queue_at_start))

    for k,v in arr_lags.items():
        set_if(row, k, float(v))
    for k,v in svc_lags.items():
        set_if(row, k, float(v))

    set_if(row, "is_first_slot", int(h==8 and m==0))
    set_if(row, "is_second_slot", int(h==8 and m==30))

def simulate_one_day(target: date, total_out: int, weather_choice: str, weather_nums: dict) -> pd.DataFrame:
    arr_booster, arr_cols, svc_booster, wait_booster, multi_cols, baselines, calib, CLIP_ARR_MAX, CLIP_SVC_MAX, CLIP_QUEUE_MAX, CLIP_WAIT_MAX = load_assets()
    wcat = normalize_weather_cat(weather_choice)

    start = datetime(target.year, target.month, target.day, OPEN_HOUR, OPEN_MIN)
    end   = datetime(target.year, target.month, target.day, LAST_HOUR, LAST_MIN)
    slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")

    queue = 0.0
    arr_lags = {"arr_Lg30":0.0,"arr_Lg60":0.0,"arr_Lg90":0.0}
    svc_lags = {"svc_Lg30":0.0,"svc_Lg60":0.0,"svc_Lg90":0.0}

    alpha_base = float(calib.get("alpha_base", 0.72))
    alpha_peak = float(calib.get("alpha_peak", 0.32))
    queue_switch = float(calib.get("queue_switch", 15))
    upshift_margin = float(calib.get("peak_upshift_margin", 0.15))
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))

    rows = []
    for ts in slots:
        ts_py = ts.to_pydatetime()
        dow = ts_py.weekday()
        month = ts_py.month
        sid = slot_id(ts_py.hour, ts_py.minute)

        # ---- arrivals ----
        ar = make_row(arr_cols)
        fill_common(ar, ts_py, target, total_out, wcat, weather_nums, queue, arr_lags, svc_lags)
        arr_raw = predict(arr_booster, arr_cols, ar)

        base_a = baseline_get_mds(baselines, "arr", month, dow, sid)
        # model 0.75 / baseline 0.25 + 分位クリップ
        arr_bl = blend_with_baseline(arr_raw, base_a, w_model=0.75, clip_min=0, clip_max=CLIP_ARR_MAX)
        arr_i = int(round(arr_bl))

        # ---- “上振れピーク判定”用：基準中央値との比較（追加入力なし） ----
        base_med_arr = float(base_a.get("median", 0.0)) if base_a is not None else 0.0
        peak_by_upshift = False
        if base_med_arr > 0:
            peak_by_upshift = (arr_i >= base_med_arr * (1.0 + upshift_margin))

        # update arr lags
        arr_lags = {"arr_Lg30": float(arr_i), "arr_Lg60": float(arr_lags["arr_Lg30"]), "arr_Lg90": float(arr_lags["arr_Lg60"])}

        # ---- service ----
        sr = make_row(multi_cols)
        fill_common(sr, ts_py, target, total_out, wcat, weather_nums, queue, arr_lags, svc_lags)
        set_if(sr, "reception_count", int(arr_i))
        svc_raw = predict(svc_booster, multi_cols, sr)

        base_s = baseline_get_mds(baselines, "svc", month, dow, sid)
        # serviceは過大だと短くなりやすいので baseline寄せ少し強め
        svc_bl = blend_with_baseline(svc_raw, base_s, w_model=0.70, clip_min=0, clip_max=CLIP_SVC_MAX)
        svc_i = int(round(svc_bl))

        svc_lags = {"svc_Lg30": float(svc_i), "svc_Lg60": float(svc_lags["svc_Lg30"]), "svc_Lg90": float(svc_lags["svc_Lg60"])}

        # ---- queue update ----
        queue = max(0.0, queue + float(arr_i) - float(svc_i))
        queue = float(min(queue, CLIP_QUEUE_MAX))

        # ---- wait ----
        wr = make_row(multi_cols)
        fill_common(wr, ts_py, target, total_out, wcat, weather_nums, queue, arr_lags, svc_lags)
        set_if(wr, "reception_count", int(arr_i))
        set_if(wr, "call_count", int(svc_i))
        wait_model = predict(wait_booster, multi_cols, wr)

        # physics proxy (queue / service * 30)
        phy = (float(queue) / max(float(svc_i), 1.0)) * float(FREQ_MIN)
        phy_cal = a * phy + b

        # peak判定：時間ピーク OR queue>=閾値 OR 受付が基準より上振れ
        peak_time = (is_peak_window(ts_py.hour, ts_py.minute) == 1)
        peak_queue = (queue >= queue_switch)
        peak = bool(peak_time or peak_queue or peak_by_upshift)

        alpha = alpha_peak if peak else alpha_base
        wait_blend = alpha * float(wait_model) + (1.0 - alpha) * float(phy_cal)

        base_w = baseline_get_mds(baselines, "wait", month, dow, sid)
        # waitは強く縛りすぎるとピークが削れるので w_model=0.90（弱めのbaseline利用）
        wait_blend = blend_with_baseline(wait_blend, base_w, w_model=0.90, clip_min=0, clip_max=CLIP_WAIT_MAX)
        wait_i = int(round(wait_blend))

        rows.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測処理数(人)": int(svc_i),
            "予測待ち人数(人)": int(round(queue)),
            "予測待ち時間(分)": int(wait_i),
            "参考_wait_model": float(wait_model),
            "参考_wait_phy_cal": float(phy_cal),
            "peak_time": int(peak_time),
            "peak_queue": int(peak_queue),
            "peak_upshift": int(peak_by_upshift),
        })

    return pd.DataFrame(rows)

def main():
    st.set_page_config(page_title="A病院 採血 混雑予測（最終・さらに強化）", layout="wide")
    st.title("🏥 A病院 採血 混雑予測（最終・さらに強化 / 追加入力なし）")
    st.caption("month×dow×slot baseline で安定化 + 上振れピーク判定 + ピークはphysics寄りで短すぎ抑制")

    required = [
        ARR_MODEL_PATH, SVC_MODEL_PATH, WAIT_MODEL_PATH,
        ARR_COLS_PATH, MULTI_COLS_PATH,
        BASELINE_PATH, CALIB_PATH
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("models/ に必要ファイルが不足しています:\n\n" + "\n".join([f"- {m}" for m in missing]))
        st.stop()

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気カテゴリ", WEATHER_CATS, index=0)

        with st.expander("気象（数値：任意。分からなければデフォルトでOK）", expanded=False):
            rain = st.number_input("降水量(mm)", value=0.0, step=0.1)
            tavg = st.number_input("平均気温(℃)", value=15.0, step=0.1)
            tmax = st.number_input("最高気温(℃)", value=18.0, step=0.1)
            tmin = st.number_input("最低気温(℃)", value=12.0, step=0.1)
            hum  = st.number_input("平均湿度(%)", value=60.0, step=1.0)
            wind = st.number_input("平均風速(m/s)", value=2.0, step=0.1)

        run = st.button("シミュレーション実行", type="primary")

    weather_nums = {
        "降水量": float(rain),
        "平均気温": float(tavg),
        "最高気温": float(tmax),
        "最低気温": float(tmin),
        "平均湿度": float(hum),
        "平均風速": float(wind),
    }

    if run:
        with st.spinner("計算中..."):
            df = simulate_one_day(target, int(total_out), str(weather), weather_nums)

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
            st.line_chart(df.set_index("時間帯")[["予測受付数","予測処理数(人)"]])

        st.info("ピーク判定は (1)8:30–11:00 (2)queue>=閾値 (3)受付が基準中央値より上振れ のORです。")

    st.divider()
    st.caption("※ 祝日CSV(data/syukujitsu.csv)があれば土日以外も休日扱いできます。")

if __name__ == "__main__":
    main()
