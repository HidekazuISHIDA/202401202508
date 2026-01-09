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
ARR_COLS_PATH  = MODELS_DIR / "columns_A_timeseries.json"

SVC_MODEL_PATH = MODELS_DIR / "model_A_service_30min.json"
WAIT_MEAN_PATH = MODELS_DIR / "model_A_waittime_30min.json"
WAIT_P90_PATH  = MODELS_DIR / "model_A_waittime_p90_30min.json"

MULTI_COLS_PATH = MODELS_DIR / "columns_A_multi_30min.json"
BASELINE_PATH   = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH      = MODELS_DIR / "wait_calibration.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"  # optional

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
SLOT_MIN = 30.0
WAIT_MAX = 180.0

WEATHER_CATS = ["晴","曇","雨","雪"]
PEAK_START = (8,30)
PEAK_END   = (11,0)

CONGESTION_CLAMP = (0.90, 1.20)  # 朝混雑係数は弱く

def in_peak(ts: datetime) -> bool:
    h, m = ts.hour, ts.minute
    after = (h > PEAK_START[0]) or (h == PEAK_START[0] and m >= PEAK_START[1])
    before= (h < PEAK_END[0]) or (h == PEAK_END[0] and m <= PEAK_END[1])
    return after and before

def _load_holidays() -> set:
    if not HOLIDAY_CSV_PATH.exists():
        return set()
    df = pd.read_csv(HOLIDAY_CSV_PATH, encoding="utf-8", engine="python")
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date","日付"]:
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
    return int((d.day - 1)//7 + 1)

def normalize_weather(w: str) -> str:
    s = str(w) if w is not None else ""
    if "雪" in s: return "雪"
    if "雨" in s: return "雨"
    if "曇" in s: return "曇"
    if "晴" in s: return "晴"
    return "曇"

def slot_id_from_ts(ts: datetime) -> int:
    return int(((ts.hour*60 + ts.minute) - (OPEN_HOUR*60)) // FREQ_MIN)

def baseline_key(month: int, dow: int, slot_id: int) -> str:
    return f"{int(month)}_{int(dow)}_{int(slot_id)}"

@st.cache_resource
def load_assets():
    arr_cols   = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    bst_arr = xgb.Booster(); bst_arr.load_model(str(ARR_MODEL_PATH))
    bst_svc = xgb.Booster(); bst_svc.load_model(str(SVC_MODEL_PATH))
    bst_wm  = xgb.Booster(); bst_wm.load_model(str(WAIT_MEAN_PATH))
    bst_wp  = xgb.Booster(); bst_wp.load_model(str(WAIT_P90_PATH))

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    alpha_base = float(calib.get("alpha_base", 0.65))
    alpha_peak = float(calib.get("alpha_peak", 0.30))

    return bst_arr, arr_cols, bst_svc, bst_wm, bst_wp, multi_cols, baseline, (a,b,alpha_base,alpha_peak)

def _make_zero_df(cols):
    return pd.DataFrame({c:[0] for c in cols})

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def _predict(bst: xgb.Booster, cols, dfrow: pd.DataFrame) -> float:
    X = _coerce_numeric(dfrow[cols].copy())
    d = xgb.DMatrix(X, feature_names=list(cols))
    return float(bst.predict(d)[0])

def add_common(dfrow: pd.DataFrame, ts: datetime, target_date: date):
    dow = ts.weekday()
    month = ts.month
    sid = slot_id_from_ts(ts)

    for k in range(7):
        c = f"dayofweek_{k}"
        if c in dfrow.columns:
            dfrow.loc[0, c] = 1 if k == dow else 0

    if "hour" in dfrow.columns: dfrow.loc[0,"hour"] = ts.hour
    if "minute" in dfrow.columns: dfrow.loc[0,"minute"] = ts.minute
    if "dow" in dfrow.columns: dfrow.loc[0,"dow"] = dow
    if "month" in dfrow.columns: dfrow.loc[0,"month"] = month
    if "slot_id" in dfrow.columns: dfrow.loc[0,"slot_id"] = sid

    if "月" in dfrow.columns: dfrow.loc[0,"月"] = month
    if "週回数" in dfrow.columns: dfrow.loc[0,"週回数"] = week_of_month(target_date)

    if "is_first_slot" in dfrow.columns:
        dfrow.loc[0,"is_first_slot"] = 1 if (ts.hour==8 and ts.minute==0) else 0
    if "is_second_slot" in dfrow.columns:
        dfrow.loc[0,"is_second_slot"] = 1 if (ts.hour==8 and ts.minute==30) else 0

    return sid, month, dow

def add_calendar(dfrow: pd.DataFrame, d: date):
    if "is_holiday" in dfrow.columns:
        dfrow.loc[0,"is_holiday"] = int(is_holiday(d))
    prev = d - timedelta(days=1)
    if "前日祝日フラグ" in dfrow.columns:
        dfrow.loc[0,"前日祝日フラグ"] = int(is_holiday(prev))

def add_weather(dfrow: pd.DataFrame, w: str):
    w = normalize_weather(w)
    if "雨フラグ" in dfrow.columns: dfrow.loc[0,"雨フラグ"] = 1 if w=="雨" else 0
    if "雪フラグ" in dfrow.columns: dfrow.loc[0,"雪フラグ"] = 1 if w=="雪" else 0
    for cat in WEATHER_CATS:
        c = f"天気カテゴリ_{cat}"
        if c in dfrow.columns:
            dfrow.loc[0,c] = 1 if w==cat else 0

def add_outpatient(dfrow: pd.DataFrame, total_out: int):
    if "total_outpatient_count" in dfrow.columns:
        dfrow.loc[0,"total_outpatient_count"] = int(total_out)

def add_state(dfrow: pd.DataFrame, queue0: float, cum_arr: float):
    if "queue_at_start_truth" in dfrow.columns:
        dfrow.loc[0,"queue_at_start_truth"] = float(queue0)
    if "cum_arrivals_sofar" in dfrow.columns:
        dfrow.loc[0,"cum_arrivals_sofar"] = float(cum_arr)

def add_lags(dfrow: pd.DataFrame, lags: dict, roll60_name: str):
    for k,v in lags.items():
        if k in dfrow.columns:
            dfrow.loc[0,k] = float(v)
    if roll60_name in dfrow.columns:
        # expects *_lag_30 and *_lag_60
        k30 = [k for k in lags.keys() if k.endswith("lag_30")][0]
        k60 = [k for k in lags.keys() if k.endswith("lag_60")][0]
        dfrow.loc[0,roll60_name] = (float(lags[k30])+float(lags[k60]))/2.0

def get_baseline(baseline: dict, month: int, dow: int, sid: int):
    k = baseline_key(month,dow,sid)
    arr = baseline.get("arr",{}).get(k,{"median":0,"p95":0})
    svc = baseline.get("svc",{}).get(k,{"median":0,"p95":0})
    wm  = baseline.get("wait_mean",{}).get(k,{"median":0,"p95":0})
    wp  = baseline.get("wait_p90",{}).get(k,{"median":0,"p95":0})
    return arr, svc, wm, wp

def simulate_one_day(target_date: date, total_out: int, weather: str) -> pd.DataFrame:
    bst_arr, arr_cols, bst_svc, bst_wm, bst_wp, multi_cols, baseline, calib = load_assets()
    a, b, alpha_base, alpha_peak = calib

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min").to_pydatetime().tolist()
    # drop 18:00
    if slots and slots[-1].hour==CLOSE_HOUR and slots[-1].minute==0:
        slots = slots[:-1]

    arr_lags = {"arr_lag_30":0.0,"arr_lag_60":0.0,"arr_lag_90":0.0}
    svc_lags = {"svc_lag_30":0.0,"svc_lag_60":0.0,"svc_lag_90":0.0}

    queue0 = 0.0
    cum_arr = 0.0

    # 朝混雑係数（8:00+8:30の受付がベースラインより多いか）
    congestion_factor = 1.0
    first_two = []

    rows=[]
    for ts in slots:
        sid, month, dow = slot_id_from_ts(ts), ts.month, ts.weekday()
        base_arr, base_svc, base_wm, base_wp = get_baseline(baseline, month, dow, sid)

        # -------- arrivals (raw) --------
        af = _make_zero_df(arr_cols)
        add_common(af, ts, target_date)
        add_calendar(af, target_date)
        add_outpatient(af, total_out)
        add_weather(af, weather)
        add_state(af, queue0, cum_arr)
        add_lags(af, arr_lags, "arr_roll_60")
        add_lags(af, svc_lags, "svc_roll_60")

        pred_arr = max(0.0, _predict(bst_arr, arr_cols, af))
        arr_i = int(round(pred_arr))

        if sid in (0,1):
            first_two.append((arr_i, max(1.0, float(base_arr.get("median",0.0)))))
            if len(first_two)==2:
                ratio = (first_two[0][0]+first_two[1][0])/(first_two[0][1]+first_two[1][1])
                congestion_factor = float(np.clip(ratio, CONGESTION_CLAMP[0], CONGESTION_CLAMP[1]))

        # peak: mild push
        if in_peak(ts):
            arr_i = int(round(arr_i * congestion_factor))

        # -------- service (log1p -> expm1) --------
        sf = _make_zero_df(multi_cols)
        add_common(sf, ts, target_date)
        add_calendar(sf, target_date)
        add_outpatient(sf, total_out)
        add_weather(sf, weather)
        add_state(sf, queue0, cum_arr)
        add_lags(sf, arr_lags, "arr_roll_60")
        add_lags(sf, svc_lags, "svc_roll_60")

        pred_svc_log = _predict(bst_svc, multi_cols, sf)
        svc_i = int(round(max(0.0, np.expm1(pred_svc_log))))

        available = int(round(queue0)) + int(arr_i)

        # ①保存則: service <= available
        svc_i = min(svc_i, max(0, available))

        # ② baseline p95 cap (service暴れ防止)
        svc_cap = int(round(float(base_svc.get("p95", svc_i))))
        svc_i = min(svc_i, max(0, min(available, svc_cap)))

        # ③ fallback: service=0でも人がいるならbaseline median
        if svc_i==0 and available>0:
            svc_i = int(round(min(available, max(1.0, float(base_svc.get("median",1.0))))))

        # -------- queue (conservation) --------
        queue_end = max(0.0, queue0 + float(arr_i) - float(svc_i))

        # -------- wait mean/p90 (log1p -> expm1) --------
        wf = _make_zero_df(multi_cols)
        add_common(wf, ts, target_date)
        add_calendar(wf, target_date)
        add_outpatient(wf, total_out)
        add_weather(wf, weather)
        add_state(wf, queue0, cum_arr)
        add_lags(wf, arr_lags, "arr_roll_60")
        add_lags(wf, svc_lags, "svc_roll_60")

        wait_model = max(0.0, np.expm1(_predict(bst_wm, multi_cols, wf)))
        wait_p90m  = max(0.0, np.expm1(_predict(bst_wp, multi_cols, wf)))

        # physics wait
        svc_for_phy = max(1.0, float(svc_i))
        wait_phy = (float(queue0)/svc_for_phy)*SLOT_MIN
        wait_phy_cal = max(0.0, a*wait_phy + b)

        alpha = alpha_peak if in_peak(ts) else alpha_base
        wait_mean = alpha*wait_model + (1.0-alpha)*wait_phy_cal

        # peak: mild congestion push
        if in_peak(ts):
            wait_mean *= congestion_factor
            wait_p90m  *= congestion_factor

        wait_mean = float(np.clip(wait_mean, 0.0, WAIT_MAX))
        wait_p90m  = float(np.clip(wait_p90m,  0.0, WAIT_MAX))

        rows.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測処理数(呼出数)": int(svc_i),
            "予測待ち人数(人)": int(round(queue0)),
            "予測平均待ち時間(分)": int(round(wait_mean)),
            "予測混雑時待ち時間_p90(分)": int(round(wait_p90m)),
        })

        # update
        cum_arr += float(arr_i)
        arr_lags = {"arr_lag_30": float(arr_i), "arr_lag_60": float(arr_lags["arr_lag_30"]), "arr_lag_90": float(arr_lags["arr_lag_60"])}
        svc_lags = {"svc_lag_30": float(svc_i), "svc_lag_60": float(svc_lags["svc_lag_30"]), "svc_lag_90": float(svc_lags["svc_lag_60"])}
        queue0 = queue_end

    return pd.DataFrame(rows), congestion_factor

def main():
    st.set_page_config(page_title="A病院 採血 待ち予測（最良版）", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（再学習対応・最良版）")
    st.caption("追加入力なし。serviceは保存則＋baseline p95で暴れを抑制。ピーク(8:30–11:00)は物理待ち寄りにブレンドし過小推定を抑えます。")

    required = [ARR_MODEL_PATH,ARR_COLS_PATH,SVC_MODEL_PATH,WAIT_MEAN_PATH,WAIT_P90_PATH,MULTI_COLS_PATH,BASELINE_PATH,CALIB_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("models/ に必要ファイルが不足しています:\n\n" + "\n".join([f"- {m}" for m in missing]))
        st.stop()

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気（簡易）", ["晴","曇","雨","雪","快晴","薄曇"], index=0)
        run = st.button("シミュレーション実行", type="primary")

    if run:
        with st.spinner("計算中..."):
            df, cf = simulate_one_day(target, int(total_out), str(weather))
        st.success(f"{target} の予測が完了しました。混雑係数（朝9時推定）: {cf:.2f}")

        c1, c2 = st.columns([2,3], gap="large")
        with c1:
            st.subheader("結果テーブル")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("CSVダウンロード", data=csv, file_name=f"A_predict_{target}.csv", mime="text/csv")

        with c2:
            st.subheader("可視化")
            st.line_chart(df.set_index("時間帯")[["予測平均待ち時間(分)","予測混雑時待ち時間_p90(分)"]])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

if __name__ == "__main__":
    main()
