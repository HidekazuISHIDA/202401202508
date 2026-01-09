import json
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# ----------------------------
# 設定・パス定義
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

# モデルファイル (v6.0対応)
ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.json"
SVC_MODEL_PATH   = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"

ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH    = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"

# 祝日データ (あれば)
HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
INCLUDE_CLOSE = False

# ----------------------------
# 共通関数
# ----------------------------
def _load_holidays() -> set:
    if not HOLIDAY_CSV_PATH.exists(): return set()
    df = None
    for enc in ["cp932", "shift_jis", "utf-8", "utf-8-sig"]:
        try:
            df = pd.read_csv(HOLIDAY_CSV_PATH, encoding=enc, engine="python")
            break
        except: continue
    if df is None: return set()
    
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "日付", "国民の祝日・休日月日", "年月日"]:
            col = c
            break
    if not col: col = df.columns[0]
    return set(pd.to_datetime(df[col], errors="coerce").dropna().dt.date.tolist())

HOLIDAYS = _load_holidays()

def is_holiday(d: date) -> bool:
    if d.weekday() >= 5: return True
    if d in HOLIDAYS: return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3): return True
    return False

def week_of_month(d: date) -> int:
    return int((d.day - 1)//7 + 1)

def normalize_weather(s: str) -> str:
    t = str(s) if s else ""
    if "雪" in t: return "雪"
    if "雨" in t: return "雨"
    if "曇" in t: return "曇"
    if "晴" in t: return "晴"
    return "曇"

def month_weekday_counts(y, m):
    start = pd.Timestamp(year=y, month=m, day=1)
    end = start + pd.offsets.MonthEnd(1)
    days = pd.date_range(start, end)
    dow = days.dayofweek
    counts = {k:int((dow==k).sum()) for k in range(7)}
    total = sum(counts[k] for k in range(5))
    return counts, total

@st.cache_resource
def load_artifacts():
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))
    
    arr_bst = xgb.Booster()
    arr_bst.load_model(str(ARR_MODEL_PATH))
    svc_bst = xgb.Booster()
    svc_bst.load_model(str(SVC_MODEL_PATH))
    wait_bst = xgb.Booster()
    wait_bst.load_model(str(WAIT_MODEL_PATH))
    
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    return arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == "O": df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)

def _predict_booster(bst, cols, df):
    X = _coerce_numeric(df[cols].copy())
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    iter_range = (0, bst.best_iteration + 1) if getattr(bst, "best_iteration", None) else None
    return float(bst.predict(dmat, iteration_range=iter_range)[0])

def baseline_lookup(baseline, name, m, d, s):
    return float(baseline.get(name, {}).get(f"{int(m)}_{int(d)}_{int(s)}", 0.0))

def slot_index(ts):
    return int((ts.hour - OPEN_HOUR) * 2 + (ts.minute // 30))

def generate_slots(target_date):
    start = datetime.combine(target_date, datetime.min.time().replace(hour=OPEN_HOUR))
    end = datetime.combine(target_date, datetime.min.time().replace(hour=CLOSE_HOUR))
    rng = pd.date_range(start, end, freq=f"{FREQ_MIN}min")
    if INCLUDE_CLOSE: return list(rng)
    return [t.to_pydatetime() for t in rng if t.to_pydatetime() != end]

# ----------------------------
# シミュレーション (v6.0対応: 2乗特徴量 + 安全装置)
# ----------------------------
def simulate_one_day(target_date, total_pat, weather_text):
    arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib = load_artifacts()

    y, m, dow = target_date.year, target_date.month, target_date.weekday()
    is_h = int(is_holiday(target_date))
    prev_h = int(is_holiday(target_date - timedelta(days=1)))
    counts, w_total = month_weekday_counts(y, m)
    w_count_in_month = int(counts.get(dow, 0))
    w_ratio_in_month = float(w_count_in_month / w_total) if w_total > 0 else 0.0
    wcat = normalize_weather(weather_text)

    # State
    lags_arr = {"arr_lag_30":0.0, "arr_lag_60":0.0, "arr_lag_90":0.0}
    lags_svc = {"svc_lag_30":0.0, "svc_lag_60":0.0, "svc_lag_90":0.0}
    cum_arr, cum_svc, q_start = 0, 0, 0.0

    # Calib Params
    a, b = float(calib.get("a", 1.0)), float(calib.get("b", 0.0))
    alpha, floor_ratio = float(calib.get("alpha", 0.4)), float(calib.get("floor_ratio", 0.9))

    results = []
    
    for ts in generate_slots(target_date):
        slot = slot_index(ts)
        arr_base = baseline_lookup(baseline, "arr_base", m, dow, slot)
        svc_base = baseline_lookup(baseline, "svc_base", m, dow, slot)
        wait_base = baseline_lookup(baseline, "wait_base", m, dow, slot)

        def set_features(df_target):
            # Basic Features (No Year)
            df_target.loc[0, "month"] = m
            df_target.loc[0, "dayofweek"] = dow
            df_target.loc[0, "is_holiday"] = is_h
            df_target.loc[0, "前日祝日フラグ"] = prev_h
            df_target.loc[0, "月"] = m
            df_target.loc[0, "週回数"] = week_of_month(target_date)
            df_target.loc[0, "month_weekday_total"] = w_count_in_month
            df_target.loc[0, "weekday_count_in_month"] = w_count_in_month
            df_target.loc[0, "weekday_ratio_in_month"] = w_ratio_in_month
            df_target.loc[0, "total_outpatient_count"] = int(total_pat)
            
            # Weather
            df_target.loc[0, "雨フラグ"] = 1 if "雨" in wcat else 0
            df_target.loc[0, "雪フラグ"] = 1 if "雪" in wcat else 0
            for c in ["晴", "曇", "雨", "雪"]:
                if f"天気カテゴリ_{c}" in df_target.columns:
                    df_target.loc[0, f"天気カテゴリ_{c}"] = 1 if c == wcat else 0
            
            # Time & Slot
            df_target.loc[0, "hour"] = ts.hour
            df_target.loc[0, "minute"] = ts.minute
            if f"dayofweek_{dow}" in df_target.columns: df_target.loc[0, f"dayofweek_{dow}"] = 1
            df_target.loc[0, "is_first_slot"] = 1 if (ts.hour==8 and ts.minute==0) else 0
            df_target.loc[0, "is_second_slot"] = 1 if (ts.hour==8 and ts.minute==30) else 0
            df_target.loc[0, "slot"] = slot
            
            # Dynamic State (Queue)
            df_target.loc[0, "queue_at_start_truth"] = float(q_start)
            
            # ★ v6.0 新機能: Queueの2乗 (雪だるま式増加を表現)
            if "queue_squared" in df_target.columns:
                df_target.loc[0, "queue_squared"] = float(q_start) ** 2
            
            # Lags & Cumulative
            df_target.loc[0, "arr_lag_30"] = float(lags_arr["arr_lag_30"])
            df_target.loc[0, "arr_lag_60"] = float(lags_arr["arr_lag_60"])
            df_target.loc[0, "arr_lag_90"] = float(lags_arr["arr_lag_90"])
            df_target.loc[0, "arr_roll_60"] = float((lags_arr["arr_lag_30"]+lags_arr["arr_lag_60"])/2)
            
            df_target.loc[0, "svc_lag_30"] = float(lags_svc["svc_lag_30"])
            df_target.loc[0, "svc_lag_60"] = float(lags_svc["svc_lag_60"])
            df_target.loc[0, "svc_lag_90"] = float(lags_svc["svc_lag_90"])
            df_target.loc[0, "svc_roll_60"] = float((lags_svc["svc_lag_30"]+lags_svc["svc_lag_60"])/2)
            
            df_target.loc[0, "cum_arrivals"] = int(cum_arr)
            df_target.loc[0, "cum_service"] = int(cum_svc)

        # 1. Arrivals
        cf = _make_zero_df(arr_cols)
        set_features(cf)
        arr_i = max(0, int(round(_predict_booster(arr_bst, arr_cols, cf))))

        # 2. Service
        mf = _make_zero_df(multi_cols)
        set_features(mf)
        svc_i = max(0, int(round(_predict_booster(svc_bst, multi_cols, mf))))
        
        # ★ 安全装置1: 行列があるなら最低1人は処理させる (幽霊行列防止)
        if q_start >= 0.5 and svc_i == 0:
            svc_i = 1

        # Queue Update
        q_next = max(0.0, float(q_start) + float(arr_i) - float(svc_i))

        # 3. Wait (Log Transform -> expm1)
        raw_wait = _predict_booster(wait_bst, multi_cols, mf)
        pred_wait_ai = max(0.0, float(np.expm1(raw_wait)))
        
        # Physics Wait (Queue / Service)
        safe_svc = max(float(svc_i), 0.5)
        wait_phy = (float(q_start) / safe_svc) * 30.0
        wait_phy = min(wait_phy, 300.0) # Cap
        
        # Calibration
        wait_phy_calib = max(0.0, a * wait_phy + b)
        
        # Ensemble
        wait_blend = alpha * pred_wait_ai + (1.0 - alpha) * wait_phy_calib
        
        # ★ 安全装置2: 行列なしなら待ち時間0
        if q_start < 0.5:
            wait_final = 0.0
        else:
            # Baseline Floor (過去の実績を下回らないように)
            wait_final = max(float(wait_base)*floor_ratio, wait_blend)

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測呼出数(処理数)": int(svc_i),
            "予測待ち人数(人)": int(round(q_next)),
            "予測平均待ち時間(分)": int(round(wait_final))
        })

        # Update State
        lags_arr = {"arr_lag_30": float(arr_i), "arr_lag_60": lags_arr["arr_lag_30"], "arr_lag_90": lags_arr["arr_lag_60"]}
        lags_svc = {"svc_lag_30": float(svc_i), "svc_lag_60": lags_svc["svc_lag_30"], "svc_lag_90": lags_svc["svc_lag_60"]}
        cum_arr += int(arr_i)
        cum_svc += int(svc_i)
        q_start = q_next

    return pd.DataFrame(results)

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="A病院 混雑予測", layout="wide")
    st.title("🏥 A病院 採血 待ち時間予測AI (v6.0)")
    st.caption("Mixed Model: AI(Log/Strict) + Physics(Queue^2)")

    # Check Files
    required = [ARR_MODEL_PATH, SVC_MODEL_PATH, WAIT_MODEL_PATH, ARR_COLS_PATH, MULTI_COLS_PATH, BASELINE_PATH, CALIB_PATH]
    if any(not p.exists() for p in required):
        st.error("モデルファイルが不足しています。models/フォルダを確認してください。")
        st.stop()

    with st.sidebar:
        st.header("条件設定")
        tdate = st.date_input("日付", value=date.today() + timedelta(days=1))
        pat_num = st.number_input("予測外来患者数", value=1400, step=50, help="平日平均: 1200-1500人")
        weather = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
        run = st.button("予測実行", type="primary")
        
        st.divider()
        st.info("v6.0 Update:\n・混雑時の感度を10倍に強化\n・行列の2乗則を考慮")

    if run:
        with st.spinner("AIがシミュレーション中..."):
            df = simulate_one_day(tdate, int(pat_num), str(weather))
        
        st.success(f"✅ {tdate.strftime('%Y/%m/%d')} の予測完了")
        
        # Metrics
        peak_wait = df["予測平均待ち時間(分)"].max()
        peak_idx = df["予測平均待ち時間(分)"].idxmax()
        peak_time = df.loc[peak_idx, "時間帯"]
        max_q = df["予測待ち人数(人)"].max()
        total_arr = df["予測受付数"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("最大待ち時間", f"{peak_wait} 分", f"@{peak_time}", delta_color="inverse")
        c2.metric("最大待ち人数", f"{max_q} 人")
        c3.metric("総受付数", f"{total_arr} 人")
        
        # Chart
        st.subheader("混雑推移")
        chart_data = df.set_index("時間帯")[["予測平均待ち時間(分)", "予測待ち人数(人)"]]
        st.line_chart(chart_data)
        
        # Table
        with st.expander("詳細データ"):
            st.dataframe(df.style.highlight_max(axis=0, color="#fffdc9"), use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSVダウンロード", csv, f"pred_{tdate}.csv", "text/csv")

if __name__ == "__main__":
    main()
