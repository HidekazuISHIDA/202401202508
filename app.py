import json
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# ----------------------------
# 設定
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

# モデルファイル (UBJ形式)
ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.ubj"
SVC_MODEL_PATH   = MODELS_DIR / "model_A_service_30min.ubj"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.ubj"

ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"
BASELINE_PATH    = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"
HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR, CLOSE_HOUR, FREQ_MIN = 8, 18, 30

# ----------------------------
# 共通関数
# ----------------------------
def _load_holidays() -> set:
    if not HOLIDAY_CSV_PATH.exists(): return set()
    df = None
    for enc in ["cp932", "shift_jis", "utf-8", "utf-8-sig"]:
        try:
            df = pd.read_csv(HOLIDAY_CSV_PATH, encoding=enc, engine="python"); break
        except: continue
    if df is None: return set()
    col = next((c for c in df.columns if str(c).strip().lower() in ["date", "日付", "国民の祝日・休日月日", "年月日"]), df.columns[0])
    return set(pd.to_datetime(df[col], errors="coerce").dropna().dt.date.tolist())

HOLIDAYS = _load_holidays()

def is_holiday(d: date) -> bool:
    if d.weekday() >= 5: return True
    if d in HOLIDAYS: return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3): return True
    return False

def week_of_month(d: date) -> int: return int((d.day - 1)//7 + 1)

def normalize_weather(s: str) -> str:
    t = str(s) if s else ""
    for w in ["雪", "雨", "曇", "晴"]:
        if w in t: return w
    return "曇"

def month_weekday_counts(y, m):
    start = pd.Timestamp(year=y, month=m, day=1)
    end = start + pd.offsets.MonthEnd(1)
    days = pd.date_range(start, end)
    counts = {k:int((days.dayofweek==k).sum()) for k in range(7)}
    return counts, sum(counts[k] for k in range(5))

# ★重要：気象データの自動補完 (これが無いと予測が0になる)
def get_default_weather_metrics(month, weather_text):
    monthly_temps = {1:5.0, 2:6.0, 3:10.0, 4:15.0, 5:20.0, 6:24.0, 7:28.0, 8:29.0, 9:25.0, 10:19.0, 11:13.0, 12:7.0}
    temp = monthly_temps.get(month, 15.0)
    rain, hum, wind = 0.0, 60.0, 2.0
    
    if "雨" in weather_text: 
        rain, hum, temp = 5.0, 85.0, temp-2.0
    elif "雪" in weather_text: 
        rain, hum, temp = 2.0, 80.0, max(temp-5.0, 0.0)
    elif "晴" in weather_text: 
        hum, temp = 50.0, temp+2.0
        
    return {
        "平均気温": temp, "最高気温": temp+5, "最低気温": temp-5, 
        "降水量": rain, "平均湿度": hum, "平均風速": wind
    }

@st.cache_resource
def load_artifacts():
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))
    arr_bst = xgb.Booster(); arr_bst.load_model(str(ARR_MODEL_PATH))
    svc_bst = xgb.Booster(); svc_bst.load_model(str(SVC_MODEL_PATH))
    wait_bst = xgb.Booster(); wait_bst.load_model(str(WAIT_MODEL_PATH))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    return arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib

def _make_zero_df(cols): return pd.DataFrame({c: [0] for c in cols})
def _coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == "O": df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)
def _predict_booster(bst, cols, df):
    X = _coerce_numeric(df[cols].copy())
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    return float(bst.predict(dmat)[0])
def baseline_lookup(baseline, name, m, d, s):
    return float(baseline.get(name, {}).get(f"{int(m)}_{int(d)}_{int(s)}", 0.0))
def slot_index(ts): return int((ts.hour - OPEN_HOUR) * 2 + (ts.minute // 30))
def generate_slots(target_date):
    start = datetime.combine(target_date, datetime.min.time().replace(hour=OPEN_HOUR))
    end = datetime.combine(target_date, datetime.min.time().replace(hour=CLOSE_HOUR))
    rng = pd.date_range(start, end, freq=f"30min")
    return [t.to_pydatetime() for t in rng if t.to_pydatetime() != end]

# ----------------------------
# シミュレーション (Pure AI)
# ----------------------------
def simulate_one_day(target_date, total_pat, weather_text):
    arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib = load_artifacts()

    y, m, dow = target_date.year, target_date.month, target_date.weekday()
    is_h, prev_h = int(is_holiday(target_date)), int(is_holiday(target_date - timedelta(days=1)))
    counts, w_total = month_weekday_counts(y, m)
    w_count, w_ratio = int(counts.get(dow, 0)), float(counts.get(dow, 0) / w_total) if w_total > 0 else 0.0
    wcat = normalize_weather(weather_text)
    w_metrics = get_default_weather_metrics(m, wcat)

    lags_arr = {"arr_lag_30":0.0, "arr_lag_60":0.0, "arr_lag_90":0.0}
    lags_svc = {"svc_lag_30":0.0, "svc_lag_60":0.0, "svc_lag_90":0.0}
    cum_arr, cum_svc, q_start = 0, 0, 0.0
    
    # Calibration (Pure AI -> alpha=1.0)
    a, b = float(calib.get("a", 1.0)), float(calib.get("b", 0.0))
    alpha, floor_ratio = float(calib.get("alpha", 1.0)), float(calib.get("floor_ratio", 1.0))

    results = []
    
    for ts in generate_slots(target_date):
        slot = slot_index(ts)
        
        def make_base_df(cols_list):
            df = _make_zero_df(cols_list)
            # Basic Features
            df.loc[0, "month"] = m
            df.loc[0, "dayofweek"] = dow
            df.loc[0, "is_holiday"] = is_h
            df.loc[0, "前日祝日フラグ"] = prev_h
            df.loc[0, "月"] = m
            df.loc[0, "週回数"] = week_of_month(target_date)
            df.loc[0, "month_weekday_total"] = w_count
            df.loc[0, "weekday_count_in_month"] = w_count
            df.loc[0, "weekday_ratio_in_month"] = w_ratio
            df.loc[0, "total_outpatient_count"] = int(total_pat)
            
            # Weather (Flags + Metrics)
            df.loc[0, "雨フラグ"] = 1 if "雨" in wcat else 0
            df.loc[0, "雪フラグ"] = 1 if "雪" in wcat else 0
            for c in ["晴", "曇", "雨", "雪"]:
                if f"天気カテゴリ_{c}" in df.columns: df.loc[0, f"天気カテゴリ_{c}"] = 1 if c == wcat else 0
            for k, v in w_metrics.items():
                if k in df.columns: df.loc[0, k] = v
            
            # Time
            df.loc[0, "hour"] = ts.hour
            df.loc[0, "minute"] = ts.minute
            if f"dayofweek_{dow}" in df.columns: df.loc[0, f"dayofweek_{dow}"] = 1
            df.loc[0, "is_first_slot"] = 1 if (ts.hour==8 and ts.minute==0) else 0
            df.loc[0, "is_second_slot"] = 1 if (ts.hour==8 and ts.minute==30) else 0
            df.loc[0, "slot"] = slot
            
            # Dynamic
            df.loc[0, "queue_at_start_truth"] = float(q_start)
            if "queue_squared" in df.columns: df.loc[0, "queue_squared"] = float(q_start) ** 2
            
            # Lags
            df.loc[0, "arr_lag_30"] = float(lags_arr["arr_lag_30"])
            df.loc[0, "arr_lag_60"] = float(lags_arr["arr_lag_60"])
            df.loc[0, "arr_lag_90"] = float(lags_arr["arr_lag_90"])
            df.loc[0, "arr_roll_60"] = float((lags_arr["arr_lag_30"]+lags_arr["arr_lag_60"])/2)
            df.loc[0, "svc_lag_30"] = float(lags_svc["svc_lag_30"])
            df.loc[0, "svc_lag_60"] = float(lags_svc["svc_lag_60"])
            df.loc[0, "svc_lag_90"] = float(lags_svc["svc_lag_90"])
            df.loc[0, "svc_roll_60"] = float((lags_svc["svc_lag_30"]+lags_svc["svc_lag_60"])/2)
            df.loc[0, "cum_arrivals"] = int(cum_arr)
            df.loc[0, "cum_service"] = int(cum_svc)
            
            # Baseline
            for t, n in [("arr_base", "arr_base"), ("svc_base", "svc_base"), ("wait_base", "wait_base")]:
                if n in df.columns: df.loc[0, n] = baseline_lookup(baseline, t, m, dow, slot)
            return df

        # 1. Arrivals
        cf = make_base_df(arr_cols)
        arr_i = max(0, int(round(_predict_booster(arr_bst, arr_cols, cf))))

        # 2. Service
        mf = make_base_df(multi_cols)
        # v7 Features
        if "arr_diff" in mf.columns:
            mf.loc[0, "arr_diff"] = float(arr_i) - float(lags_arr["arr_lag_30"])
        if "queue_density" in mf.columns:
            mf.loc[0, "queue_density"] = float(q_start) / (float(arr_i) + 1.0)

        svc_i = max(0, int(round(_predict_booster(svc_bst, multi_cols, mf))))
        
        # 安全装置: 幽霊行列防止のみ残す（これは物理的な整合性のため必須）
        if q_start >= 0.5 and svc_i == 0: svc_i = 1

        q_next = max(0.0, float(q_start) + float(arr_i) - float(svc_i))

        # 3. Wait (Pure AI)
        raw_wait = _predict_booster(wait_bst, multi_cols, mf)
        pred_wait_ai = max(0.0, float(np.expm1(raw_wait)))
        
        # 物理モデル計算（参考用・ブレンドには使わないが念のため計算）
        safe_svc = max(float(svc_i), 0.5)
        wait_phy = min((float(q_start) / safe_svc) * 30.0, 300.0)
        wait_phy_calib = max(0.0, a * wait_phy + b)
        
        # アンサンブル (alpha=1.0 なら AIのみ)
        wait_blend = alpha * pred_wait_ai + (1.0 - alpha) * wait_phy_calib
        
        # 行列なしなら待ち時間0（これは物理的真理）
        wait_final = 0.0 if q_start < 0.5 else wait_blend

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測呼出数": int(svc_i),
            "予測待ち人数": int(round(q_next)),
            "予測平均待ち時間(分)": int(round(wait_final))
        })

        lags_arr = {"arr_lag_30": float(arr_i), "arr_lag_60": lags_arr["arr_lag_30"], "arr_lag_90": lags_arr["arr_lag_60"]}
        lags_svc = {"svc_lag_30": float(svc_i), "svc_lag_60": lags_svc["svc_lag_30"], "svc_lag_90": lags_svc["svc_lag_60"]}
        cum_arr += int(arr_i)
        cum_svc += int(svc_i)
        q_start = q_next

    return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="A病院 混雑予測", layout="wide")
    st.title("🏥 A病院 採血 待ち時間予測AI")
    st.caption("Pure AI Model (No Calibration)")

    required = [ARR_MODEL_PATH, SVC_MODEL_PATH, WAIT_MODEL_PATH, ARR_COLS_PATH, MULTI_COLS_PATH, BASELINE_PATH]
    if any(not p.exists() for p in required):
        st.error("モデルファイル不足: modelsフォルダを確認してください")
        st.stop()

    with st.sidebar:
        st.header("条件設定")
        tdate = st.date_input("日付", value=date.today() + timedelta(days=1))
        pat_num = st.number_input("予測外来患者数 (予定)", value=1300, step=50)
        weather = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
        run = st.button("予測実行", type="primary")

    if run:
        with st.spinner("AI演算中..."):
            df = simulate_one_day(tdate, int(pat_num), str(weather))
        
        st.success(f"✅ {tdate.strftime('%Y/%m/%d')} 予測完了")
        
        peak_wait = df["予測平均待ち時間(分)"].max()
        peak_time = df.loc[df["予測平均待ち時間(分)"].idxmax(), "時間帯"]
        max_q = df["予測待ち人数"].max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最大待ち時間", f"{peak_wait} 分", f"@{peak_time}", delta_color="inverse")
        c2.metric("最大待ち人数", f"{max_q} 人")
        c3.metric("受付/処理", f"{df['予測受付数'].sum()} / {df['予測呼出数'].sum()}")
        
        st.subheader("混雑推移")
        st.line_chart(df.set_index("時間帯")[["予測平均待ち時間(分)", "予測待ち人数"]])
        
        with st.expander("詳細データ"):
            st.dataframe(df.style.highlight_max(axis=0, color="#fffdc9"), use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSVダウンロード", csv, f"pred_{tdate}.csv", "text/csv")

if __name__ == "__main__":
    main()
