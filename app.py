import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import datetime
from pathlib import Path

# --- 設定 ---
MODELS_DIR = Path("models") 
META_PATH = MODELS_DIR / "model_meta.json"
ARR_MODEL_PATH = MODELS_DIR / "lgb_arrival.txt"
SVC_MODEL_PATH = MODELS_DIR / "lgb_service.txt"

# --- モデル読み込み ---
@st.cache_resource
def load_models():
    if not META_PATH.exists(): return None, None, None
    
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    
    # LightGBMモデル読み込み
    bst_arr = lgb.Booster(model_file=str(ARR_MODEL_PATH))
    bst_svc = lgb.Booster(model_file=str(SVC_MODEL_PATH))
    
    return bst_arr, bst_svc, meta

# --- シミュレーション ---
def predict_day(date_val, total_pat, weather_idx, bst_arr, bst_svc, meta):
    features = meta["features"]
    cfg = meta["config"]
    
    # 時間枠生成
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    w_labels = ["晴", "曇", "雨", "雪"]
    weather_text = w_labels[weather_idx]
    
    results = []
    current_queue = 0
    
    for ts in timestamps:
        # 特徴量作成
        row = {}
        row["month"] = ts.month
        row["dow"] = ts.dayofweek
        # 簡易休日判定
        is_hol = 1 if ts.dayofweek >= 5 or (ts.month==1 and ts.day<=3) else 0
        row["is_holiday"] = is_hol
        row["week_of_month"] = (ts.day - 1) // 7 + 1
        row["hour"] = ts.hour
        row["minute"] = ts.minute
        row["slot_id"] = (ts.hour * 60 + ts.minute) // 30
        row["total_outpatient"] = total_pat 
        
        # 気象 (簡易補完)
        row["rain"] = 5.0 if weather_text == "雨" else 0.0
        row["temp"] = 5.0 if ts.month in [12, 1, 2] else 15.0
        for w in w_labels:
            row[f"is_{w}"] = 1 if weather_text == w else 0
            
        # 予測実行 (DataFrameで渡すのが安全)
        X_df = pd.DataFrame([row])[features]
        pred_arr = max(0, bst_arr.predict(X_df)[0])
        pred_svc = max(0, bst_svc.predict(X_df)[0])
        
        # --- 物理シミュレーション ---
        # 行列 = 前の行列 + 到着 - 処理
        actual_processed = min(current_queue + pred_arr, pred_svc)
        next_queue = current_queue + pred_arr - actual_processed
        
        # 待ち時間推定
        if actual_processed < 0.1:
            wait_time = 0 if next_queue < 1 else 30 
        else:
            wait_time = (next_queue / actual_processed) * 30.0
            
        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": round(pred_arr),
            "予測呼出数": round(pred_svc),
            "予測待ち人数": round(next_queue),
            "予測待ち時間(分)": round(wait_time)
        })
        current_queue = next_queue
        
    return pd.DataFrame(results)

# --- UI ---
def main():
    st.set_page_config(page_title="A病院 混雑予測 AI v9.0", layout="centered")
    st.title("🏥 混雑予測システム v9.0")
    st.caption("LightGBM Model")
    
    bst_arr, bst_svc, meta = load_models()
    
    if bst_arr is None:
        st.error("モデルファイルが見つかりません。modelsフォルダを確認してください。")
        st.stop()
        
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_input = st.date_input("日付", value=datetime.date(2026, 1, 9))
        with col2:
            weather_idx = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
            
        pat_num = st.number_input("予想来院数 (人)", value=1300, step=50)
        submitted = st.form_submit_button("予測実行")
        
    if submitted:
        df_res = predict_day(date_input, pat_num, weather_idx, bst_arr, bst_svc, meta)
        
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        
        st.success("予測完了")
        c1, c2, c3 = st.columns(3)
        c1.metric("最大待ち時間", f"{peak['予測待ち時間(分)']} 分", f"@{peak['時間帯']}")
        c2.metric("最大行列", f"{peak['予測待ち人数']} 人")
        c3.metric("ピーク受付", f"{peak['予測受付数']} 人")
        
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測待ち人数"]])
        st.dataframe(df_res)
        st.download_button("CSVダウンロード", df_res.to_csv(index=False).encode('utf-8-sig'), "predict.csv")

if __name__ == "__main__":
    main()
