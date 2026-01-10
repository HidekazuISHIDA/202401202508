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
    with open(META_PATH, "r") as f: meta = json.load(f)
    bst_arr = lgb.Booster(model_file=str(ARR_MODEL_PATH))
    bst_svc = lgb.Booster(model_file=str(SVC_MODEL_PATH))
    return bst_arr, bst_svc, meta

# --- シミュレーション (v11.0: 物理法則完全準拠) ---
def predict_day(date_val, total_pat, weather_text, bst_arr, bst_svc, meta):
    features = meta["features"]
    cfg = meta["config"]
    
    # 時間枠生成 (8:00 start)
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    w_labels = ["晴", "曇", "雨", "雪"]
    
    results = []
    current_queue = 0
    
    for ts in timestamps:
        # 特徴量
        row = {}
        row["month"] = ts.month
        row["dow"] = ts.dayofweek
        row["is_holiday"] = 1 if ts.dayofweek >= 5 or (ts.month==1 and ts.day<=3) else 0
        row["week_of_month"] = (ts.day - 1) // 7 + 1
        row["hour"] = ts.hour
        row["minute"] = ts.minute
        row["slot_id"] = (ts.hour * 60 + ts.minute) // 30
        row["total_outpatient"] = total_pat 
        
        # 気象補完
        temp_base = {1:5, 2:6, 3:10, 4:15, 5:20, 6:24, 7:28, 8:30, 9:26, 10:20, 11:14, 12:8}
        t = temp_base.get(ts.month, 15)
        if weather_text == "雨": row["rain"], row["temp"] = 5.0, t - 2.0
        elif weather_text == "雪": row["rain"], row["temp"] = 2.0, min(t - 5.0, 1.0)
        elif weather_text == "晴": row["rain"], row["temp"] = 0.0, t + 2.0
        else: row["rain"], row["temp"] = 0.0, t
        for w in w_labels: row[f"is_{w}"] = 1 if weather_text == w else 0
            
        # 予測 (Capacity Prediction)
        # Serviceモデルは「95%タイル（最大能力）」を学習しているため、
        # 16:00のような閑散時でも「スタッフがいれば捌ける数」を返す。
        X_df = pd.DataFrame([row])[features]
        pred_arr = max(0, bst_arr.predict(X_df)[0])
        pred_capacity_30m = max(0, bst_svc.predict(X_df)[0]) # 30分あたりの最大処理能力
        
        # --- 物理シミュレーション ---
        
        # 1. 稼働時間の計算
        # 8:00の枠は 8:15〜8:30 の15分間のみ稼働
        current_time = ts.time()
        operating_minutes = 30.0
        
        if current_time == datetime.time(8, 0):
            operating_minutes = 15.0 # 8:15 start
        
        # 2. 実効処理能力の計算 (Proportional Capacity)
        # 30分で pred_capacity_30m 捌けるなら、15分ならその半分
        effective_capacity = pred_capacity_30m * (operating_minutes / 30.0)
        
        # 3. 実際の処理数 (Actual Processed)
        # 需要(行列+新規) と 供給(実効能力) の小さい方
        processed = min(current_queue + pred_arr, effective_capacity)
        
        # 4. 次の行列
        next_queue = current_queue + pred_arr - processed
        
        # 5. 待ち時間 (Wait Time)
        # 分速処理能力 = 30分能力 / 30分 (単位時間あたりのスピードは一定と仮定)
        # ※8:00枠でも、動いている間のスピードは「分速」で評価すべき
        capacity_per_min = pred_capacity_30m / 30.0
        
        if capacity_per_min < 0.1:
            wait_time = 0 if next_queue < 1 else 30 # 能力なし
        else:
            wait_time = next_queue / capacity_per_min
            
        # 8:00枠の特別ルール: 8:15までは絶対待つ
        if current_time == datetime.time(8, 0) and next_queue > 0:
            wait_time += 15.0

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": round(pred_arr),
            "予測呼出数": round(processed),
            "最大処理能力(30分)": round(pred_capacity_30m),
            "予測待ち人数": round(next_queue),
            "予測待ち時間(分)": round(wait_time)
        })
        current_queue = next_queue
        
    return pd.DataFrame(results)

# --- UI ---
def main():
    st.set_page_config(page_title="A病院 混雑予測 AI v11.0", layout="centered")
    st.title("🏥 混雑予測システム v11.0 (Pro)")
    st.caption("Quantile Regression Capacity Model")
    
    bst_arr, bst_svc, meta = load_models()
    if bst_arr is None:
        st.error("モデルファイルが見つかりません。")
        st.stop()
        
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1: date_input = st.date_input("日付", value=datetime.date(2026, 1, 9))
        with col2: weather_text = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
        pat_num = st.number_input("予想来院数 (人)", value=1300, step=50)
        submitted = st.form_submit_button("予測実行")
        
    if submitted:
        df_res = predict_day(date_input, pat_num, weather_text, bst_arr, bst_svc, meta)
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        
        st.success("予測完了")
        c1, c2, c3 = st.columns(3)
        c1.metric("最大待ち時間", f"{peak['予測待ち時間(分)']} 分", f"@{peak['時間帯']}", delta_color="inverse")
        c2.metric("最大行列", f"{peak['予測待ち人数']} 人")
        c3.metric("ピーク受付", f"{peak['予測受付数']} 人")
        
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測待ち人数"]])
        with st.expander("詳細データ"):
            st.dataframe(df_res.style.highlight_max(axis=0, subset=["予測待ち時間(分)"], color="#fffdc9"))
            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSVダウンロード", csv, "predict_result_v11.csv")

if __name__ == "__main__":
    main()
