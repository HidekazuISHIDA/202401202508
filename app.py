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
    
    bst_arr = lgb.Booster(model_file=str(ARR_MODEL_PATH))
    bst_svc = lgb.Booster(model_file=str(SVC_MODEL_PATH))
    
    return bst_arr, bst_svc, meta

# --- シミュレーション ---
def predict_day(date_val, total_pat, weather_text, bst_arr, bst_svc, meta):
    features = meta["features"]
    cfg = meta["config"]
    
    # 時間枠生成
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    w_labels = ["晴", "曇", "雨", "雪"]
    
    results = []
    current_queue = 0 # 朝イチの行列は0
    
    # シミュレーションループ
    for ts in timestamps:
        # 特徴量作成
        row = {}
        row["month"] = ts.month
        row["dow"] = ts.dayofweek
        # 簡易休日判定 (土日 or 1/1-1/3)
        is_hol = 1 if ts.dayofweek >= 5 or (ts.month==1 and ts.day<=3) else 0
        row["is_holiday"] = is_hol
        row["week_of_month"] = (ts.day - 1) // 7 + 1
        row["hour"] = ts.hour
        row["minute"] = ts.minute
        row["slot_id"] = (ts.hour * 60 + ts.minute) // 30
        row["total_outpatient"] = total_pat 
        
        # 気象 (簡易補完ロジック: 月と天気からそれっぽい数値を作る)
        # ※ここがないと「異常値」とみなされて予測が0になる
        temp_base = {1:5, 2:6, 3:10, 4:15, 5:20, 6:24, 7:28, 8:30, 9:26, 10:20, 11:14, 12:8}
        t = temp_base.get(ts.month, 15)
        
        if weather_text == "雨":
            row["rain"] = 5.0
            row["temp"] = t - 2.0
        elif weather_text == "雪":
            row["rain"] = 2.0
            row["temp"] = min(t - 5.0, 1.0)
        elif weather_text == "晴":
            row["rain"] = 0.0
            row["temp"] = t + 2.0
        else: # 曇
            row["rain"] = 0.0
            row["temp"] = t

        for w in w_labels:
            row[f"is_{w}"] = 1 if weather_text == w else 0
            
        # 予測実行
        X_df = pd.DataFrame([row])[features]
        pred_arr = max(0, bst_arr.predict(X_df)[0])
        pred_svc = max(0, bst_svc.predict(X_df)[0])
        
        # --- 物理シミュレーション (待ち時間計算) ---
        # 到着数(Arrivals) - 処理能力(Service) = 行列の増減
        
        # 実際の処理数は「行列+到着」と「処理能力」の小さい方
        potential_throughput = pred_svc
        actual_processed = min(current_queue + pred_arr, potential_throughput)
        
        # 次の時間の行列
        next_queue = current_queue + pred_arr - actual_processed
        
        # 待ち時間推定 (Queue / ServiceSpeed)
        # 処理能力が極端に低い場合はペナルティ
        if potential_throughput < 0.1:
            wait_time = 0 if next_queue < 1 else 30 # 詰まっている
        else:
            # 処理速度 (人/30分) -> 1人あたり (30/svc) 分
            wait_time = next_queue * (30.0 / potential_throughput)
            
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
    st.set_page_config(page_title="A病院 混雑予測 AI v10.0", layout="centered")
    st.title("🏥 混雑予測システム v10.0")
    st.caption("Powered by LightGBM & Physics Simulation")
    
    bst_arr, bst_svc, meta = load_models()
    
    if bst_arr is None:
        st.error("モデルファイルが見つかりません。modelsフォルダを確認してください。")
        st.stop()
        
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_input = st.date_input("日付", value=datetime.date(2026, 1, 9))
        with col2:
            # ここが修正点: weather_text として直接受け取る
            weather_text = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
            
        pat_num = st.number_input("予想来院数 (人)", value=1300, step=50, help="平均: 1000-1500")
        
        submitted = st.form_submit_button("予測実行")
        
    if submitted:
        st.info(f"{date_input.strftime('%Y/%m/%d')} (天気: {weather_text}, 来院予定: {pat_num}人) の予測を行います...")
        
        # 修正済みの関数を呼び出し
        df_res = predict_day(date_input, pat_num, weather_text, bst_arr, bst_svc, meta)
        
        # ピーク検出
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        
        st.success("予測完了！")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("最大待ち時間", f"{peak['予測待ち時間(分)']} 分", f"@{peak['時間帯']}")
        m2.metric("最大行列", f"{peak['予測待ち人数']} 人")
        m3.metric("ピーク受付", f"{peak['予測受付数']} 人/30分")
        
        st.subheader("一日の推移")
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測待ち人数"]])
        
        with st.expander("詳細データを見る"):
            st.dataframe(df_res)
            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSVダウンロード", csv, "predict_result.csv")

if __name__ == "__main__":
    main()
