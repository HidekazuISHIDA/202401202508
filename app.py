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

# --- シミュレーション (v10.1: 運用現実対応版) ---
def predict_day(date_val, total_pat, weather_text, bst_arr, bst_svc, meta):
    features = meta["features"]
    cfg = meta["config"]
    
    # 時間枠生成
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    w_labels = ["晴", "曇", "雨", "雪"]
    
    results = []
    current_queue = 0 # 朝イチの行列
    
    # ★運用設定: 呼出開始時刻
    SERVICE_START_TIME = datetime.time(8, 15)
    
    for ts in timestamps:
        # 特徴量作成
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
        if weather_text == "雨":
            row["rain"], row["temp"] = 5.0, t - 2.0
        elif weather_text == "雪":
            row["rain"], row["temp"] = 2.0, min(t - 5.0, 1.0)
        elif weather_text == "晴":
            row["rain"], row["temp"] = 0.0, t + 2.0
        else:
            row["rain"], row["temp"] = 0.0, t

        for w in w_labels:
            row[f"is_{w}"] = 1 if weather_text == w else 0
            
        # AI予測 (来院数と潜在能力)
        X_df = pd.DataFrame([row])[features]
        pred_arr = max(0, bst_arr.predict(X_df)[0])
        pred_svc_capacity = max(0, bst_svc.predict(X_df)[0])
        
        # --- 物理シミュレーション (Reality Logic) ---
        
        # 1. 運用ルールの適用 (8:15までは処理ゼロ)
        # 現在の時間枠の終了時刻を確認
        # 例: 08:00枠 -> 08:00〜08:30。このうち08:15までは処理しない。
        # つまり、08:00枠の処理能力は実質半分になる。
        
        current_time = ts.time()
        actual_svc_power = pred_svc_capacity # 基本能力
        
        if current_time < datetime.time(8, 0): 
            actual_svc_power = 0 # ありえないが念のため
        elif current_time == datetime.time(8, 0):
            # 8:00〜8:30の枠。
            # 8:00〜8:15は処理なし。8:15〜8:30のみ稼働。
            # よって処理能力は 50% とみなす。
            actual_svc_power = pred_svc_capacity * 0.5
        
        # 2. 行列計算
        # 処理できた人数 = min(今の行列 + 新規客, 実際の処理能力)
        processed = min(current_queue + pred_arr, actual_svc_power)
        
        # 次に持ち越す行列
        next_queue = current_queue + pred_arr - processed
        
        # 3. 待ち時間計算 (Little's Law Custom)
        # 処理速度 (人/分)
        # 08:00枠の場合、稼働は15分間だけなので、分速は processed / 15
        if current_time == datetime.time(8, 0):
            svc_per_min = actual_svc_power / 15.0
        else:
            svc_per_min = actual_svc_power / 30.0
            
        if svc_per_min < 0.1:
            # 処理が止まっている場合、行列がいれば待ち時間は増え続ける
            wait_time = 0 if next_queue < 1 else 30 + (next_queue * 2) 
        else:
            wait_time = next_queue / svc_per_min
            
        # 8:00の枠に来た人は、少なくとも8:15までは待つので、最低15分のオフセット
        if current_time == datetime.time(8, 0) and next_queue > 0:
            wait_time += 15.0

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": round(pred_arr),
            "予測呼出数": round(processed), # 実際の処理数
            "潜在処理能力": round(pred_svc_capacity),
            "予測待ち人数": round(next_queue),
            "予測待ち時間(分)": round(wait_time)
        })
        current_queue = next_queue
        
    return pd.DataFrame(results)

# --- UI ---
def main():
    st.set_page_config(page_title="A病院 混雑予測 AI v10.1", layout="centered")
    st.title("🏥 混雑予測システム v10.1")
    st.caption("Reality Simulation: 8:15 Start Logic Implemented")
    
    bst_arr, bst_svc, meta = load_models()
    
    if bst_arr is None:
        st.error("モデルファイル不足: modelsフォルダを確認してください")
        st.stop()
        
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_input = st.date_input("日付", value=datetime.date(2026, 1, 9))
        with col2:
            weather_text = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
            
        pat_num = st.number_input("予想来院数 (人)", value=1300, step=50, help="平均: 1000-1500")
        submitted = st.form_submit_button("予測実行")
        
    if submitted:
        st.info(f"シミュレーション実行中... (8:15 呼出開始ロジック適用)")
        
        df_res = predict_day(date_input, pat_num, weather_text, bst_arr, bst_svc, meta)
        
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        
        st.success("予測完了")
        
        # メトリクス
        c1, c2, c3 = st.columns(3)
        c1.metric("最大待ち時間", f"{peak['予測待ち時間(分)']} 分", f"@{peak['時間帯']}", delta_color="inverse")
        c2.metric("最大行列", f"{peak['予測待ち人数']} 人")
        c3.metric("ピーク時受付", f"{peak['予測受付数']} 人")
        
        # グラフ
        st.subheader("混雑推移")
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測待ち人数"]])
        
        # 詳細データ
        with st.expander("詳細データ"):
            st.dataframe(df_res.style.highlight_max(axis=0, subset=["予測待ち時間(分)", "予測待ち人数"], color="#fffdc9"))
            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSVダウンロード", csv, "predict_result_v10_1.csv")

if __name__ == "__main__":
    main()
