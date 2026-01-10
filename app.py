import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import datetime
import os
from pathlib import Path

# --- 設定 ---
MODELS_DIR = Path("models") # 解凍したフォルダ
META_PATH = MODELS_DIR / "model_meta.json"

# --- モデル読み込み ---
@st.cache_resource
def load_resources():
    if not META_PATH.exists(): return None, None
    with open(META_PATH, "r") as f: meta = json.load(f)
    
    # 到着予測用グローバルモデル
    bst_arr = lgb.Booster(model_file=str(MODELS_DIR / "lgb_arrival_global.txt"))
    
    # 時間帯別待ち時間モデル (辞書に格納)
    wait_models = {}
    for slot in meta["slots"]:
        safe_slot = slot.replace(":", "")
        model_path = MODELS_DIR / f"wait_{safe_slot}.txt"
        if model_path.exists():
            wait_models[slot] = lgb.Booster(model_file=str(model_path))
            
    return bst_arr, wait_models, meta

# --- シミュレーション ---
def predict_scenario(date_val, total_pat, weather_text, bst_arr, wait_models, meta):
    feat_arr = meta["features_arr"]
    feat_wait = meta["features_wait"]
    cfg = meta["config"]
    
    # 時間枠生成
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    w_labels = ["晴", "曇", "雨", "雪"]
    
    # 1. まず、一日分の特徴量テーブルを作成
    df_sim = pd.DataFrame({"ts": timestamps})
    df_sim["time_str"] = df_sim["ts"].dt.strftime("%H:%M")
    df_sim["month"] = date_val.month
    df_sim["dow"] = date_val.weekday()
    df_sim["is_holiday"] = 1 if date_val.weekday() >= 5 else 0 # 簡易
    df_sim["week_of_month"] = (date_val.day - 1) // 7 + 1
    df_sim["hour"] = df_sim["ts"].dt.hour
    df_sim["minute"] = df_sim["ts"].dt.minute
    df_sim["total_outpatient"] = total_pat
    
    # 気象補完
    t_base = {1:5, 8:30} # 簡易辞書
    temp = t_base.get(date_val.month, 15)
    rain = 0.0
    if weather_text == "雨": rain, temp = 5.0, temp-2
    elif weather_text == "雪": rain, temp = 2.0, temp-5
    elif weather_text == "晴": temp += 2
    
    df_sim["rain"] = rain
    df_sim["temp"] = temp
    for w in w_labels: df_sim[f"is_{w}"] = 1 if weather_text == w else 0
    
    # 2. 到着数 (Arrivals) を一括予測
    X_arr = df_sim[feat_arr]
    df_sim["pred_arrivals"] = bst_arr.predict(X_arr)
    df_sim["pred_arrivals"] = df_sim["pred_arrivals"].apply(lambda x: max(0, round(x)))
    
    # 3. 累積到着数 (Cumulative Arrivals) を計算
    # これが「その時間がどれくらいパンクしているか」の指標になる
    df_sim["daily_cum_arrivals"] = df_sim["pred_arrivals"].cumsum()
    
    # 4. 時間帯別モデルで待ち時間を予測
    results = []
    
    for _, row in df_sim.iterrows():
        slot = row["time_str"]
        model = wait_models.get(slot)
        
        wait_time = 0
        if model:
            # モデル入力用に特徴量を整形
            # feature names: feat_arr + ["actual_arrivals", "daily_cum_arrivals"]
            # ここでは pred を actual として入力する
            input_row = row[feat_arr].to_dict()
            input_row["actual_arrivals"] = row["pred_arrivals"]
            input_row["daily_cum_arrivals"] = row["daily_cum_arrivals"]
            
            # DataFrame変換 (順序保証のため)
            X_wait = pd.DataFrame([input_row])[feat_wait]
            
            # 予測
            wait_time = model.predict(X_wait)[0]
            wait_time = max(0, round(wait_time)) # 負の値は0に
            
        results.append({
            "時間帯": slot,
            "予測受付数": int(row["pred_arrivals"]),
            "累積受付数": int(row["daily_cum_arrivals"]),
            "予測待ち時間(分)": int(wait_time)
        })
        
    return pd.DataFrame(results)

# --- UI ---
def main():
    st.set_page_config(page_title="A病院 混雑予測 AI v12.0", layout="centered")
    st.title("🏥 混雑予測システム v12.0")
    st.caption("Time-Slot Specific Modeling (No Manual Adjustments)")
    
    bst_arr, wait_models, meta = load_resources()
    
    if not bst_arr:
        st.error("モデルが見つかりません。modelsフォルダを確認してください。")
        st.stop()
        
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1: date_input = st.date_input("日付", value=datetime.date(2026, 1, 9))
        with col2: weather_text = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=1)
        pat_num = st.number_input("予想来院数 (人)", value=1300, step=50)
        submitted = st.form_submit_button("予測実行")
        
    if submitted:
        df_res = predict_scenario(date_input, pat_num, weather_text, bst_arr, wait_models, meta)
        
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        
        st.success(f"予測完了: ピークは {peak['時間帯']} ({peak['予測待ち時間(分)']}分待ち)")
        
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測受付数"]])
        
        # 8:00と8:30の比較を強調表示
        st.write("### 🕣 午前中の詳細")
        st.dataframe(df_res.head(5).style.highlight_max(axis=0, subset=["予測待ち時間(分)"], color="#ffcccc"))
        
        with st.expander("全データ"):
            st.dataframe(df_res)
            st.download_button("CSV保存", df_res.to_csv(index=False).encode("utf-8-sig"), "result_v12.csv")

if __name__ == "__main__":
    main()
