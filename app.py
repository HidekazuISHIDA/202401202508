import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import datetime
import os
from pathlib import Path

# --- 設定 ---
MODELS_DIR = Path("models") 
META_PATH = MODELS_DIR / "model_meta.json"

# --- モデル読み込み ---
@st.cache_resource
def load_resources():
    if not META_PATH.exists(): return None, None, None
    with open(META_PATH, "r") as f: meta = json.load(f)
    
    bst_arr = lgb.Booster(model_file=str(MODELS_DIR / "lgb_arrival_global.txt"))
    
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
    
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    # DataFrame構築
    df_sim = pd.DataFrame({"ts": timestamps})
    df_sim["time_str"] = df_sim["ts"].dt.strftime("%H:%M")
    df_sim["month"] = date_val.month
    df_sim["dow"] = date_val.weekday()
    # 簡易休日判定（ライブラリなし）
    is_hol = 1 if date_val.weekday() >= 5 or (date_val.month==1 and date_val.day<=3) else 0
    df_sim["is_holiday"] = is_hol
    
    # ★祝日明け判定（簡易）: 月曜なら1とする（連休明けの代表として）
    # 本格的にやるならカレンダーロジックが必要だが、ここでは「月曜＝混む」を表現
    df_sim["is_after_holiday"] = 1 if date_val.weekday() == 0 else 0
    
    df_sim["week_of_month"] = (date_val.day - 1) // 7 + 1
    df_sim["hour"] = df_sim["ts"].dt.hour
    df_sim["minute"] = df_sim["ts"].dt.minute
    df_sim["total_outpatient"] = total_pat
    
    # 気象補完
    w_labels = ["晴", "曇", "雨", "雪"]
    t_base = {1:5, 2:6, 3:10, 4:15, 5:20, 6:24, 7:28, 8:30, 9:26, 10:20, 11:14, 12:8}
    temp = t_base.get(date_val.month, 15)
    rain = 0.0
    if weather_text == "雨": rain, temp = 5.0, temp-2
    elif weather_text == "雪": rain, temp = 2.0, temp-5
    elif weather_text == "晴": temp += 2
    
    df_sim["rain"] = rain
    df_sim["temp"] = temp
    for w in w_labels: df_sim[f"is_{w}"] = 1 if weather_text == w else 0
    
    # 到着予測
    X_arr = df_sim[feat_arr]
    df_sim["pred_arrivals"] = bst_arr.predict(X_arr)
    df_sim["pred_arrivals"] = df_sim["pred_arrivals"].apply(lambda x: max(0, round(x)))
    df_sim["daily_cum_arrivals"] = df_sim["pred_arrivals"].cumsum()
    
    # 待ち時間予測
    results = []
    for _, row in df_sim.iterrows():
        slot = row["time_str"]
        model = wait_models.get(slot)
        
        wait_time = 0
        if model:
            input_row = row[feat_arr].to_dict()
            input_row["actual_arrivals"] = row["pred_arrivals"]
            input_row["daily_cum_arrivals"] = row["daily_cum_arrivals"]
            
            X_wait = pd.DataFrame([input_row])[feat_wait]
            
            # ★対数からの復元: expm1
            pred_log = model.predict(X_wait)[0]
            wait_time = np.expm1(pred_log)
            
            # 安全装置: 負の値は0、上限は180
            wait_time = max(0, min(wait_time, 180))
            
        results.append({
            "時間帯": slot,
            "予測受付数": int(row["pred_arrivals"]),
            "累積受付数": int(row["daily_cum_arrivals"]),
            "予測待ち時間(分)": int(round(wait_time))
        })
        
    return pd.DataFrame(results)

# --- UI ---
def main():
    st.set_page_config(page_title="A病院 混雑予測 AI v13.0", layout="centered")
    st.title("🏥 混雑予測システム v13.0")
    st.caption("Robust Log-Transformed Model with Holiday Logic")
    
    bst_arr, wait_models, meta = load_resources()
    
    if not bst_arr:
        st.error("モデルファイルが見つかりません。")
        st.stop()
    
    # デフォルト値の計算
    now = datetime.datetime.now()
    # 午後(12:00以降)なら明日、午前なら今日
    if now.hour >= 12:
        default_date = now.date() + datetime.timedelta(days=1)
    else:
        default_date = now.date()
        
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_input = st.date_input("日付", value=default_date)
        with col2:
            # デフォルトを「晴」(index=0)に変更
            weather_text = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=0)
            
        pat_num = st.number_input("予想来院数 (人)", value=1300, step=50)
        submitted = st.form_submit_button("予測実行")
        
    if submitted:
        df_res = predict_scenario(date_input, pat_num, weather_text, bst_arr, wait_models, meta)
        
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        
        st.success(f"予測完了: ピークは {peak['時間帯']} ({peak['予測待ち時間(分)']}分待ち)")
        
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測受付数"]])
        
        st.write("### 🕣 午前中の詳細")
        st.dataframe(df_res.head(6).style.highlight_max(axis=0, subset=["予測待ち時間(分)"], color="#ffcccc"))
        
        with st.expander("全データ"):
            st.dataframe(df_res)
            st.download_button("CSV保存", df_res.to_csv(index=False).encode("utf-8-sig"), "result_v13.csv")

if __name__ == "__main__":
    main()
