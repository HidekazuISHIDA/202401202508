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
STATS_PATH = MODELS_DIR / "dow_stats.json"

# --- モデル読み込み ---
@st.cache_resource
def load_resources():
    if not META_PATH.exists(): return None, None, None
    
    with open(META_PATH, "r") as f: meta = json.load(f)
    
    # 統計情報の読み込み (v16新機能)
    dow_stats = {}
    if STATS_PATH.exists():
        with open(STATS_PATH, "r") as f: dow_stats = json.load(f)
    
    bst_arr = lgb.Booster(model_file=str(MODELS_DIR / "lgb_arrival_global.txt"))
    
    wait_models = {}
    for slot in meta["slots"]:
        safe_slot = slot.replace(":", "")
        model_path = MODELS_DIR / f"wait_{safe_slot}.txt"
        if model_path.exists():
            wait_models[slot] = lgb.Booster(model_file=str(model_path))
            
    return bst_arr, wait_models, meta, dow_stats

# --- シミュレーション ---
def predict_scenario(date_val, total_pat, weather_text, bst_arr, wait_models, meta, dow_stats):
    feat_arr = meta["features_arr"]
    feat_wait = meta["features_wait"]
    cfg = meta["config"]
    
    start_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["OPEN_TIME"], "%H:%M").time())
    end_dt = datetime.datetime.combine(date_val, datetime.datetime.strptime(cfg["CLOSE_TIME"], "%H:%M").time())
    timestamps = pd.date_range(start_dt, end_dt, freq=cfg["FREQ"])
    
    df_sim = pd.DataFrame({"ts": timestamps})
    df_sim["time_str"] = df_sim["ts"].dt.strftime("%H:%M")
    df_sim["month"] = date_val.month
    df_sim["dow"] = date_val.weekday()
    
    # 休日判定 (土日 or 1/1-1/3)
    df_sim["is_holiday"] = 1 if date_val.weekday() >= 5 or (date_val.month==1 and date_val.day<=3) else 0
    # 祝日明け判定 (簡易: 月曜=1) ※厳密にはカレンダーAPIが必要だがここでは簡易実装
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
    
    # ★統計情報のマージ
    # 予測対象日の曜日(dow)と時間(time_str)に対応する統計値をセット
    def get_stat(row, key_suffix):
        k = f"{int(row['dow'])}_{row['time_str']}"
        val = dow_stats.get(k, {})
        return val.get(key_suffix, 0)

    df_sim["stat_arr_mean"] = df_sim.apply(lambda r: get_stat(r, "stat_arr_mean"), axis=1)
    df_sim["stat_wait_p90"] = df_sim.apply(lambda r: get_stat(r, "stat_wait_p90"), axis=1)
    
    # 1. 到着予測
    X_arr = df_sim[feat_arr]
    df_sim["pred_arrivals"] = bst_arr.predict(X_arr)
    df_sim["pred_arrivals"] = df_sim["pred_arrivals"].apply(lambda x: max(0, round(x)))
    
    # 累積計算
    df_sim["daily_cum_arrivals"] = df_sim["pred_arrivals"].cumsum()
    
    # 2. 待ち時間予測
    results = []
    for _, row in df_sim.iterrows():
        slot = row["time_str"]
        model = wait_models.get(slot)
        
        wait_time = 0
        if model:
            input_row = row[feat_arr].to_dict()
            input_row["actual_arrivals"] = row["pred_arrivals"]
            input_row["daily_cum_arrivals"] = row["daily_cum_arrivals"]
            # 統計情報(stat_wait_p90)も入力に含まれていることに注意(feat_wait内)
            input_row["stat_wait_p90"] = row["stat_wait_p90"]
            
            X_wait = pd.DataFrame([input_row])[feat_wait]
            
            # 90%ile予測
            wait_time = model.predict(X_wait)[0]
            wait_time = max(0, wait_time)
            
        results.append({
            "時間帯": slot,
            "予測受付数": int(row["pred_arrivals"]),
            "累積受付数": int(row["daily_cum_arrivals"]),
            "予測待ち時間(分)": int(round(wait_time))
        })
        
    return pd.DataFrame(results)

# --- UI ---
def main():
    st.set_page_config(page_title="A病院 混雑予測 AI v16.0", layout="wide")
    st.title("🏥 混雑予測システム v16.0")
    st.caption("Day-Aware Model (曜日・祝日特性対応)")
    
    bst_arr, wait_models, meta, dow_stats = load_resources()
    
    if not bst_arr:
        st.error("モデルファイルが見つかりません。")
        st.stop()
    
    # デフォルト設定 (UI改善)
    # 現在時刻を取得 (JST考慮: +9時間)
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    if now.hour >= 12:
        default_date = now.date() + datetime.timedelta(days=1) # 午後は明日
    else:
        default_date = now.date() # 午前は今日
        
    with st.sidebar:
        st.header("条件設定")
        date_input = st.date_input("日付", value=default_date)
        
        # デフォルト天気: 晴れ(index=0)
        weather_text = st.selectbox("天気", ["晴", "曇", "雨", "雪"], index=0)
        pat_num = st.number_input("外来患者数 (全体)", value=1638, step=50)
        
        run = st.button("予測実行", type="primary")

    if run:
        df_res = predict_scenario(date_input, pat_num, weather_text, bst_arr, wait_models, meta, dow_stats)
        
        peak = df_res.loc[df_res["予測待ち時間(分)"].idxmax()]
        total_blood = df_res["予測受付数"].sum()
        
        # 曜日情報の表示
        dow_str = ["月", "火", "水", "木", "金", "土", "日"][date_input.weekday()]
        st.success(f"予測完了: {date_input} ({dow_str})")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最大待ち時間 (厳しめ)", f"{peak['予測待ち時間(分)']} 分", f"@{peak['時間帯']}", delta_color="inverse")
        c2.metric("ピーク時受付", f"{peak['予測受付数']} 人")
        c3.metric("採血室 総来室数", f"{total_blood} 人")
        
        st.line_chart(df_res.set_index("時間帯")[["予測待ち時間(分)", "予測受付数"]])
        
        st.write("### 🕣 詳細データ")
        st.dataframe(df_res.style.highlight_max(axis=0, subset=["予測待ち時間(分)"], color="#ffcccc"), use_container_width=True)
        
        with st.expander("CSVダウンロード"):
            csv = df_res.to_csv(index=False).encode("utf-8-sig")
            st.download_button("結果を保存", csv, f"result_{date_input}.csv")

if __name__ == "__main__":
    main()
