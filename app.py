import json
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# パス設定
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

# ファイルパス定義
ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.json"
SVC_MODEL_PATH   = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"

ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH    = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"

# 祝日CSV (オプション)
HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

# 設定
OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
INCLUDE_CLOSE = False  # 18:00枠除外

# ----------------------------
# ヘルパー関数群
# ----------------------------
def _load_holidays() -> set:
    """祝日CSVを読み込む（Shift-JIS/UTF-8両対応）"""
    if not HOLIDAY_CSV_PATH.exists():
        return set()
    
    df = None
    # エンコーディング対応
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            df = pd.read_csv(HOLIDAY_CSV_PATH, encoding=enc, engine="python")
            break
        except Exception:
            continue
            
    if df is None:
        return set()

    col = None
    # 日付列を探す
    for c in df.columns:
        if str(c).strip().lower() in ["date", "日付", "国民の祝日・休日月日"]:
            col = c
            break
    if col is None:
        col = df.columns[0]
    
    s = pd.to_datetime(df[col], errors="coerce").dropna().dt.date
    return set(s.tolist())

HOLIDAYS = _load_holidays()

def is_holiday(d: date) -> bool:
    if d.weekday() >= 5: return True
    if d in HOLIDAYS: return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3): return True
    return False

def week_of_month(d: date) -> int:
    return int((d.day - 1)//7 + 1)

def normalize_weather(s: str) -> str:
    t = str(s) if s is not None else ""
    if "雪" in t: return "雪"
    if "雨" in t: return "雨"
    if "曇" in t: return "曇"
    if "晴" in t: return "晴"
    return "曇"

def month_weekday_counts(y: int, m: int):
    start = pd.Timestamp(year=y, month=m, day=1)
    end = (start + pd.offsets.MonthEnd(1))
    days = pd.date_range(start, end, freq="D")
    dow = days.dayofweek
    counts = {k:int((dow==k).sum()) for k in range(7)}
    weekday_total = sum(counts[k] for k in range(5))
    return counts, weekday_total

@st.cache_resource
def load_artifacts():
    # Columns
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    # Models
    arr_bst = xgb.Booster()
    arr_bst.load_model(str(ARR_MODEL_PATH))

    svc_bst = xgb.Booster()
    svc_bst.load_model(str(SVC_MODEL_PATH))

    wait_bst = xgb.Booster()
    wait_bst.load_model(str(WAIT_MODEL_PATH))

    # Baseline & Calib
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))

    return arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)

def _predict_booster(bst: xgb.Booster, cols, df: pd.DataFrame) -> float:
    X = df[cols].copy()
    X = _coerce_numeric(X)
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    if getattr(bst, "best_iteration", None) is not None:
        pred = bst.predict(dmat, iteration_range=(0, bst.best_iteration + 1))
    else:
        pred = bst.predict(dmat)
    return float(pred[0])

def baseline_lookup(baseline: dict, table_name: str, month: int, dow: int, slot: int) -> float:
    # key format: "m_d_slot" (String)
    table = baseline.get(table_name, {})
    key = f"{int(month)}_{int(dow)}_{int(slot)}"
    v = table.get(key, 0.0)
    return float(v)

def slot_index(ts: datetime) -> int:
    return int((ts.hour - OPEN_HOUR) * 2 + (ts.minute // 30))

def generate_time_slots(target_date: date):
    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    rng = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")
    if INCLUDE_CLOSE:
        return list(rng)
    close_t = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    return [t.to_pydatetime() for t in rng if t.to_pydatetime() != close_t]

# ----------------------------
# シミュレーション本体
# ----------------------------
def simulate_one_day(
    target_date: date,
    total_outpatient_count: int,
    weather_text: str
) -> pd.DataFrame:
    arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib = load_artifacts()

    y = target_date.year
    m = target_date.month
    d = target_date.day
    dow = target_date.weekday()
    is_h = int(is_holiday(target_date))
    prev_h = int(is_holiday(target_date - timedelta(days=1)))

    counts, weekday_total = month_weekday_counts(y, m)
    weekday_count_in_month = int(counts.get(dow, 0))
    weekday_ratio_in_month = float(weekday_count_in_month / weekday_total) if weekday_total > 0 else 0.0

    wcat = normalize_weather(weather_text)

    # 状態変数
    lags_arr = {"arr_lag_30": 0.0, "arr_lag_60": 0.0, "arr_lag_90": 0.0}
    lags_svc = {"svc_lag_30": 0.0, "svc_lag_60": 0.0, "svc_lag_90": 0.0}
    cum_arrivals = 0
    cum_service = 0
    queue_at_start = 0.0

    # キャリブレーション定数
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    alpha = float(calib.get("alpha", 0.55))
    floor_ratio = float(calib.get("floor_ratio", 0.90))

    results = []
    for ts in generate_time_slots(target_date):
        slot = slot_index(ts)

        # Baseline
        arr_base  = baseline_lookup(baseline, "arr_base",  m, dow, slot)
        svc_base  = baseline_lookup(baseline, "svc_base",  m, dow, slot)
        wait_base = baseline_lookup(baseline, "wait_base", m, dow, slot)

        # --- 1) Arrivals Model ---
        cf = _make_zero_df(arr_cols)
        def set_if(df_target, col, val):
            if col in df_target.columns:
                df_target.loc[0, col] = val

        # 共通特徴量セット
        def set_common(df_target):
            set_if(df_target, "year", y)
            set_if(df_target, "month", m)
            set_if(df_target, "dayofweek", dow)
            set_if(df_target, "is_holiday", is_h)
            set_if(df_target, "前日祝日フラグ", prev_h)
            set_if(df_target, "月", m)
            set_if(df_target, "週回数", week_of_month(target_date))
            set_if(df_target, "month_weekday_total", weekday_count_in_month)
            set_if(df_target, "weekday_count_in_month", weekday_count_in_month)
            set_if(df_target, "weekday_ratio_in_month", weekday_ratio_in_month)
            set_if(df_target, "total_outpatient_count", int(total_outpatient_count))
            
            set_if(df_target, "雨フラグ", 1 if "雨" in wcat else 0)
            set_if(df_target, "雪フラグ", 1 if "雪" in wcat else 0)
            set_if(df_target, f"天気カテゴリ_{wcat}", 1)
            
            set_if(df_target, "hour", ts.hour)
            set_if(df_target, "minute", ts.minute)
            set_if(df_target, f"dayofweek_{dow}", 1)
            set_if(df_target, "is_first_slot", 1 if (ts.hour==8 and ts.minute==0) else 0)
            set_if(df_target, "is_second_slot", 1 if (ts.hour==8 and ts.minute==30) else 0)
            
            set_if(df_target, "arr_base", float(arr_base))
            set_if(df_target, "svc_base", float(svc_base))
            set_if(df_target, "wait_base", float(wait_base))

            set_if(df_target, "queue_at_start_truth", float(queue_at_start))
            set_if(df_target, "arr_lag_30", float(lags_arr["arr_lag_30"]))
            set_if(df_target, "arr_lag_60", float(lags_arr["arr_lag_60"]))
            set_if(df_target, "arr_lag_90", float(lags_arr["arr_lag_90"]))
            set_if(df_target, "arr_roll_60", float((lags_arr["arr_lag_30"] + lags_arr["arr_lag_60"]) / 2.0))
            set_if(df_target, "svc_lag_30", float(lags_svc["svc_lag_30"]))
            set_if(df_target, "svc_lag_60", float(lags_svc["svc_lag_60"]))
            set_if(df_target, "svc_lag_90", float(lags_svc["svc_lag_90"]))
            set_if(df_target, "svc_roll_60", float((lags_svc["svc_lag_30"] + lags_svc["svc_lag_60"]) / 2.0))
            
            set_if(df_target, "cum_arrivals", int(cum_arrivals))
            set_if(df_target, "cum_service", int(cum_service))

        set_common(cf)
        pred_arr = _predict_booster(arr_bst, arr_cols, cf)
        arr_i = max(0, int(round(pred_arr)))

        # --- 2) Service & Wait Models ---
        mf = _make_zero_df(multi_cols)
        set_common(mf) # 同じ特徴量セットを適用（arr_iはまだ入らない。前スロットまでの情報で推論）

        # Service Predict
        pred_svc = _predict_booster(svc_bst, multi_cols, mf)
        svc_i = max(0, int(round(pred_svc)))

        # Queue Update (Conservation)
        q_next = max(0.0, float(queue_at_start) + float(arr_i) - float(svc_i))

        # Wait Model Predict (ML)
        pred_wait_model = _predict_booster(wait_bst, multi_cols, mf)
        pred_wait_model = max(0.0, float(pred_wait_model))

        # Physics Wait (Queue / Service)
        # ゼロ除算対策: 処理数が極端に少ない場合の安全策 (最低でも30分で0.5人は進むと仮定)
        safe_svc = max(float(svc_i), 0.5)
        wait_phy = (float(queue_at_start) / safe_svc) * 30.0
        
        # 物理モデルの暴走防止（上限キャップ）
        wait_phy = min(wait_phy, 300.0) 
        wait_phy = max(0.0, a * wait_phy + b)

        # Ensemble
        wait_blend = alpha * pred_wait_model + (1.0 - alpha) * wait_phy
        
        # Baseline Floor (極端な下振れ防止)
        wait_floor = float(wait_base) * float(floor_ratio)
        wait_final = max(wait_floor, wait_blend)

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測呼出数(処理数)": int(svc_i),
            "予測待ち人数(人)": int(round(q_next)),
            "予測平均待ち時間(分)": int(round(wait_final)),
        })

        # Update State
        lags_arr = {"arr_lag_30": float(arr_i), "arr_lag_60": float(lags_arr["arr_lag_30"]), "arr_lag_90": float(lags_arr["arr_lag_60"])}
        lags_svc = {"svc_lag_30": float(svc_i), "svc_lag_60": float(lags_svc["svc_lag_30"]), "svc_lag_90": float(lags_svc["svc_lag_60"])}

        cum_arrivals += int(arr_i)
        cum_service  += int(svc_i)
        queue_at_start = q_next

    return pd.DataFrame(results)

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="A病院 予測シミュレータ", layout="wide")
    st.title("🏥 A病院 採血 待ち時間予測AI")
    st.caption("Weekday Count, Physics Ensemble, Baseline Floor 搭載版")

    # ファイル存在チェック
    required = [
        ARR_MODEL_PATH, SVC_MODEL_PATH, WAIT_MODEL_PATH,
        ARR_COLS_PATH, MULTI_COLS_PATH, BASELINE_PATH, CALIB_PATH
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error(f"以下のモデルファイルが見つかりません。models/フォルダを確認してください:\n\n" + "\n".join(missing))
        st.stop()

    with st.sidebar:
        st.header("条件設定")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        
        # デフォルト値を過去の中央値あたりに設定
        total_out = st.number_input("予測来院患者数(延べ)", min_value=0, value=1200, step=10, help="病院全体の予測来院数")
        
        weather = st.selectbox("天気予報", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=1)
        
        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.markdown("**モデル情報**")
        st.caption(f"Wait Model α: {load_artifacts()[6].get('alpha', 'N/A')}")
        
    if run:
        with st.spinner("AIが推論中..."):
            df = simulate_one_day(target, int(total_out), str(weather))
        
        st.success(f"📅 {target.strftime('%Y-%m-%d')} の予測完了")

        # メトリクス表示
        avg_wait = df["予測平均待ち時間(分)"].mean()
        max_wait = df["予測平均待ち時間(分)"].max()
        peak_idx = df["予測平均待ち時間(分)"].idxmax()
        peak_time = df.loc[peak_idx, "時間帯"]

        m1, m2, m3 = st.columns(3)
        m1.metric("平均待ち時間", f"{avg_wait:.1f} 分")
        m2.metric("最大待ち時間", f"{max_wait} 分", f"@{peak_time}")
        m3.metric("総受付数", f"{df['予測受付数'].sum()} 人")

        # グラフ
        st.subheader("予測チャート")
        chart_data = df.set_index("時間帯")[["予測平均待ち時間(分)", "予測待ち人数(人)"]]
        st.line_chart(chart_data)

        # テーブル
        with st.expander("詳細データを見る", expanded=True):
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("CSVダウンロード", data=csv, file_name=f"predict_{target}.csv", mime="text/csv")

if __name__ == "__main__":
    main()
