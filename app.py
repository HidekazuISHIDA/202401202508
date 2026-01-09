import json
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# ----------------------------
# 設定・定数
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

# モデルファイルパス
ARR_MODEL_PATH   = MODELS_DIR / "model_A_timeseries.json"
SVC_MODEL_PATH   = MODELS_DIR / "model_A_service_30min.json"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"

ARR_COLS_PATH    = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

BASELINE_PATH    = MODELS_DIR / "baseline_tables_mds.json"
CALIB_PATH       = MODELS_DIR / "wait_calibration.json"

# 祝日データ (オプション)
HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30
INCLUDE_CLOSE = False

# ----------------------------
# 関数定義
# ----------------------------
def _load_holidays() -> set:
    """祝日CSVを安全に読み込む"""
    if not HOLIDAY_CSV_PATH.exists():
        return set()
    
    df = None
    # 複数の文字コードで試行
    for enc in ["cp932", "shift_jis", "utf-8", "utf-8-sig"]:
        try:
            df = pd.read_csv(HOLIDAY_CSV_PATH, encoding=enc, engine="python")
            break
        except Exception:
            continue
            
    if df is None:
        return set()

    col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "日付", "国民の祝日・休日月日", "年月日"]:
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
    # 年末年始 (12/29-1/3)
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
    # 特徴量リスト
    arr_cols = json.loads(ARR_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    # モデル (XGBoost Booster)
    arr_bst = xgb.Booster()
    arr_bst.load_model(str(ARR_MODEL_PATH))

    svc_bst = xgb.Booster()
    svc_bst.load_model(str(SVC_MODEL_PATH))

    wait_bst = xgb.Booster()
    wait_bst.load_model(str(WAIT_MODEL_PATH))

    # ベースライン & 補正係数
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
    # Key format: "m_d_slot"
    table = baseline.get(table_name, {})
    key = f"{int(month)}_{int(dow)}_{int(slot)}"
    return float(table.get(key, 0.0))

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
# シミュレーションロジック
# ----------------------------
def simulate_one_day(
    target_date: date,
    total_outpatient_count: int,
    weather_text: str
) -> pd.DataFrame:
    arr_bst, arr_cols, svc_bst, wait_bst, multi_cols, baseline, calib = load_artifacts()

    y = target_date.year
    m = target_date.month
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

    # キャリブレーション設定
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    alpha = float(calib.get("alpha", 0.5))
    floor_ratio = float(calib.get("floor_ratio", 0.9))

    results = []
    
    # タイムスロットごとのループ
    for ts in generate_time_slots(target_date):
        slot = slot_index(ts)

        # Baseline
        arr_base  = baseline_lookup(baseline, "arr_base",  m, dow, slot)
        svc_base  = baseline_lookup(baseline, "svc_base",  m, dow, slot)
        wait_base = baseline_lookup(baseline, "wait_base", m, dow, slot)

        # 特徴量作成用ヘルパー
        def set_common_features(df_target):
            # カレンダー・基本情報
            df_target.loc[0, "year"] = y
            df_target.loc[0, "month"] = m
            df_target.loc[0, "dayofweek"] = dow
            df_target.loc[0, "is_holiday"] = is_h
            df_target.loc[0, "前日祝日フラグ"] = prev_h
            df_target.loc[0, "月"] = m
            df_target.loc[0, "週回数"] = week_of_month(target_date)
            df_target.loc[0, "month_weekday_total"] = weekday_count_in_month
            df_target.loc[0, "weekday_count_in_month"] = weekday_count_in_month
            df_target.loc[0, "weekday_ratio_in_month"] = weekday_ratio_in_month
            df_target.loc[0, "total_outpatient_count"] = int(total_outpatient_count)
            
            # 天気
            df_target.loc[0, "雨フラグ"] = 1 if "雨" in wcat else 0
            df_target.loc[0, "雪フラグ"] = 1 if "雪" in wcat else 0
            for c in ["晴", "曇", "雨", "雪"]:
                col_name = f"天気カテゴリ_{c}"
                if col_name in df_target.columns:
                     df_target.loc[0, col_name] = 1 if c == wcat else 0
            
            # 時間枠
            df_target.loc[0, "hour"] = ts.hour
            df_target.loc[0, "minute"] = ts.minute
            if f"dayofweek_{dow}" in df_target.columns:
                df_target.loc[0, f"dayofweek_{dow}"] = 1
            df_target.loc[0, "is_first_slot"] = 1 if (ts.hour==8 and ts.minute==0) else 0
            df_target.loc[0, "is_second_slot"] = 1 if (ts.hour==8 and ts.minute==30) else 0
            df_target.loc[0, "slot"] = slot
            
            # Baseline & State
            df_target.loc[0, "arr_base"] = float(arr_base)
            df_target.loc[0, "svc_base"] = float(svc_base)
            df_target.loc[0, "wait_base"] = float(wait_base)
            
            df_target.loc[0, "queue_at_start_truth"] = float(queue_at_start)
            df_target.loc[0, "arr_lag_30"] = float(lags_arr["arr_lag_30"])
            df_target.loc[0, "arr_lag_60"] = float(lags_arr["arr_lag_60"])
            df_target.loc[0, "arr_lag_90"] = float(lags_arr["arr_lag_90"])
            df_target.loc[0, "arr_roll_60"] = float((lags_arr["arr_lag_30"] + lags_arr["arr_lag_60"]) / 2.0)
            
            df_target.loc[0, "svc_lag_30"] = float(lags_svc["svc_lag_30"])
            df_target.loc[0, "svc_lag_60"] = float(lags_svc["svc_lag_60"])
            df_target.loc[0, "svc_lag_90"] = float(lags_svc["svc_lag_90"])
            df_target.loc[0, "svc_roll_60"] = float((lags_svc["svc_lag_30"] + lags_svc["svc_lag_60"]) / 2.0)
            
            df_target.loc[0, "cum_arrivals"] = int(cum_arrivals)
            df_target.loc[0, "cum_service"] = int(cum_service)

        # --- 1. 受付数予測 (Arrivals) ---
        cf = _make_zero_df(arr_cols)
        set_common_features(cf)
        pred_arr = _predict_booster(arr_bst, arr_cols, cf)
        arr_i = max(0, int(round(pred_arr)))

        # --- 2. 処理数予測 (Service) ---
        mf = _make_zero_df(multi_cols)
        set_common_features(mf) # arr_i 反映前
        pred_svc = _predict_booster(svc_bst, multi_cols, mf)
        svc_i = max(0, int(round(pred_svc)))

        # キュー更新
        q_next = max(0.0, float(queue_at_start) + float(arr_i) - float(svc_i))

        # --- 3. 待ち時間予測 (Wait) ---
        # ★重要修正: 対数変換(log1p)で学習したので、expm1 で戻す
        raw_pred = _predict_booster(wait_bst, multi_cols, mf)
        pred_wait_model = max(0.0, float(np.expm1(raw_pred)))

        # 物理モデル (行列 ÷ 処理速度 * 30分)
        safe_svc = max(float(svc_i), 0.5) # ゼロ除算防止
        wait_phy = (float(queue_at_start) / safe_svc) * 30.0
        wait_phy = min(wait_phy, 300.0)   # 暴走防止キャップ
        
        # 物理補正 (Calibration)
        wait_phy_calibrated = max(0.0, a * wait_phy + b)

        # アンサンブル (AI vs Physics)
        wait_blend = alpha * pred_wait_model + (1.0 - alpha) * wait_phy_calibrated
        
        # 下限フロア (極端な下振れ防止)
        wait_floor = float(wait_base) * floor_ratio
        wait_final = max(wait_floor, wait_blend)

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": int(arr_i),
            "予測呼出数(処理数)": int(svc_i),
            "予測待ち人数(人)": int(round(q_next)),
            "予測平均待ち時間(分)": int(round(wait_final)),
        })

        # 状態更新
        lags_arr = {
            "arr_lag_30": float(arr_i), 
            "arr_lag_60": float(lags_arr["arr_lag_30"]), 
            "arr_lag_90": float(lags_arr["arr_lag_60"])
        }
        lags_svc = {
            "svc_lag_30": float(svc_i), 
            "svc_lag_60": float(lags_svc["svc_lag_30"]), 
            "svc_lag_90": float(lags_svc["svc_lag_60"])
        }
        cum_arrivals += int(arr_i)
        cum_service  += int(svc_i)
        queue_at_start = q_next

    return pd.DataFrame(results)

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="A病院 混雑予測", layout="wide")
    st.title("🏥 A病院 採血 待ち時間予測AI (v3.0)")
    st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    st.caption("AIモデル + 待ち行列理論ハイブリッド / ログ変換対応版")

    # ファイルチェック
    required_files = [
        ARR_MODEL_PATH, SVC_MODEL_PATH, WAIT_MODEL_PATH,
        ARR_COLS_PATH, MULTI_COLS_PATH, BASELINE_PATH, CALIB_PATH
    ]
    missing = [f.name for f in required_files if not f.exists()]
    if missing:
        st.error(f"⚠️ 以下のモデルファイルが見つかりません (modelsフォルダを確認してください):\n\n" + "\n".join(missing))
        st.stop()

    with st.sidebar:
        st.header("予測条件入力")
        target_date = st.date_input("日付選択", value=date.today() + timedelta(days=1))
        
        st.subheader("来院状況")
        # デフォルト値を少し高めに設定（安全側）
        default_pat = 1300
        total_out = st.number_input("延べ外来患者数 (予測)", min_value=0, value=default_pat, step=50, 
                                    help="過去の実績: 平日1200-1500人程度")
        
        weather = st.selectbox("天気予報", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=1)
        
        run_btn = st.button("予測実行", type="primary")
        
        st.divider()
        st.info(f"Model Version: v3.0\nWait Log-Transform: ON")

    if run_btn:
        with st.spinner("AIがシミュレーション中..."):
            df_res = simulate_one_day(target_date, int(total_out), str(weather))
        
        # 結果表示
        st.success(f"✅ {target_date.strftime('%Y/%m/%d')} の予測が完了しました")
        
        # メトリクス
        col1, col2, col3, col4 = st.columns(4)
        peak_wait = df_res["予測平均待ち時間(分)"].max()
        avg_wait = df_res["予測平均待ち時間(分)"].mean()
        peak_time = df_res.loc[df_res["予測平均待ち時間(分)"].idxmax(), "時間帯"]
        total_arr = df_res["予測受付数"].sum()

        col1.metric("ピーク待ち時間", f"{peak_wait} 分", f"@{peak_time}", delta_color="inverse")
        col2.metric("平均待ち時間", f"{avg_wait:.1f} 分")
        col3.metric("最大待ち人数", f"{df_res['予測待ち人数(人)'].max()} 人")
        col4.metric("総受付数", f"{total_arr} 人")

        # グラフ描画
        st.subheader("📊 混雑推移グラフ")
        chart_data = df_res.set_index("時間帯")[["予測平均待ち時間(分)", "予測待ち人数(人)"]]
        st.line_chart(chart_data)

        # データテーブル
        with st.expander("詳細データテーブル (クリックで開閉)"):
            st.dataframe(df_res.style.highlight_max(axis=0, color="#fffdc9"), use_container_width=True)
            
            csv = df_res.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "CSVダウンロード",
                data=csv,
                file_name=f"predict_{target_date}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
