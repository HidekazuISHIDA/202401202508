
import json
from pathlib import Path
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

COUNT_MODEL_PATH = MODELS_DIR / "model_A_timeseries.json"
WAIT_MODEL_PATH  = MODELS_DIR / "model_A_waittime_30min.json"
QUEUE_MODEL_PATH = MODELS_DIR / "model_A_queue_30min.json"

COUNT_COLS_PATH  = MODELS_DIR / "columns_A_timeseries.json"
MULTI_COLS_PATH  = MODELS_DIR / "columns_A_multi_30min.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"

OPEN_HOUR = 8
CLOSE_HOUR = 18
FREQ_MIN = 30

def _load_holidays() -> set:
    if not HOLIDAY_CSV_PATH.exists():
        return set()
    df = pd.read_csv(HOLIDAY_CSV_PATH, encoding="utf-8", engine="python")
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "日付"]:
            col = c
            break
    if col is None:
        col = df.columns[0]
    s = pd.to_datetime(df[col], errors="coerce").dropna().dt.date
    return set(s.tolist())

HOLIDAYS = _load_holidays()

def is_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    if d in HOLIDAYS:
        return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return True
    return False

@st.cache_resource
def load_models_and_columns():
    count_cols = json.loads(COUNT_COLS_PATH.read_text(encoding="utf-8"))
    multi_cols = json.loads(MULTI_COLS_PATH.read_text(encoding="utf-8"))

    count_booster = xgb.Booster()
    count_booster.load_model(str(COUNT_MODEL_PATH))

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAIT_MODEL_PATH))

    queue_booster = xgb.Booster()
    queue_booster.load_model(str(QUEUE_MODEL_PATH))

    return count_booster, count_cols, wait_booster, queue_booster, multi_cols

def _make_zero_df(cols):
    return pd.DataFrame({c: [0] for c in cols})

def _predict_booster(booster: xgb.Booster, cols, df: pd.DataFrame) -> float:
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    dmat = xgb.DMatrix(X, feature_names=list(cols))
    pred = booster.predict(dmat)
    return float(pred[0])

def simulate_one_day(target_date: date, total_outpatient_count: int, weather: str) -> pd.DataFrame:
    count_booster, count_cols, wait_booster, queue_booster, multi_cols = load_models_and_columns()

    is_h = is_holiday(target_date)
    prev = target_date - timedelta(days=1)
    is_prev_h = is_holiday(prev)

    start = datetime(target_date.year, target_date.month, target_date.day, OPEN_HOUR, 0)
    end   = datetime(target_date.year, target_date.month, target_date.day, CLOSE_HOUR, 0)
    time_slots = pd.date_range(start=start, end=end, freq=f"{FREQ_MIN}min")

    lags = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0}
    queue_at_start = 0

    results = []
    for ts in time_slots:
        cf = _make_zero_df(count_cols)

        if "hour" in cf.columns: cf.loc[0, "hour"] = int(ts.hour)
        if "minute" in cf.columns: cf.loc[0, "minute"] = int(ts.minute)
        if "月" in cf.columns: cf.loc[0, "月"] = int(ts.month)
        if "週回数" in cf.columns: cf.loc[0, "週回数"] = int((ts.day - 1) // 7 + 1)
        if "前日祝日フラグ" in cf.columns: cf.loc[0, "前日祝日フラグ"] = int(is_prev_h)
        if "total_outpatient_count" in cf.columns: cf.loc[0, "total_outpatient_count"] = int(total_outpatient_count)
        if "is_holiday" in cf.columns: cf.loc[0, "is_holiday"] = int(is_h)

        if "雨フラグ" in cf.columns: cf.loc[0, "雨フラグ"] = 1 if ("雨" in weather) else 0
        if "雪フラグ" in cf.columns: cf.loc[0, "雪フラグ"] = 1 if ("雪" in weather) else 0

        wcat = weather[0] if weather else ""
        wcol = f"天気カテゴリ_{wcat}"
        if wcol in cf.columns: cf.loc[0, wcol] = 1

        dcol = f"dayofweek_{ts.dayofweek}"
        if dcol in cf.columns: cf.loc[0, dcol] = 1

        rolling_mean = (lags["lag_30min"] + lags["lag_60min"]) / 2.0
        if "rolling_mean_60min" in cf.columns: cf.loc[0, "rolling_mean_60min"] = float(rolling_mean)
        for k, v in lags.items():
            if k in cf.columns:
                cf.loc[0, k] = float(v)

        pred_reception = _predict_booster(count_booster, count_cols, cf)
        pred_reception_i = max(0, int(round(float(pred_reception))))

        mf = _make_zero_df(multi_cols)
        if "hour" in mf.columns: mf.loc[0, "hour"] = int(ts.hour)
        if "minute" in mf.columns: mf.loc[0, "minute"] = int(ts.minute)
        if "reception_count" in mf.columns: mf.loc[0, "reception_count"] = int(pred_reception_i)
        if "queue_at_start_of_slot" in mf.columns: mf.loc[0, "queue_at_start_of_slot"] = int(queue_at_start)
        if "月" in mf.columns: mf.loc[0, "月"] = int(ts.month)
        if "週回数" in mf.columns: mf.loc[0, "週回数"] = int((ts.day - 1) // 7 + 1)
        if "前日祝日フラグ" in mf.columns: mf.loc[0, "前日祝日フラグ"] = int(is_prev_h)
        if "total_outpatient_count" in mf.columns: mf.loc[0, "total_outpatient_count"] = int(total_outpatient_count)
        if "is_holiday" in mf.columns: mf.loc[0, "is_holiday"] = int(is_h)
        if "雨フラグ" in mf.columns: mf.loc[0, "雨フラグ"] = 1 if ("雨" in weather) else 0
        if "雪フラグ" in mf.columns: mf.loc[0, "雪フラグ"] = 1 if ("雪" in weather) else 0
        wcol2 = f"天気カテゴリ_{wcat}"
        if wcol2 in mf.columns: mf.loc[0, wcol2] = 1
        dcol2 = f"dayofweek_{ts.dayofweek}"
        if dcol2 in mf.columns: mf.loc[0, dcol2] = 1

        pred_queue = _predict_booster(queue_booster, multi_cols, mf)
        pred_wait  = _predict_booster(wait_booster,  multi_cols, mf)

        pred_queue_i = max(0, int(round(float(pred_queue))))
        pred_wait_i  = max(0, int(round(float(pred_wait))))

        results.append({
            "時間帯": ts.strftime("%H:%M"),
            "予測受付数": pred_reception_i,
            "予測待ち人数(人)": pred_queue_i,
            "予測平均待ち時間(分)": pred_wait_i,
        })

        lags = {"lag_30min": float(pred_reception_i), "lag_60min": float(lags["lag_30min"]), "lag_90min": float(lags["lag_60min"])}
        queue_at_start = pred_queue_i

    return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="A病院 採血 待ち人数・待ち時間 予測", layout="wide")
    st.title("🏥 A病院 採血 待ち人数・待ち時間 予測（3モデル統合）")
    st.caption("※ Streamlit Cloud 互換（jpholiday不使用・祝日CSVで判定）")

    with st.sidebar:
        st.header("入力")
        target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
        total_out = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
        weather = st.selectbox("天気（簡易）", ["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=0)
        run = st.button("シミュレーション実行", type="primary")

        st.divider()
        st.subheader("モデル/ファイル")
        st.write("受付数モデル:", COUNT_MODEL_PATH.name)
        st.write("待ち時間モデル:", WAIT_MODEL_PATH.name)
        st.write("待ち人数モデル:", QUEUE_MODEL_PATH.name)

    missing = []
    for p in [COUNT_MODEL_PATH, WAIT_MODEL_PATH, QUEUE_MODEL_PATH, COUNT_COLS_PATH, MULTI_COLS_PATH]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        st.error(
    """必要ファイルが不足しています。
models/ に以下を配置してください：

- model_A_timeseries.json
- columns_A_timeseries.json
"""
)

        st.stop()

    if run:
        with st.spinner("計算中..."):
            df = simulate_one_day(target, int(total_out), str(weather))
        st.success(f"{target} の予測が完了しました。")

        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.subheader("結果テーブル")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("CSVダウンロード", data=csv, file_name=f"A_predict_{target}.csv", mime="text/csv")

        with c2:
            st.subheader("可視化")
            st.line_chart(df.set_index("時間帯")[["予測平均待ち時間(分)"]])
            st.bar_chart(df.set_index("時間帯")[["予測待ち人数(人)"]])

    st.divider()
    st.caption("※ 祝日判定は data/syukujitsu.csv を参照（なければ土日・年末年始のみ）")

if __name__ == "__main__":
    main()
