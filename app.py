import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Sentinel: Transaction Monitoring", layout="wide")

DATA_PATH = Path(__file__).parent / "scored_transactions.parquet"


@st.cache_data
def load_scored_data():
    return pd.read_parquet(DATA_PATH)


def evaluate_at_threshold(df: pd.DataFrame, threshold: float) -> dict:
    flagged = df[df["anomaly_score"] >= threshold]
    true_positives = flagged["isFraud"].sum()
    total_flagged = len(flagged)
    total_fraud = df["isFraud"].sum()

    precision = (true_positives / total_flagged * 100) if total_flagged > 0 else 0
    recall = (true_positives / total_fraud * 100) if total_fraud > 0 else 0

    return {
        "total_flagged": total_flagged,
        "true_positives": int(true_positives),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }


df = load_scored_data()

st.title("Sentinel: Transaction Monitoring")
st.caption(
    f"{len(df):,} transactions analysed (TRANSFER and CASH_OUT only) · "
    f"{int(df['isFraud'].sum()):,} confirmed fraud cases"
)

st.divider()

min_score = float(df["anomaly_score"].min())
max_score = float(df["anomaly_score"].max())

threshold = st.slider(
    "Suspicion threshold — higher means fewer, more suspicious transactions flagged",
    min_value=min_score,
    max_value=max_score,
    value=float(df["anomaly_score"].quantile(0.99)),
    step=(max_score - min_score) / 500,
)

result = evaluate_at_threshold(df, threshold)

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{result['precision']}%")
col2.metric("Recall", f"{result['recall']}%")
col3.metric("Flagged transactions", f"{result['total_flagged']:,}")

st.caption(
    f"At this threshold, an analyst reviews {result['total_flagged']:,} transactions "
    f"to catch {result['true_positives']:,} of {int(df['isFraud'].sum()):,} real fraud cases."
)

st.divider()

st.subheader("Precision vs recall across all thresholds")

curve_thresholds = np.linspace(min_score, max_score, 40)
curve_data = pd.DataFrame(
    [evaluate_at_threshold(df, t) for t in curve_thresholds]
)
curve_data["threshold"] = curve_thresholds

st.line_chart(
    curve_data.set_index("threshold")[["precision", "recall"]]
)

st.divider()

st.subheader("Top flagged transactions at current threshold")
flagged_display = (
    df[df["anomaly_score"] >= threshold]
    .sort_values("anomaly_score", ascending=False)
    [["nameOrig", "type", "amount", "anomaly_score", "isFraud"]]
    .head(50)
)
st.dataframe(flagged_display, use_container_width=True)