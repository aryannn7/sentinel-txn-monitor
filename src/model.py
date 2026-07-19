import duckdb
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"


def load_and_engineer_features() -> pd.DataFrame:
    """
    Load transactions and engineer features for the Isolation Forest.
    We only use TRANSFER and CASH_OUT since that's where fraud
    actually occurs in this dataset (confirmed in Day 2 analysis).
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE transactions AS
        SELECT * FROM read_csv_auto(?)
    """, [str(DATA_PATH)])

    query = """
        SELECT
            nameOrig,
            step,
            type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            isFraud,
            -- engineered features
            (oldbalanceOrg - newbalanceOrig)                  AS balance_delta_orig,
            (newbalanceDest - oldbalanceDest)                 AS balance_delta_dest,
            CASE WHEN oldbalanceOrg > 0
                 THEN amount / oldbalanceOrg
                 ELSE 0 END                                    AS amount_to_balance_ratio,
            CASE WHEN newbalanceOrig = 0 AND oldbalanceOrg > 0
                 THEN 1 ELSE 0 END                              AS drains_account
        FROM transactions
        WHERE type IN ('TRANSFER', 'CASH_OUT')
    """
    df = con.execute(query).df()
    con.close()
    return df


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.01):
    """
    Train Isolation Forest on engineered features.
    contamination is our guess at what fraction of transactions
    are anomalous. Real fraud rate in this filtered subset is
    about 0.3%, but we set it slightly higher to catch a wider net,
    which we can tighten later.
    """
    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"])

    feature_cols = [
        "amount",
        "balance_delta_orig",
        "balance_delta_dest",
        "amount_to_balance_ratio",
        "drains_account",
        "type_encoded",
    ]
    X = df[feature_cols]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # decision_function: higher = more normal, lower = more anomalous
    # we flip sign so higher = more anomalous, which is more intuitive
    df["anomaly_score"] = -model.decision_function(X)
    df["predicted_anomaly"] = model.predict(X)  # -1 = anomaly, 1 = normal

    return df, model


def evaluate_at_threshold(df: pd.DataFrame, score_threshold: float) -> dict:
    """
    Given a score threshold, compute precision and recall.
    This is what will power the dashboard slider.
    """
    flagged = df[df["anomaly_score"] >= score_threshold]
    true_positives = flagged["isFraud"].sum()
    total_flagged = len(flagged)
    total_fraud = df["isFraud"].sum()

    precision = (true_positives / total_flagged * 100) if total_flagged > 0 else 0
    recall = (true_positives / total_fraud * 100) if total_fraud > 0 else 0

    return {
        "threshold": score_threshold,
        "total_flagged": total_flagged,
        "true_positives": int(true_positives),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }


if __name__ == "__main__":
    print("Loading data and engineering features...")
    df = load_and_engineer_features()
    print(f"Transactions after filtering to TRANSFER/CASH_OUT: {len(df)}")

    print("\nTraining Isolation Forest...")
    df, model = train_isolation_forest(df, contamination=0.01)

    print("\nEvaluating at several score thresholds:")
    thresholds_to_try = df["anomaly_score"].quantile([0.90, 0.95, 0.99, 0.995, 0.999]).values
    for t in thresholds_to_try:
        result = evaluate_at_threshold(df, t)
        print(result)

    df.to_parquet(Path(__file__).parent.parent / "scored_transactions.parquet")
    print("\nSaved scored transactions to scored_transactions.parquet")