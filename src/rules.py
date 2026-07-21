import duckdb
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"


def flag_draining_transactions() -> pd.DataFrame:
    """
    Flag every transaction that fully drains the origin account.
    No LIMIT here: we want the true fraud rate across all flagged
    rows, not just a sorted slice of them.
    """
    con = duckdb.connect()

    con.execute("""
        CREATE TABLE transactions AS
        SELECT * FROM read_csv_auto(?)
    """, [str(DATA_PATH)])

    query = """
        SELECT
            nameOrig,
            type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            isFraud
        FROM transactions
        WHERE type IN ('TRANSFER', 'CASH_OUT')
        AND   oldbalanceOrg > 0
        AND   newbalanceOrig = 0
    """

    df = con.execute(query).df()

    total_fraud_query = "SELECT COUNT(*) AS total_fraud FROM transactions WHERE isFraud = 1"
    total_fraud = con.execute(total_fraud_query).df()["total_fraud"][0]

    con.close()
    return df, total_fraud


if __name__ == "__main__":
    print("Loading transactions and flagging account-draining events...")
    flagged, total_fraud_in_dataset = flag_draining_transactions()

    true_positives = flagged["isFraud"].sum()
    total_flagged = len(flagged)

    precision = true_positives / total_flagged * 100
    recall = true_positives / total_fraud_in_dataset * 100

    print(f"\nTotal flagged (draining transactions): {total_flagged}")
    print(f"True fraud caught within flagged:       {true_positives}")
    print(f"Total fraud cases in entire dataset:    {total_fraud_in_dataset}")
    print(f"\nPrecision (of what we flagged, % actually fraud): {precision:.1f}%")
    print(f"Recall (of all fraud, % we actually caught):      {recall:.1f}%")

    print("\nSample of flagged, largest first:")
    print(flagged.sort_values("amount", ascending=False)[
        ["nameOrig", "type", "amount", "isFraud"]
    ].head(10))
    