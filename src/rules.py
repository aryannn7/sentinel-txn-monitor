import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sentinel.duckdb"


def flag_high_value_transactions(multiplier: float = 3.0) -> pd.DataFrame:
    """
    Flag transactions where amount exceeds multiplier times
    the sender's average transaction amount.

    This is a velocity rule: unusual spend relative to a
    customer's own baseline.
    """
    con = duckdb.connect(str(DB_PATH))

    query = """
        WITH customer_stats AS (
            SELECT
                nameOrig,
                amount,
                AVG(amount) OVER (PARTITION BY nameOrig) AS avg_amount,
                COUNT(*)    OVER (PARTITION BY nameOrig) AS txn_count
            FROM transactions
        )
        SELECT
            nameOrig,
            amount,
            ROUND(avg_amount, 2)                    AS avg_amount,
            ROUND(amount / avg_amount, 2)           AS ratio,
            txn_count,
            amount > (avg_amount * ?)               AS is_flagged
        FROM customer_stats
        WHERE amount > (avg_amount * ?)
        AND   txn_count > 1
        ORDER BY ratio DESC
        LIMIT 100
    """

    df = con.execute(query, [multiplier, multiplier]).df()
    con.close()
    return df


if __name__ == "__main__":
    flagged = flag_high_value_transactions(multiplier=3.0)
    print(f"Flagged transactions: {len(flagged)}")
    print(flagged[["nameOrig", "amount", "avg_amount", "ratio"]].head(10))