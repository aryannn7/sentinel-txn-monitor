import duckdb
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"
DB_PATH = Path(__file__).parent.parent / "sentinel.duckdb"


def load_transactions() -> pd.DataFrame:
    """Load PaySim CSV into DuckDB and return as a DataFrame."""
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS transactions AS
        SELECT * FROM read_csv_auto(?)
    """, [str(DATA_PATH)])

    df = con.execute("SELECT * FROM transactions LIMIT 5").df()
    con.close()
    return df


if __name__ == "__main__":
    df = load_transactions()
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Rows loaded: {len(df)}")

    