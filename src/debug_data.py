import duckdb
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"

con = duckdb.connect()
con.execute("""
    CREATE TABLE transactions AS
    SELECT * FROM read_csv_auto(?)
""", [str(DATA_PATH)])

print("Total rows:")
print(con.execute("SELECT COUNT(*) FROM transactions").df())

print("\nDistinct nameOrig count:")
print(con.execute("SELECT COUNT(DISTINCT nameOrig) FROM transactions").df())

print("\nDistribution of transactions per nameOrig:")
print(con.execute("""
    SELECT txn_count, COUNT(*) AS how_many_customers
    FROM (
        SELECT nameOrig, COUNT(*) AS txn_count
        FROM transactions
        GROUP BY nameOrig
    )
    GROUP BY txn_count
    ORDER BY txn_count
    LIMIT 10
""").df())

print("\nColumn types:")
print(con.execute("DESCRIBE transactions").df())

con.close()
