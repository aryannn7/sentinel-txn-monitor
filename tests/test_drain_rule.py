import duckdb
import pandas as pd
import pytest


def flag_draining_transactions_on_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same logic as src/rules.py's SQL query, run against an in-memory
    DataFrame instead of the full CSV, so tests run in milliseconds
    instead of loading 6 million rows.
    """
    con = duckdb.connect()
    con.register("transactions", df)
    result = con.execute("""
        SELECT nameOrig, type, amount, oldbalanceOrg, newbalanceOrig, isFraud
        FROM transactions
        WHERE type IN ('TRANSFER', 'CASH_OUT')
        AND   oldbalanceOrg > 0
        AND   newbalanceOrig = 0
    """).df()
    con.close()
    return result


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "nameOrig":       ["C1", "C2", "C3", "C4", "C5"],
        "type":           ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER", "CASH_OUT"],
        "amount":         [500.0, 200.0, 300.0, 100.0, 50.0],
        "oldbalanceOrg":  [500.0, 200.0, 300.0, 100.0, 0.0],
        "newbalanceOrig": [0.0,   0.0,   0.0,   50.0,  0.0],
        "isFraud":        [1,     0,     1,     0,     0],
    })


def test_flags_only_transfer_and_cash_out(sample_transactions):
    """
    PAYMENT type must never be flagged, even if it drains the account,
    since the rule is scoped to TRANSFER and CASH_OUT only.
    """
    result = flag_draining_transactions_on_df(sample_transactions)
    assert "PAYMENT" not in result["type"].values


def test_flags_only_fully_drained_accounts(sample_transactions):
    """
    C4 transfers 100 but leaves 50 behind, a partial transfer,
    not a drain, so it must not be flagged.
    """
    result = flag_draining_transactions_on_df(sample_transactions)
    assert "C4" not in result["nameOrig"].values


def test_requires_positive_starting_balance(sample_transactions):
    """
    C5 starts at zero balance already, there's nothing to drain,
    so it should not be flagged even though newbalanceOrig is 0.
    """
    result = flag_draining_transactions_on_df(sample_transactions)
    assert "C5" not in result["nameOrig"].values


def test_correctly_flags_genuine_drains(sample_transactions):
    """
    C1 and C3 both start with a positive balance and end at zero
    via TRANSFER or CASH_OUT. Both should be flagged.
    """
    result = flag_draining_transactions_on_df(sample_transactions)
    flagged_names = set(result["nameOrig"].values)
    assert flagged_names == {"C1", "C3"}