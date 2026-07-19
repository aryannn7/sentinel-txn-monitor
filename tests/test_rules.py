import pandas as pd
import pytest
from src.model import evaluate_at_threshold


@pytest.fixture
def sample_scored_df():
    """
    A small, hand-built dataframe standing in for real scored data.
    Using known values means we know the correct answer in advance,
    which is what makes this a real test rather than a smoke test.
    """
    return pd.DataFrame({
        "anomaly_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "isFraud":       [1,   1,   0,   1,   0,   0,   0,   0],
    })


def test_evaluate_at_threshold_basic_precision_recall(sample_scored_df):
    """
    At threshold 0.7, rows with score >= 0.7 are flagged: three rows
    (0.9, 0.8, 0.7), of which two are real fraud (0.9, 0.8).
    Total fraud in the full dataframe is 3.
    Precision = 2/3, Recall = 2/3.
    """
    result = evaluate_at_threshold(sample_scored_df, score_threshold=0.7)

    assert result["total_flagged"] == 3
    assert result["true_positives"] == 2
    assert result["precision"] == pytest.approx(66.67, abs=0.1)
    assert result["recall"] == pytest.approx(66.67, abs=0.1)


def test_evaluate_at_threshold_catches_everything(sample_scored_df):
    """
    A very low threshold flags every row, so recall must be 100%.
    """
    result = evaluate_at_threshold(sample_scored_df, score_threshold=0.0)

    assert result["total_flagged"] == len(sample_scored_df)
    assert result["recall"] == pytest.approx(100.0, abs=0.1)


def test_evaluate_at_threshold_catches_nothing(sample_scored_df):
    """
    A threshold above every score in the data flags nothing.
    Precision and recall should both safely return 0, not crash
    on a division by zero.
    """
    result = evaluate_at_threshold(sample_scored_df, score_threshold=1.0)

    assert result["total_flagged"] == 0
    assert result["precision"] == 0
    assert result["recall"] == 0


def test_precision_recall_never_exceed_100(sample_scored_df):
    """
    Sanity bound: no matter the threshold, these percentages
    must stay within a valid 0 to 100 range.
    """
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = evaluate_at_threshold(sample_scored_df, score_threshold=t)
        assert 0 <= result["precision"] <= 100
        assert 0 <= result["recall"] <= 100