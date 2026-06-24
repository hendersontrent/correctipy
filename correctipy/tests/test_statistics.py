import numpy as np
import pandas as pd
import pytest

from correctipy import resampled_ttest, kfold_ttest, repkfold_ttest


@pytest.fixture
def vectors():
    rng = np.random.default_rng(42)
    x = rng.normal(0.6, 0.1, 30)
    y = rng.normal(0.4, 0.1, 30)
    return x, y


@pytest.fixture
def rep_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'model': np.repeat([1, 2], 60),
        'values': np.concatenate((rng.normal(0.6, 0.1, 60), rng.normal(0.4, 0.1, 60))),
        'k': [1, 1, 2, 2] * 30,
        'r': [1, 2] * 60,
    })


# ---- shape parity with correctR (returns statistic + p_value) ----

def test_resampled_returns_two_columns(vectors):
    x, y = vectors
    out = resampled_ttest(x, y, n=30, n1=80, n2=20)
    assert list(out.columns) == ["statistic", "p_value"]
    assert out.shape == (1, 2)


def test_kfold_returns_two_columns(vectors):
    x, y = vectors
    out = kfold_ttest(x, y, n=100, k=30)
    assert list(out.columns) == ["statistic", "p_value"]
    assert out.shape == (1, 2)


def test_repkfold_returns_two_columns(rep_data):
    out = repkfold_ttest(rep_data, n1=80, n2=20, k=2, r=2)
    assert list(out.columns) == ["statistic", "p_value"]
    assert out.shape == (1, 2)


# ---- formulas match correctR exactly ----

def test_resampled_formula(vectors):
    x, y = vectors
    d = x - y
    expected = np.mean(d) / np.sqrt(np.var(d, ddof=1) * (1/30 + 20/80))
    out = resampled_ttest(x, y, n=30, n1=80, n2=20)
    assert np.isclose(out.statistic[0], expected)


def test_kfold_formula(vectors):
    x, y = vectors
    d = x - y
    expected = np.mean(d) / np.sqrt(np.var(d, ddof=1) * ((1/100 + 1/30) / (1 - 1/30)))
    out = kfold_ttest(x, y, n=100, k=30)
    assert np.isclose(out.statistic[0], expected)


# ---- tailed / greater behaviour ----

def test_two_tailed_is_double_smaller_one_tailed(vectors):
    x, y = vectors
    two = resampled_ttest(x, y, 30, 80, 20)
    one_x = resampled_ttest(x, y, 30, 80, 20, tailed="one", greater="x")
    one_y = resampled_ttest(x, y, 30, 80, 20, tailed="one", greater="y")
    assert np.isclose(one_x.p_value[0] + one_y.p_value[0], 1.0)
    assert np.isclose(two.p_value[0], 2 * min(one_x.p_value[0], one_y.p_value[0]))


def test_repkfold_one_tailed_greater_label(rep_data):
    one1 = repkfold_ttest(rep_data, 80, 20, 2, 2, tailed="one", greater=1)
    one2 = repkfold_ttest(rep_data, 80, 20, 2, 2, tailed="one", greater=2)
    assert np.isclose(one1.p_value[0] + one2.p_value[0], 1.0)


# ---- sum(d) == 0 edge case ----

def test_resampled_zero_difference():
    z = np.array([1.0, 2.0, 3.0])
    out = resampled_ttest(z, z, n=3, n1=80, n2=20)
    assert out.statistic[0] == 0 and out.p_value[0] == 1


# ---- argument validation ----

def test_mismatched_lengths():
    with pytest.raises(ValueError):
        resampled_ttest(np.array([1.0, 2.0]), np.array([1.0]), n=2, n1=80, n2=20)


def test_invalid_tailed(vectors):
    x, y = vectors
    with pytest.raises(ValueError):
        resampled_ttest(x, y, 30, 80, 20, tailed="three")


def test_one_tailed_requires_greater(vectors):
    x, y = vectors
    with pytest.raises(ValueError):
        kfold_ttest(x, y, 100, 30, tailed="one")


def test_repkfold_requires_columns():
    bad = pd.DataFrame({"model": [1, 2], "values": [0.1, 0.2]})
    with pytest.raises(ValueError):
        repkfold_ttest(bad, 80, 20, 2, 2)
