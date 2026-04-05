"""Dynamic DCA weight computation using Gradient Boosting on-chain valuation model.

This module computes daily investment weights for a Bitcoin DCA strategy
using a Gradient Boosting classifier trained on 14 on-chain features
identified through consensus feature selection (LASSO + RF + GBM) in Part 4.

Model predicts P(favourable 30-day accumulation window) and converts
predictions to buy multipliers:
- High probability -> buy more (overweight)
- Low probability -> buy less (underweight)
- Budget remains neutral (weights sum to 1.0 per window)

Walk-forward training retrains the model quarterly using only past data,
ensuring no future information leaks into predictions.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"

# Rolling window parameters
ROLLING_WINDOW = 730  # 2-year lookback for z-scores (same as Part 2/4)
MA_WINDOW = 200  # 200-day MA (kept for reference)

# Walk-forward model parameters
MIN_TRAIN_DAYS = 730  # Minimum training samples before model can predict
RETRAIN_FREQ_DAYS = 90  # Retrain model quarterly
FWD_HORIZON = 30  # 30-day forward return horizon for target

# Strategy parameters
MIN_W = 1e-6
DYNAMIC_STRENGTH = 4.0  # Multiplier scaling intensity

# GBM hyperparameters (from Part 4 notebook walk-forward CV)
GBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "subsample": 0.8,
    "random_state": 42,
}

# 14 features selected via consensus ranking in Part 4 notebook
# (LASSO + Random Forest + Gradient Boosting importance, threshold >= 0.15)
SELECTED_FEATURES = [
    "fee_zscore",
    "CapMVRVCur",
    "exch_supply_zscore",
    "MVRV_zscore",
    "price_mom_90d",
    "MVRV_30d",
    "price_zscore",
    "hashrate_zscore",
    "volatility_30d",
    "fee_to_issuance_ratio",
    "hashrate_mom_30d",
    "NVT_14d",
    "composite_signal",
    "issuance_rate",
]

# Feature column names (for compatibility with backtest framework)
FEATS = SELECTED_FEATURES


# =============================================================================
# Helper Functions
# =============================================================================


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


def _zscore_rolling(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Compute rolling z-score with configurable lookback."""
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window, min_periods=window // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (series - rolling_mean) / rolling_std
    return result


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Compute percentage change over N periods."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = series / series.shift(periods) - 1
    return result


# =============================================================================
# Walk-Forward Model Training
# =============================================================================


def _generate_walk_forward_predictions(
    features: pd.DataFrame,
    price: pd.Series,
) -> pd.Series:
    """Generate out-of-sample GBM predictions via walk-forward retraining.

    Retrains quarterly using all available past data. For each retrain
    point, the model predicts forward until the next retrain point.

    Target: 30-day forward return > expanding median (no look-ahead).

    Args:
        features: DataFrame with SELECTED_FEATURES columns (already lagged)
        price: Unlagged price series for target computation

    Returns:
        Series of predicted P(favourable), NaN where model cannot predict
    """
    probas = pd.Series(np.nan, index=features.index)

    # Forward returns for target construction (training only)
    fwd_returns = price.shift(-FWD_HORIZON) / price - 1

    # Identify dates where all features are finite (not NaN/Inf)
    X_all = features[SELECTED_FEATURES].values
    valid_mask = np.all(np.isfinite(X_all), axis=1)
    valid_dates = features.index[valid_mask]

    if len(valid_dates) < MIN_TRAIN_DAYS + 100:
        logging.warning("Insufficient data for walk-forward GBM training")
        return probas

    # Quarterly retraining schedule
    first_predict_date = valid_dates[MIN_TRAIN_DAYS]
    retrain_schedule = pd.date_range(
        start=first_predict_date,
        end=valid_dates[-1],
        freq=f"{RETRAIN_FREQ_DAYS}D",
    )

    logging.info(
        f"Walk-forward GBM: {len(retrain_schedule)} retraining points, "
        f"{len(valid_dates)} valid feature dates"
    )

    for rd_idx, retrain_date in enumerate(retrain_schedule):
        # --- Training: all valid dates before this retrain point ---
        train_dates = valid_dates[valid_dates < retrain_date]
        if len(train_dates) < MIN_TRAIN_DAYS:
            continue

        X_train = features.loc[train_dates, SELECTED_FEATURES].values
        fwd_ret_train = fwd_returns.loc[train_dates].values

        # Exclude rows without valid forward returns (last 30 days of window)
        valid_target = np.isfinite(fwd_ret_train)
        X_train = X_train[valid_target]
        fwd_ret_train = fwd_ret_train[valid_target]

        if len(X_train) < 100:
            continue

        # Binary target: forward return > expanding median (no look-ahead)
        expanding_med = pd.Series(fwd_ret_train).expanding().median().values
        y_train = (fwd_ret_train > expanding_med).astype(int)

        # Safety: remove any remaining NaN/Inf rows
        valid_rows = np.isfinite(X_train).all(axis=1)
        X_train = X_train[valid_rows]
        y_train = y_train[valid_rows]

        if len(X_train) < 100 or len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = GradientBoostingClassifier(**GBM_PARAMS)
        model.fit(X_train_s, y_train)

        # --- Predict: this retrain period until the next one ---
        if rd_idx + 1 < len(retrain_schedule):
            predict_end = retrain_schedule[rd_idx + 1]
        else:
            predict_end = valid_dates[-1] + pd.Timedelta(days=1)

        predict_dates = valid_dates[
            (valid_dates >= retrain_date) & (valid_dates < predict_end)
        ]

        if len(predict_dates) == 0:
            continue

        X_pred = features.loc[predict_dates, SELECTED_FEATURES].values
        valid_pred = np.isfinite(X_pred).all(axis=1)

        if valid_pred.any():
            X_pred_s = scaler.transform(X_pred[valid_pred])
            pred_probas = model.predict_proba(X_pred_s)[:, 1]
            probas.loc[predict_dates[valid_pred]] = pred_probas

    n_predicted = probas.notna().sum()
    logging.info(f"Walk-forward GBM: generated {n_predicted} predictions")

    return probas


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 14 on-chain features and walk-forward GBM predictions.

    Features selected via consensus ranking (LASSO + RF + GBM) in Part 4:
    - Valuation: CapMVRVCur, MVRV_30d, MVRV_zscore, NVT_14d, composite_signal
    - Price: price_zscore, price_mom_90d, volatility_30d
    - Exchange: exch_supply_zscore
    - Mining: hashrate_zscore, hashrate_mom_30d, fee_zscore,
              fee_to_issuance_ratio, issuance_rate

    All features lagged 1 day to prevent look-ahead bias.
    Model probability generated via quarterly walk-forward GBM retraining.

    Args:
        df: DataFrame with PriceUSD_coinmetrics and Coin Metrics on-chain columns

    Returns:
        DataFrame with price, model_proba, and feature columns
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()
    idx = price.index
    nan_s = pd.Series(np.nan, index=idx)

    # --- Extract raw on-chain columns ---
    mvrv = df["CapMVRVCur"].reindex(idx) if "CapMVRVCur" in df.columns else nan_s.copy()
    cap_mkt = df["CapMrktCurUSD"].reindex(idx) if "CapMrktCurUSD" in df.columns else nan_s.copy()
    vol_spot = df["volume_reported_spot_usd_1d"].reindex(idx) if "volume_reported_spot_usd_1d" in df.columns else nan_s.copy()
    fee_tot = df["FeeTotNtv"].reindex(idx) if "FeeTotNtv" in df.columns else nan_s.copy()
    iss_tot = df["IssTotNtv"].reindex(idx) if "IssTotNtv" in df.columns else nan_s.copy()
    sply_ex = df["SplyExNtv"].reindex(idx) if "SplyExNtv" in df.columns else nan_s.copy()
    sply_cur = df["SplyCur"].reindex(idx) if "SplyCur" in df.columns else nan_s.copy()
    hashrate = df["HashRate"].reindex(idx) if "HashRate" in df.columns else nan_s.copy()

    # --- Derived metrics ---
    nvt_14d = cap_mkt / vol_spot.rolling(14).mean()
    mvrv_30d = mvrv.rolling(30).mean()

    # --- Z-scores (2-year rolling lookback) ---
    mvrv_zscore = _zscore_rolling(mvrv)
    nvt_zscore = _zscore_rolling(nvt_14d)
    price_zscore = _zscore_rolling(price)
    hashrate_zscore = _zscore_rolling(hashrate)
    fee_zscore = _zscore_rolling(fee_tot)
    exch_supply_ratio = sply_ex / sply_cur
    exch_supply_zscore = _zscore_rolling(exch_supply_ratio)

    # --- Composite valuation signal (MVRV + NVT z-scores averaged) ---
    composite = pd.Series(np.nan, index=idx)
    both_valid = mvrv_zscore.notna() & nvt_zscore.notna() & np.isfinite(nvt_zscore)
    mvrv_only = mvrv_zscore.notna() & ~both_valid
    composite.loc[both_valid] = (mvrv_zscore[both_valid] + nvt_zscore[both_valid]) / 2
    composite.loc[mvrv_only] = mvrv_zscore[mvrv_only]

    # --- Momentum features ---
    price_mom_90d = _pct_change(price, 90)
    daily_return = price / price.shift(1) - 1
    volatility_30d = daily_return.rolling(30).std()
    hashrate_mom_30d = _pct_change(hashrate, 30)

    # --- Mining / supply features ---
    fee_to_issuance_ratio = fee_tot / iss_tot
    issuance_rate = iss_tot / sply_cur

    # --- Build features DataFrame ---
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "CapMVRVCur": mvrv,
            "MVRV_30d": mvrv_30d,
            "MVRV_zscore": mvrv_zscore,
            "NVT_14d": nvt_14d,
            "composite_signal": composite,
            "price_zscore": price_zscore,
            "price_mom_90d": price_mom_90d,
            "volatility_30d": volatility_30d,
            "hashrate_zscore": hashrate_zscore,
            "hashrate_mom_30d": hashrate_mom_30d,
            "fee_zscore": fee_zscore,
            "fee_to_issuance_ratio": fee_to_issuance_ratio,
            "issuance_rate": issuance_rate,
            "exch_supply_zscore": exch_supply_zscore,
        },
        index=idx,
    )

    # --- Lag all model features by 1 day (prevent look-ahead) ---
    for col in SELECTED_FEATURES:
        features[col] = features[col].shift(1)

    # --- Walk-forward GBM predictions ---
    logging.info("Running walk-forward GBM training...")
    features["model_proba"] = _generate_walk_forward_predictions(features, price)

    # NaN model_proba → 0.5 (neutral / uniform weighting during warmup)
    features["model_proba"] = features["model_proba"].fillna(0.5)

    return features


# =============================================================================
# Weight Allocation
# =============================================================================


def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights using cumulative mean normalization.

    signal[i] = raw[i] / mean(raw[0:i+1])

    This ensures weights only depend on past data.
    """
    n = len(raw)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean
    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate weights with lock-on-compute stability.

    Past weights are locked and never change. Future days absorb remainder.

    Args:
        raw: Raw weight values for all dates
        n_past: Number of past/current dates (locked)
        locked_weights: Optional pre-computed locked weights from database

    Returns:
        Weights summing to 1.0
    """
    n = len(raw)
    if n == 0:
        return np.array([])
    if n_past <= 0:
        return np.full(n, 1.0 / n)

    n_past = min(n_past, n)
    w = np.zeros(n)
    base_weight = 1.0 / n

    # Compute or use locked weights for past days
    if locked_weights is not None and len(locked_weights) >= n_past:
        w[:n_past] = locked_weights[:n_past]
    else:
        for i in range(n_past):
            signal = _compute_stable_signal(raw[: i + 1])[-1]
            w[i] = signal * base_weight

    # Scale past weights if they exceed budget
    past_sum = w[:n_past].sum()
    target_budget = n_past / n
    if past_sum > target_budget + 1e-10:
        w[:n_past] *= target_budget / past_sum

    # Future days (except last): uniform
    n_future = n - n_past
    if n_future > 1:
        w[n_past : n - 1] = base_weight

    # Last day absorbs remainder
    w[n - 1] = max(1.0 - w[: n - 1].sum(), 0)

    return w


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_dynamic_multiplier(model_proba: np.ndarray) -> np.ndarray:
    """Compute weight multiplier from GBM probability signal.

    Maps P(favourable) to buy multiplier via exponential scaling:
    - P > 0.5 -> multiplier > 1 (buy more)
    - P < 0.5 -> multiplier < 1 (buy less)
    - P = 0.5 -> multiplier = 1 (neutral / uniform)

    Args:
        model_proba: Predicted P(favourable) in [0, 1]

    Returns:
        Multipliers centered around 1.0
    """
    # Center around 0 and scale to [-1, 1]
    signal = (model_proba - 0.5) * 2

    # Apply strength and clip
    adjustment = signal * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -3, 3)

    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


def _clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.where(np.isfinite(arr), arr, 0)


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract model probability (0.5 = neutral for missing/placeholder dates)
    model_proba = _clean_array(df["model_proba"].values)
    model_proba = np.where(model_proba < 1e-6, 0.5, model_proba)

    # Compute dynamic weights from GBM probability
    dyn = compute_dynamic_multiplier(model_proba)
    raw = base * dyn

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability.

    Two modes:
    1. BACKTEST (locked_weights=None): Signal-based allocation
    2. PRODUCTION (locked_weights provided): DB-backed stability

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Investment window start
        end_date: Investment window end
        current_date: Current date (past/future boundary)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights summing to 1.0
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates (model_proba defaults to 0.5 = neutral)
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder_vals = {col: 0.0 for col in features_df.columns}
        placeholder_vals["model_proba"] = 0.5
        placeholder = pd.DataFrame(placeholder_vals, index=missing)
        features_df = pd.concat([features_df, placeholder]).sort_index()

    # Determine past/future split
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
