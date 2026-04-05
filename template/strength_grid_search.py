"""Grid search over DYNAMIC_STRENGTH using the real backtest pipeline.

Patches the module-level constant, recomputes features once (model predictions
are independent of strength), then re-runs SPD scoring at each strength value.

Usage:
    cd OMSA/template
    python strength_grid_search.py
"""

import logging
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.WARNING,  # suppress per-window logs
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Import template modules ---
try:
    from template.prelude_template import backtest_dynamic_dca, load_data
    from template import model_development_template as mdt
    from template.backtest_template import compute_weights_modal
except ImportError:
    from prelude_template import backtest_dynamic_dca, load_data
    import model_development_template as mdt
    from backtest_template import compute_weights_modal

# ============================================================================
# Configuration
# ============================================================================

STRENGTH_GRID = np.concatenate([
    np.arange(0.5, 2.0, 0.5),    # 0.5, 1.0, 1.5
    np.arange(2.0, 5.0, 1.0),    # 2.0, 3.0, 4.0
    np.arange(5.0, 8.0, 1.0),    # 5.0, 6.0, 7.0
    np.arange(8.0, 12.1, 2.0),   # 8.0, 10.0, 12.0
])

# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 90)
    print("DYNAMIC_STRENGTH Grid Search — Real Backtest Pipeline")
    print("=" * 90)

    # --- Step 1: Load data & precompute features ONCE ---
    # Features (model_proba) don't depend on DYNAMIC_STRENGTH, so we only
    # need to compute them once. Strength only affects compute_dynamic_multiplier().
    print("\n[1/3] Loading data...")
    btc_df = load_data()

    print("[2/3] Precomputing features (walk-forward GBM — takes a minute)...")
    logging.getLogger().setLevel(logging.INFO)
    features_df = mdt.precompute_features(btc_df)
    logging.getLogger().setLevel(logging.WARNING)

    # Store features globally for compute_weights_modal
    import backtest_template as bt
    bt._FEATURES_DF = features_df

    print(f"[3/3] Running backtest at {len(STRENGTH_GRID)} strength values...\n")

    # --- Step 2: Grid search ---
    results = []
    print(f"{'Strength':>10} {'Win Rate%':>10} {'ExpDecay%':>10} {'Score%':>10} {'Time(s)':>8}")
    print("─" * 55)

    for s in STRENGTH_GRID:
        # Patch the module-level constant
        mdt.DYNAMIC_STRENGTH = s

        t0 = time.time()
        df_spd, exp_decay_pct = backtest_dynamic_dca(
            btc_df,
            compute_weights_modal,
            features_df=features_df,
            strategy_label=f"s={s:.1f}",
        )
        elapsed = time.time() - t0

        win_rate = (
            df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]
        ).mean() * 100
        score = 0.5 * win_rate + 0.5 * exp_decay_pct

        results.append({
            "strength": s,
            "win_rate": win_rate,
            "exp_decay_percentile": exp_decay_pct,
            "score": score,
            "n_windows": len(df_spd),
            "mean_excess": (
                df_spd["dynamic_percentile"] - df_spd["uniform_percentile"]
            ).mean(),
        })

        print(f"{s:>10.1f} {win_rate:>10.2f} {exp_decay_pct:>10.2f} {score:>10.2f} {elapsed:>8.1f}")
        sys.stdout.flush()

    # --- Step 3: Summary ---
    res_df = pd.DataFrame(results).sort_values("score", ascending=False)
    best = res_df.iloc[0]

    print("\n" + "=" * 90)
    print("RESULTS — sorted by score")
    print("=" * 90)
    print(f"\n{'Strength':>10} {'Win%':>8} {'ExpDecay%':>10} {'Score%':>8} {'MeanExcess%':>12}")
    print("─" * 55)
    for _, row in res_df.iterrows():
        marker = " ◀ BEST" if row["strength"] == best["strength"] else ""
        print(
            f"{row['strength']:>10.1f} {row['win_rate']:>8.2f} "
            f"{row['exp_decay_percentile']:>10.2f} {row['score']:>8.2f} "
            f"{row['mean_excess']:>+12.3f}{marker}"
        )

    print(f"\n✓ Optimal DYNAMIC_STRENGTH = {best['strength']:.1f}")
    print(f"  Score        = {best['score']:.2f}%")
    print(f"  Win rate     = {best['win_rate']:.2f}%")
    print(f"  Exp-decay    = {best['exp_decay_percentile']:.2f}%")
    print(f"  Mean excess  = {best['mean_excess']:+.3f}%")

    # Save results
    out_path = "strength_grid_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
