# Scoring logic for stock selection

import pandas as pd, numpy as np

def score_stocks(bhav: pd.DataFrame, pre: dict) -> pd.DataFrame:
    # liquidity filter: ≥₹50 Cr turnover
    bhav["turnover"] = bhav["ClsPric"] * bhav["TtlTradgVol"]
    liquid = bhav[bhav["turnover"] >= 50e7]

    # gap calculation
    liquid["gap_pct"] = (liquid["TckrSymb"].map(pre) - liquid["ClsPric"]) / liquid["ClsPric"]

    # keep mild gaps (0.3 % – 1 %)
    within_gap = liquid[liquid["gap_pct"].abs().between(0.003, 0.01)]

    # simple score: bigger volume + aligned gap ⇒ higher rank
    within_gap["score"] = np.abs(within_gap["gap_pct"]) * within_gap["turnover"]
    picks = within_gap.nlargest(5, "score")[["TckrSymb", "ClsPric", "gap_pct"]]

    # attach entry/stop/target
    picks["entry"]   = picks["ClsPric"] * (1 + picks["gap_pct"])
    picks["stop"]    = picks["entry"] * 0.995    # 0.5 % SL
    picks["target"]  = picks["entry"] * 1.010    # 1 % target ⇒ ≈₹1000 on ₹10 k
    return picks.reset_index(drop=True)
