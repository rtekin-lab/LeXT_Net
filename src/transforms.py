# -*- coding: utf-8 -*-
"""
transforms.py  —  3-lead input generation strategies
=====================================================
Implements the four lead-reduction strategies evaluated in the paper:

  EintGBeqV1 (EGV1) : {I, II, V1}
  EintGBeqV2 (EGV2) : {I, II, V2}
  EintGBeqV3 (EGV3) : {I, II, V3}
  Quasi      (QSI)  : {V6, II, -0.5·V2}   (quasi-orthogonal, Frank-inspired)

All functions accept a signals array of shape (N, T, 12) where the
12-lead order is [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6],
and return a 3-lead array of shape (N, T, 3).
"""

import numpy as np
from typing import Literal

# Lead indices in standard order [I, II, III, aVR, aVL, aVF, V1..V6]
I    = 0
II   = 1
III  = 2
aVR  = 3
aVL  = 4
aVF  = 5
V1   = 6
V2   = 7
V3   = 8
V4   = 9
V5   = 10
V6   = 11

STRATEGY_NAMES = ("EGV1", "EGV2", "EGV3", "QSI")

# mInd lead indices per strategy (independent target leads, excluding input
# leads and Einthoven-Goldberger derivable limb leads)
MIND_LEADS = {
    "EGV1": [V2, V3, V4, V5, V6],          # V1 is input; III,aVR,aVL,aVF derivable
    "EGV2": [V1, V3, V4, V5, V6],
    "EGV3": [V1, V2, V4, V5, V6],
    "QSI":  [I, III, aVR, aVL, aVF, V1, V3, V4, V5],  # II,V2,V6 are inputs
}


def apply_strategy(
    signals: np.ndarray,
    strategy: Literal["EGV1", "EGV2", "EGV3", "QSI"],
) -> np.ndarray:
    """
    Reduce (N, T, 12) → (N, T, 3) using the selected input strategy.

    Parameters
    ----------
    signals  : float32 array (N, T, 12)
    strategy : one of "EGV1", "EGV2", "EGV3", "QSI"

    Returns
    -------
    z : float32 array (N, T, 3)
    """
    s = signals  # alias
    if strategy == "EGV1":
        # Ch1=I, Ch2=II, Ch3=V1
        return np.stack([s[..., I], s[..., II], s[..., V1]], axis=-1).astype(np.float32)

    if strategy == "EGV2":
        # Ch1=I, Ch2=II, Ch3=V2
        return np.stack([s[..., I], s[..., II], s[..., V2]], axis=-1).astype(np.float32)

    if strategy == "EGV3":
        # Ch1=I, Ch2=II, Ch3=V3
        return np.stack([s[..., I], s[..., II], s[..., V3]], axis=-1).astype(np.float32)

    if strategy == "QSI":
        # Ch1=V6(X), Ch2=II(Y), Ch3=-0.5·V2(Z)
        return np.stack(
            [s[..., V6], s[..., II], -0.5 * s[..., V2]], axis=-1
        ).astype(np.float32)

    raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGY_NAMES}.")


def get_mind_indices(strategy: str) -> list[int]:
    """Return the mInd lead indices for a given strategy."""
    if strategy not in MIND_LEADS:
        raise ValueError(f"Unknown strategy '{strategy}'.")
    return MIND_LEADS[strategy]
