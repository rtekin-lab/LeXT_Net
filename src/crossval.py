# -*- coding: utf-8 -*-
"""
crossval.py  —  Sequential 5-fold cross-validation
====================================================
Implements the patient-wise sequential split used in the paper.

Protocol (Section 3.1)
----------------------
- The dataset is divided into 5 consecutive, non-overlapping 20% blocks.
- In fold k, the k-th block is the test set; the remaining 80% is training.
- Because splits are patient-wise (PTB-XL patient IDs), no patient appears
  in both training and test within a fold.
- No validation set is carved out during evaluation; the loss weights were
  tuned in a preliminary grid search and then fixed.

Usage
-----
    from src.crossval import sequential_patient_folds

    folds = sequential_patient_folds(rec_ids, patient_ids, n_folds=5)
    for fold, (train_idx, test_idx) in enumerate(folds):
        x_train, x_test = signals[train_idx], signals[test_idx]
        ...
"""

import numpy as np
from typing import List, Tuple, Optional


def sequential_patient_folds(
    rec_ids: List[str],
    patient_ids: Optional[List[str]] = None,
    n_folds: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate sequential, patient-stratified cross-validation folds.

    For PTB-XL, `patient_ids` prevents the same patient from appearing in
    both the training and test partitions of a fold.  For CPSC2018 and
    Chapman (where a single CSV = one recording per patient), passing
    ``patient_ids=None`` is equivalent to per-recording stratification.

    Parameters
    ----------
    rec_ids    : list of N recording identifiers (in dataset order).
    patient_ids: list of N patient identifiers.  When provided, the split
                 boundary is adjusted so that all recordings of a patient
                 land in the same partition.  If None, boundaries are set
                 purely by index.
    n_folds    : number of folds (default 5 → each test block ≈ 20 %).

    Returns
    -------
    folds : list of (train_indices, test_indices) tuples.
            Indices are into the original ``rec_ids`` list.

    Notes
    -----
    When patient_ids are provided the actual test-set size per fold may
    differ slightly from 20 % because the boundary is snapped to the
    nearest patient boundary.  This is the expected behaviour and matches
    the protocol described in Section 3.1 of the paper.
    """
    n = len(rec_ids)
    indices = np.arange(n)

    if patient_ids is None:
        # Pure sequential split by index
        folds = []
        fold_size = n // n_folds
        for k in range(n_folds):
            start = k * fold_size
            end   = start + fold_size if k < n_folds - 1 else n
            test_idx  = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_idx, test_idx))
        return folds

    # Patient-aware split ─────────────────────────────────────────────────────
    patient_ids = np.asarray(patient_ids)
    unique_patients, first_occurrence = np.unique(patient_ids, return_index=True)
    # Sort patients by their first occurrence to maintain sequential order
    order = np.argsort(first_occurrence)
    unique_patients = unique_patients[order]
    n_patients = len(unique_patients)

    # Split patients into n_folds consecutive blocks
    pat_fold_size = n_patients // n_folds
    folds = []
    for k in range(n_folds):
        pat_start = k * pat_fold_size
        pat_end   = pat_start + pat_fold_size if k < n_folds - 1 else n_patients
        test_patients = set(unique_patients[pat_start:pat_end])

        test_mask  = np.array([pid in test_patients for pid in patient_ids])
        test_idx   = indices[test_mask]
        train_idx  = indices[~test_mask]
        folds.append((train_idx, test_idx))

    return folds


def get_fold(
    signals: np.ndarray,
    targets: np.ndarray,
    fold: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: split arrays by a pre-computed fold tuple.

    Parameters
    ----------
    signals : (N, T, C_in)
    targets : (N, T, 12)
    fold    : (train_indices, test_indices)

    Returns
    -------
    x_train, x_test, y_train, y_test
    """
    train_idx, test_idx = fold
    return (
        signals[train_idx], signals[test_idx],
        targets[train_idx], targets[test_idx],
    )
