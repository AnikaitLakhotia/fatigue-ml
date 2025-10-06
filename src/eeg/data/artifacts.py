"""Artifact detection and ICA cleaning."""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import mne
import numpy as np
from mne.preprocessing import ICA
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def find_bad_channels_by_stats(raw: mne.io.Raw, flat_threshold: float = 1e-8, noisy_z: float = 6.0) -> List[str]:
    """Return channels that are flat or have extreme variance."""
    data = raw.get_data()
    stds = np.std(data, axis=1)
    median = float(np.median(stds))
    bads: List[str] = []
    for idx, s in enumerate(stds):
        if s < flat_threshold or s > noisy_z * median:
            bads.append(raw.ch_names[idx])
    if bads:
        logger.info("Detected bad channels: %s", bads)
    return bads


def run_ica_and_remove(
    raw: mne.io.Raw,
    n_components: Optional[int] = None,
    method: str = "fastica",
    random_state: int = 42,
    eog_channels: Optional[Iterable[str]] = None,
    reject_criteria: Optional[dict] = None,
    max_iter: int = 1000,
) -> Tuple[mne.io.Raw, ICA]:
    """Fit ICA and remove components correlated with EOG channels if available.

    Returns cleaned raw and fitted ICA object.
    """
    n_components_final = n_components if n_components is not None else min(max(1, raw.info["nchan"] - 1), 32)
    logger.info("Fitting ICA (method=%s, n_components=%s)", method, n_components_final)
    ica = ICA(n_components=n_components_final, method=method, random_state=int(random_state), max_iter=max_iter, verbose=False)
    ica.fit(raw, reject=reject_criteria)

    exclude_idx: List[int] = []
    if eog_channels:
        for ch in eog_channels:
            if ch in raw.ch_names:
                try:
                    inds, _ = ica.find_bads_eog(raw, ch_name=ch)
                    exclude_idx.extend(inds)
                except Exception:
                    logger.debug("No EOG-like ICs found for %s", ch)

    exclude_idx = sorted(set(exclude_idx))
    ica.exclude = exclude_idx
    logger.info("Excluding ICA components: %s", exclude_idx)
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean, ica
