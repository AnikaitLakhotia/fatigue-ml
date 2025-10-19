"""ICA-based artifact removal and simple channel-statistics based bad-channel detection."""

from __future__ import annotations
from typing import List
import numpy as np
import mne

from ..utils.logger import get_logger

logger = get_logger(__name__)


def find_bad_channels_by_stats(raw: mne.io.BaseRaw, z_thresh: float = 3.0) -> List[str]:
    """
    Identify bad EEG channels based on standard deviation z-score.

    Args:
        raw: MNE Raw object.
        z_thresh: Z-score threshold; channels with |z| > z_thresh are flagged.

    Returns:
        List of bad channel names.
    """
    data = raw.get_data()
    ch_stds = np.std(data, axis=1)
    zscores = (ch_stds - np.mean(ch_stds)) / (np.std(ch_stds) + 1e-12)
    bads = [raw.ch_names[i] for i, z in enumerate(zscores) if abs(z) > z_thresh]
    if bads:
        logger.info("Flagged bad channels: %s", bads)
    return bads


def run_ica_and_remove(
    raw: mne.io.BaseRaw,
    n_components: int = 20,
    random_state: int | None = 42,
    reject_eyeblinks: bool = True,
) -> mne.io.BaseRaw:
    """
    Run ICA on `raw` and remove artifact components.

    Args:
        raw: MNE Raw object.
        n_components: Number of ICA components; capped to n_channels.
        random_state: Seed for reproducibility.
        reject_eyeblinks: If True, automatically find and exclude EOG components.

    Returns:
        A cleaned MNE Raw object (a copy with ICA applied).
    """
    n_components = min(n_components, len(raw.ch_names))
    logger.info("Running ICA with n_components=%d", n_components)
    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=random_state, max_iter="auto"
    )
    ica.fit(raw)
    if reject_eyeblinks:
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
            if eog_inds:
                ica.exclude = eog_inds
                logger.info("Excluded EOG components: %s", eog_inds)
            else:
                logger.info("No EOG-related components detected.")
        except RuntimeError:
            logger.warning("No EOG channel(s) found — skipping EOG artifact rejection.")
    raw_clean = ica.apply(raw.copy())
    logger.info("ICA artifact removal complete.")
    return raw_clean
