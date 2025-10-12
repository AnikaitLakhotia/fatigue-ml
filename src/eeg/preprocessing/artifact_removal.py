"""
artifact_removal.py

EEG artifact detection and cleaning utilities.
Handles channel rejection, ICA-based artifact removal, and bad-segment marking.
"""

from __future__ import annotations
import mne
import numpy as np
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def find_bad_channels_by_stats(raw: mne.io.BaseRaw, z_thresh: float = 3.0) -> list[str]:
    """
    Identify bad EEG channels based on standard deviation statistics.

    Args:
        raw: MNE Raw object.
        z_thresh: Z-score threshold for identifying outlier channels.

    Returns:
        List of bad channel names exceeding the threshold.
    """
    data = raw.get_data()
    ch_stds = np.std(data, axis=1)
    zscores = (ch_stds - np.mean(ch_stds)) / np.std(ch_stds)
    bads = [raw.ch_names[i] for i, z in enumerate(zscores) if abs(z) > z_thresh]

    if bads:
        logger.info(f"Flagged bad channels: {bads}")
    return bads


def run_ica_and_remove(
    raw: mne.io.BaseRaw,
    n_components: int = 20,
    random_state: int | None = 42,
    reject_eyeblinks: bool = True,
) -> mne.io.BaseRaw:
    """
    Apply ICA decomposition to remove ocular and muscle artifacts.

    Caps n_components to the number of EEG channels if necessary.
    Gracefully handles datasets without EOG channels.

    Args:
        raw: MNE Raw object.
        n_components: Target number of ICA components (auto-capped).
        random_state: Seed for reproducibility.
        reject_eyeblinks: Whether to automatically remove EOG-like components.

    Returns:
        Cleaned MNE Raw object.
    """
    n_components = min(n_components, len(raw.ch_names))
    logger.info(f"Running ICA for artifact removal (n_components={n_components})...")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw)

    if reject_eyeblinks:
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
            if eog_inds:
                ica.exclude = eog_inds
                logger.info(f"Excluded {len(eog_inds)} EOG-related components.")
            else:
                logger.info("No EOG-related components detected.")
        except RuntimeError:
            logger.warning("No EOG channel(s) found — skipping EOG artifact rejection.")

    raw_clean = ica.apply(raw.copy())
    logger.info("ICA artifact removal complete.")
    return raw_clean
