"""Filtering and referencing helpers."""

from __future__ import annotations
from typing import Iterable, Optional
import mne
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def apply_bandpass(raw: mne.io.Raw, l_freq: float = 0.5, h_freq: float = 45.0, picks: Optional[Iterable[str]] = None, fir_design: str = "firwin") -> mne.io.Raw:
    """Apply zero-phase bandpass filter to Raw (in-place).

    Args:
        raw: mne Raw object.
        l_freq: low cutoff (Hz).
        h_freq: high cutoff (Hz).
        picks: channels to filter.
        fir_design: FIR design string.

    Returns:
        The same raw object filtered in-place.
    """
    picks_final = picks or mne.pick_types(raw.info, eeg=True)
    logger.info("Applying bandpass %.2f-%.2f Hz", l_freq, h_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks_final, fir_design=fir_design, verbose=False)
    return raw


def apply_notch(raw: mne.io.Raw, freqs: Iterable[float] = (50.0,), picks: Optional[Iterable[str]] = None) -> mne.io.Raw:
    """Apply notch filter to remove mains interference."""
    picks_final = picks or mne.pick_types(raw.info, eeg=True)
    freqs_list = list(freqs)
    logger.info("Applying notch filter at %s Hz", freqs_list)
    raw.notch_filter(freqs=freqs_list, picks=picks_final, verbose=False)
    return raw


def set_reference(raw: mne.io.Raw, reference: Optional[str] = "average") -> mne.io.Raw:
    """Set common average reference (or specific channel)."""
    if reference is None:
        logger.info("No reference change requested")
        return raw
    if reference == "average":
        logger.info("Setting common average reference")
        raw.set_eeg_reference("average", projection=False)
    else:
        if reference not in raw.ch_names:
            raise ValueError(f"Reference {reference} not found in raw.ch_names")
        logger.info("Setting reference to %s", reference)
        raw.set_eeg_reference(reference, projection=False)
    return raw
