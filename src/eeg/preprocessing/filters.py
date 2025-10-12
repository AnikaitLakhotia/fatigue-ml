"""Filtering and referencing utilities used by the preprocessing stage.

This module exposes three small, well-documented helpers:
  - apply_bandpass(raw, l_freq, h_freq, picks=None, fir_design="firwin")
  - apply_notch(raw, freqs, picks=None)
  - set_reference(raw, reference="average")

They perform in-place operations on an MNE Raw object and return it for
functional-style chaining. Each helper logs its activity for traceability.
"""

from __future__ import annotations
from typing import Iterable, Optional
import mne

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def apply_bandpass(
    raw: mne.io.Raw,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
    picks: Optional[Iterable[str]] = None,
    fir_design: str = "firwin",
) -> mne.io.Raw:
    """Apply a zero-phase bandpass filter to the Raw object.

    Modifies and returns `raw`.

    Args:
        raw: mne.io.Raw instance (preloaded or not).
        l_freq: low cutoff frequency (Hz), pass None to disable.
        h_freq: high cutoff frequency (Hz), pass None to disable.
        picks: channel selection compatible with MNE (None = all EEG).
        fir_design: FIR design to pass to MNE filter call.
    """
    logger.info("Applying bandpass %.2f–%.2f Hz", float(l_freq) if l_freq else 0.0, float(h_freq) if h_freq else 0.0)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, fir_design=fir_design, verbose=False)
    return raw


def apply_notch(raw: mne.io.Raw, freqs: Iterable[float], picks: Optional[Iterable[str]] = None) -> mne.io.Raw:
    """Apply notch filters in-place at the provided frequencies.

    Modifies and returns `raw`.

    Args:
        raw: mne.io.Raw instance
        freqs: iterable of notch frequencies (e.g., [50.0] or [50.0, 100.0])
        picks: channel selection (None => all EEG)
    """
    freqs_list = list(freqs)
    logger.info("Applying notch filter(s) at: %s", freqs_list)
    raw.notch_filter(freqs=freqs_list, picks=picks, verbose=False)
    return raw


def set_reference(raw: mne.io.Raw, reference: Optional[str] = "average") -> mne.io.Raw:
    """Set EEG reference for the Raw object.

    Modifies and returns `raw`.

    Args:
        raw: mne.io.Raw instance
        reference: 'average' for common-average reference or a channel name (string).
    """
    if reference is None or reference == "average":
        logger.info("Setting common average reference")
        raw.set_eeg_reference("average", projection=False)
    else:
        logger.info("Setting reference to channel: %s", reference)
        raw.set_eeg_reference(ref_channels=[reference], projection=False)
    return raw
