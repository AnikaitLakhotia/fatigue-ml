"""Filtering and referencing utilities used by the preprocessing stage.

Functions:
  - apply_bandpass
  - apply_notch
  - set_reference

Each function modifies the provided MNE Raw object in-place and returns it for
functional chaining. Logging included for traceability.
"""

from __future__ import annotations
from typing import Iterable, Optional
import mne

from ..utils.logger import get_logger

logger = get_logger(__name__)


def apply_bandpass(
    raw: mne.io.BaseRaw,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 45.0,
    picks: Optional[Iterable[str]] = None,
    fir_design: str = "firwin",
) -> mne.io.BaseRaw:
    """
    Apply a zero-phase bandpass filter to an MNE Raw object.

    Args:
        raw: MNE Raw instance.
        l_freq: Low cutoff in Hz (None disables).
        h_freq: High cutoff in Hz (None disables).
        picks: Channels to apply filter to (None -> all EEG).
        fir_design: FIR design argument for MNE's filter.

    Returns:
        The modified Raw object.
    """
    logger.info("Applying bandpass %.2f–%.2f Hz", float(l_freq) if l_freq else 0.0, float(h_freq) if h_freq else 0.0)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, fir_design=fir_design, verbose=False)
    return raw


def apply_notch(raw: mne.io.BaseRaw, freqs: Iterable[float], picks: Optional[Iterable[str]] = None) -> mne.io.BaseRaw:
    """
    Apply notch filters in-place at the provided frequencies.

    Args:
        raw: MNE Raw instance.
        freqs: Iterable of notch frequencies (e.g., [50.0]).
        picks: Channel selection (None => all EEG).

    Returns:
        The modified Raw object.
    """
    freqs_list = list(freqs)
    logger.info("Applying notch filter(s) at: %s", freqs_list)
    raw.notch_filter(freqs=freqs_list, picks=picks, verbose=False)
    return raw


def set_reference(raw: mne.io.BaseRaw, reference: Optional[str] = "average") -> mne.io.BaseRaw:
    """
    Set EEG reference for the Raw object.

    Args:
        raw: MNE Raw instance.
        reference: 'average' or channel name.

    Returns:
        The modified Raw object.
    """
    if reference is None or reference == "average":
        logger.info("Setting common average reference")
        raw.set_eeg_reference("average", projection=False)
    else:
        logger.info("Setting reference to channel: %s", reference)
        raw.set_eeg_reference(ref_channels=[reference], projection=False)
    return raw
