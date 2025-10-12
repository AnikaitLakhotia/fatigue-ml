"""Montage and channel geometry helpers.

Functions:
  - load_standard_montage(name)
  - apply_montage_to_raw(raw, montage_name)
  - get_channel_positions(raw)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import mne

from .logger import get_logger

logger = get_logger(__name__)


def load_standard_montage(name: str = "standard_1020") -> mne.channels.DigMontage:
    """
    Load a standard montage recognized by MNE.

    Args:
        name: Montage name (default 'standard_1020').

    Returns:
        mne.channels.DigMontage object.

    Raises:
        Exception: If montage cannot be created.
    """
    try:
        mont = mne.channels.make_standard_montage(name)
        logger.info("Loaded montage: %s", name)
        return mont
    except Exception:
        logger.exception("Failed to load montage %s", name)
        raise


def apply_montage_to_raw(raw: mne.io.BaseRaw, montage_name: str = "standard_1020") -> mne.io.BaseRaw:
    """
    Attach a montage to an MNE Raw object in-place.

    Args:
        raw: MNE Raw instance.
        montage_name: Name of the standard montage to attach.

    Returns:
        The same Raw object with montage set.
    """
    try:
        mont = load_standard_montage(montage_name)
        raw.set_montage(mont)
        logger.info("Applied montage %s to Raw", montage_name)
    except Exception:
        logger.exception("Failed to apply montage; returning original raw")
    return raw


def get_channel_positions(raw: mne.io.BaseRaw) -> Dict[str, Tuple[float, float, float]]:
    """
    Extract channel 3D positions from an MNE Raw object if available.

    Args:
        raw: MNE Raw instance.

    Returns:
        Mapping of channel name -> (x, y, z). Empty dict if none found.
    """
    try:
        pos_map = {}
        mont = raw.get_montage()
        if mont is not None:
            ch_pos = mont.get_positions().get("ch_pos", {})
            for ch, pos in ch_pos.items():
                pos_map[ch] = (float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            # fall back to raw.info entries
            for ch in raw.info.get("chs", []):
                name = ch.get("ch_name")
                loc = ch.get("loc")
                if loc and len(loc) >= 3:
                    pos_map[name] = (float(loc[0]), float(loc[1]), float(loc[2]))
        logger.info("Extracted positions for %d channels", len(pos_map))
        return pos_map
    except Exception:
        logger.exception("Failed to extract channel positions; returning empty dict")
        return {}
