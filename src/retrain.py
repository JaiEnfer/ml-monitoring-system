from __future__ import annotations

import threading

from src.train import train

_retrain_lock = threading.Lock()


def retrain_safely() -> bool:
    """
    Returns True if retrain ran, False if it was already running.
    """
    acquired = _retrain_lock.acquire(blocking=False)
    if not acquired:
        return False
    try:
        train()
        return True
    finally:
        _retrain_lock.release()
