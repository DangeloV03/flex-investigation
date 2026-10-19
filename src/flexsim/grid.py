"""Parameter grids."""


def frange(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive float range [lo, hi] at fixed step (rounded to 6 decimals)."""
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 6) for i in range(n)]
