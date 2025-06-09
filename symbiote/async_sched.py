"""Async micro‑tick scheduler."""


def should_update(tick: int, x: int, y: int, period: int = 8) -> bool:
    """
    _summary_

    _extended_summary_

    Args:
        tick (int): _description_
        x (int): _description_
        y (int): _description_
        period (int, optional): _description_. Defaults to 8.

    Returns:
        bool: _description_
    """
    return (tick + x + y) % period == 0
