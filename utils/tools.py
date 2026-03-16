from datetime import date


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _year_fraction(expiry_date: date, trade_date: date) -> float:
    return max((expiry_date - trade_date).days, 1) / 365.0