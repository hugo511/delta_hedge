from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from utils.logger import log_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FutureConfig:
    name: str
    exchange: str
    fut_code: str
    update_basic: bool


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date


@dataclass
class HedgeConfig:
    # frequency is single value after expansion in _build_config
    frequency: str
    contract_selection_mode: str
    straddle_size: int
    roll_days_before_maturity: int
    use_contract_unit: bool


@dataclass
class ExperimentConfig:
    backtest: BacktestConfig
    future: FutureConfig
    hedge: HedgeConfig


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid date format for '{field_name}': {value!r}. "
                "Expected YYYY-MM-DD."
            ) from exc
    raise TypeError(f"Unsupported date type for '{field_name}': {type(value)!r}")


def _build_config(raw: dict[str, Any]) -> list[ExperimentConfig]:
    backtest_raw = raw["backtest"]
    # Compat: allow old key "option", prefer new key "future"
    future_raw = raw.get("future") or raw["option"]
    hedge_raw = raw["hedge"]

    backtest_config = BacktestConfig(
        start_date=_parse_date(backtest_raw["start_date"], "backtest.start_date"),
        end_date=_parse_date(backtest_raw["end_date"], "backtest.end_date"),
    )
    future_config = FutureConfig(
        name=future_raw["name"],
        exchange=future_raw["exchange"],
        fut_code=future_raw.get("fut_code"),
        update_basic=bool(future_raw["update_basic"]),
    )
    freq_raw = hedge_raw["frequency"]
    if isinstance(freq_raw, list):
        freq_list = [str(x) for x in freq_raw]
    else:
        freq_list = [str(freq_raw)]

    cfg_list: list[ExperimentConfig] = []
    for freq in freq_list:
        hedge_config = HedgeConfig(
            frequency=freq,  # 单个回测配置只保留一个频率
            contract_selection_mode=hedge_raw["contract_selection_mode"],
            straddle_size=int(hedge_raw["straddle_size"]),
            roll_days_before_maturity=int(hedge_raw["roll_days_before_maturity"]),
            use_contract_unit=bool(hedge_raw["use_contract_unit"]),
        )
        cfg_list.append(
            ExperimentConfig(
                backtest=backtest_config,
                future=future_config,
                hedge=hedge_config,
            )
        )
    return cfg_list


def load_confg(path: str | Path) -> list[ExperimentConfig]:
    """Load experiment config from a yaml file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError("Config file must define a top-level mapping.")

    return _build_config(raw)


def load_config(path: str | Path) -> list[ExperimentConfig]:
    """Alias for load_confg."""
    return load_confg(path)


def demo_print_config(path: str | Path = "config/shfe_ag_demo.yaml") -> None:
    cfg_list = load_confg(path)
    for i, cfg in enumerate(cfg_list):
        log_info(f"=== Config #{i+1} ===")
        log_info("=== BacktestConfig ===")
        log_info(cfg.backtest)
        log_info("=== FutureConfig ===")
        log_info(cfg.future)
        log_info("=== HedgeConfig ===")
        log_info(cfg.hedge)


if __name__ == "__main__":
    demo_print_config()
