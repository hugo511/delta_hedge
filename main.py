from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from core.backtest_runner import DeltaHedgeRunner, plot_daily_pnl
from core.config_shema import ExperimentConfig, load_config
from utils.logger import log_info


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def _serialize(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _build_daily_position(backtest_result: pl.DataFrame) -> pl.DataFrame:
    if backtest_result.is_empty():
        return pl.DataFrame()
    return (
        backtest_result.sort("bar_ts")
        .group_by("trade_date")
        .agg(
            pl.col("future_ts_code").last().alias("future_ts_code"),
            pl.col("call_ts_code").last().alias("call_ts_code"),
            pl.col("put_ts_code").last().alias("put_ts_code"),
            pl.col("current_future_position").last().alias("future_position"),
            pl.col("option_position_call").last().alias("call_position"),
            pl.col("option_position_put").last().alias("put_position"),
            pl.col("pnl").sum().alias("daily_pnl"),
            pl.col("nav").last().alias("nav"),
        )
        .sort("trade_date")
    )


def _build_summary(daily_result: pl.DataFrame, backtest_result: pl.DataFrame, freq: str) -> pl.DataFrame:
    if daily_result.is_empty():
        return pl.DataFrame(
            [
                {
                    "frequency": freq,
                    "total_pnl": 0.0,
                    "final_nav": None,
                    "nav_ratio": None,
                    "strategy_return": None,
                    "capital_usage": None,
                    "total_fee": 0.0,
                }
            ]
        )
    final_nav_ratio = (
        float(daily_result["nav_ratio"].tail(1).item())
        if "nav_ratio" in daily_result.columns
        else None
    )
    final_strategy_return = (
        float(daily_result["strategy_return"].tail(1).item())
        if "strategy_return" in daily_result.columns
        else None
    )
    final_capital_usage = (
        float(daily_result["capital_usage"].tail(1).item())
        if "capital_usage" in daily_result.columns
        else None
    )
    return pl.DataFrame(
        [
            {
                "frequency": freq,
                "total_pnl": float(daily_result["daily_pnl"].sum()),
                "final_nav": float(daily_result["nav"].tail(1).item()),
                "nav_ratio": final_nav_ratio,
                "strategy_return": final_strategy_return,
                "capital_usage": final_capital_usage,
                "total_fee": float(daily_result["daily_fee"].sum()) if "daily_fee" in daily_result.columns else 0.0,
                "bars": int(backtest_result.height),
                "days": int(daily_result.height),
            }
        ]
    )


def run_one_config(cfg: ExperimentConfig, run_dir: Path) -> None:
    freq = str(cfg.hedge.frequency)
    freq_dir = run_dir / f"freq_{freq}"
    freq_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"开始回测频率: {freq}")

    runner = DeltaHedgeRunner(cfg_exp=cfg, strategy_name=f"delta_hedge_{freq}")
    backtest_result = runner.run()
    daily_result = runner.daily_result
    daily_position = _build_daily_position(backtest_result)
    summary = _build_summary(daily_result, backtest_result, freq=freq)

    backtest_csv = freq_dir / "backtest_detail.csv"
    daily_csv = freq_dir / "daily_result.csv"
    position_csv = freq_dir / "daily_position.csv"
    summary_csv = freq_dir / "summary.csv"
    pnl_png = freq_dir / "daily_pnl.png"
    strategy_curves_png = freq_dir / "strategy_curves.png"

    backtest_result.write_csv(backtest_csv)
    daily_result.write_csv(daily_csv)
    daily_position.write_csv(position_csv)
    summary.write_csv(summary_csv)
    plot_daily_pnl(daily_result, output_path=str(pnl_png), show=False)

    log_info(
        f"频率 {freq} 输出完成: {backtest_csv.name}, {daily_csv.name}, "
        f"{position_csv.name}, {summary_csv.name}, {pnl_png.name}, {strategy_curves_png.name}"
    )


def main(config_path: str = "config/shfe_ag_demo.yaml") -> None:
    cfg_list = load_config(config_path)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次运行使用的原始配置（保持与config_path一致）
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    full_cfg_dict = _serialize(raw_cfg)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(full_cfg_dict, f, ensure_ascii=False, indent=2)

    for cfg_one in cfg_list:
        run_one_config(cfg_one, run_dir=run_dir)

    log_info(f"全部回测结束，输出目录: {run_dir}")


if __name__ == "__main__":
    main()
