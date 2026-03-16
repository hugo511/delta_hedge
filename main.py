# from __future__ import  annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from core.backtest_runner_self_prod import DeltaHedgeRunner
from core.config_shema import ExperimentConfig, load_config
from utils.logger import log_info


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs_self"


def _serialize(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _build_curve_data(backtest_result: pl.DataFrame) -> pl.DataFrame:
    if backtest_result.is_empty():
        return pl.DataFrame()
    required_cols = {"bar_ts", "trade_date", "nav", "future_notional", "option_notional"}
    if not required_cols.issubset(set(backtest_result.columns)):
        log_info(f"backtest_result 缺少必要列: {required_cols}")
        return pl.DataFrame()

    return (
        backtest_result
        .sort("bar_ts")
        .with_columns(
            (
                pl.col("future_notional").abs() + pl.col("option_notional").abs()
            ).alias("capital_used"),
        )
        .with_columns(
            (
                pl.col("capital_used") / pl.col("nav").clip(lower_bound=1e-9)
            ).alias("capital_usage"),
            (
                pl.col("nav") / pl.col("nav").first().clip(lower_bound=1e-9) - 1.0
            ).alias("return_rate"),
        )
        .select(
            [
                "bar_ts",
                "trade_date",
                "nav",
                "capital_used",
                "capital_usage",
                "return_rate",
            ]
        )
    )


def _plot_curves(curve_data: pl.DataFrame, output_path: Path, freq: str) -> None:
    if curve_data.is_empty():
        log_info("curve_data 为空，跳过绘图")
        return

    pdf = curve_data.select(["bar_ts", "nav", "return_rate"]).to_pandas()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(pdf["bar_ts"], pdf["nav"], lw=1.8, color="#1f77b4")
    axes[0].set_title(f"NAV (freq={freq})")
    axes[0].grid(alpha=0.25)

    axes[1].plot(pdf["bar_ts"], pdf["return_rate"], lw=1.6, color="#2ca02c")
    axes[1].set_title(f"Return Rate (freq={freq})")
    axes[1].set_xlabel("Bar Time")
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="x", rotation=30)

    # 分钟级频率显示到时分，其他频率显示到日期
    is_minute_freq = "min" in freq.lower()
    x_formatter = (
        mdates.DateFormatter("%Y-%m-%d %H:%M")
        if is_minute_freq
        else mdates.DateFormatter("%Y-%m-%d")
    )
    for ax in axes: 
        ax.xaxis.set_major_formatter(x_formatter)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log_info(f"已保存走势图: {output_path}")


def _compute_metrics(curve_data: pl.DataFrame, freq: str) -> pl.DataFrame:
    if curve_data.is_empty():
        return pl.DataFrame(
            [
                {
                    "frequency": freq,
                    "start_date": None,
                    "end_date": None,
                    "trading_days": 0,
                    "initial_nav": None,
                    "final_nav": None,
                    "total_return": None,
                    "annual_return": None,
                    "max_drawdown": None,
                    "max_drawdown_start_date": None,
                    "max_drawdown_date": None,
                    "max_drawdown_recovery_date": None,
                    "max_drawdown_duration_days": None,
                    "sharpe": None,
                    "calmar": None,
                }
            ]
        )

    daily_nav_df = (
        curve_data.sort("bar_ts")
        .group_by("trade_date")
        .agg(pl.col("nav").last().alias("nav"))
        .sort("trade_date")
    )
    if daily_nav_df.is_empty():
        return pl.DataFrame()

    dates = daily_nav_df["trade_date"].to_list()
    navs = [float(v) for v in daily_nav_df["nav"].to_list()]
    trading_days = len(navs)

    initial_nav = navs[0]
    final_nav = navs[-1]
    total_return = (final_nav / max(initial_nav, 1e-9)) - 1.0
    annual_return = None
    if trading_days > 1 and initial_nav > 0:
        annual_return = (final_nav / initial_nav) ** (252.0 / (trading_days - 1)) - 1.0

    running_max: list[float] = []
    peak_idx_for_each_bar: list[int] = []
    peak_value = -float("inf")
    peak_idx = 0
    for i, nav in enumerate(navs):
        if nav >= peak_value:
            peak_value = nav
            peak_idx = i
        running_max.append(peak_value)
        peak_idx_for_each_bar.append(peak_idx)

    drawdowns = [
        (nav / max(running_max[i], 1e-9)) - 1.0
        for i, nav in enumerate(navs)
    ]
    trough_idx = int(np.argmin(np.array(drawdowns)))
    max_drawdown = float(drawdowns[trough_idx])
    drawdown_start_idx = peak_idx_for_each_bar[trough_idx]
    drawdown_start_date = dates[drawdown_start_idx]
    drawdown_date = dates[trough_idx]

    recovery_idx: int | None = None
    peak_before_drawdown = running_max[trough_idx]
    for i in range(trough_idx + 1, len(navs)):
        if navs[i] >= peak_before_drawdown:
            recovery_idx = i
            break
    recovery_date = dates[recovery_idx] if recovery_idx is not None else None
    duration_days = (
        int(recovery_idx - drawdown_start_idx)
        if recovery_idx is not None
        else int((len(navs) - 1) - drawdown_start_idx)
    )

    daily_returns = []
    for i in range(1, len(navs)):
        prev_nav = max(navs[i - 1], 1e-9)
        daily_returns.append(navs[i] / prev_nav - 1.0)

    sharpe = None
    if len(daily_returns) >= 2:
        ret_arr = np.array(daily_returns, dtype=float)
        ret_std = float(np.std(ret_arr, ddof=1))
        if ret_std > 0:
            sharpe = float(np.sqrt(252.0) * np.mean(ret_arr) / ret_std)

    calmar = None
    if annual_return is not None and max_drawdown < 0:
        calmar = float(annual_return / abs(max_drawdown))

    return pl.DataFrame(
        [
            {
                "frequency": freq,
                "start_date": dates[0],
                "end_date": dates[-1],
                "trading_days": trading_days,
                "initial_nav": initial_nav,
                "final_nav": final_nav,
                "total_return": total_return,
                "annual_return": annual_return,
                "max_drawdown": max_drawdown,
                "max_drawdown_start_date": drawdown_start_date,
                "max_drawdown_date": drawdown_date,
                "max_drawdown_recovery_date": recovery_date,
                "max_drawdown_duration_days": duration_days,
                "sharpe": sharpe,
                "calmar": calmar,
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
    daily_decision = runner.daily_decision
    
    curve_data = _build_curve_data(backtest_result)
    metrics = _compute_metrics(curve_data, freq=freq)

    backtest_csv = freq_dir / "backtest_detail.csv"
    daily_decision_csv = freq_dir / "daily_decision.csv"
    curve_csv = freq_dir / "curve_data.csv"
    metrics_csv = freq_dir / "performance_metrics.csv"
    curves_png = freq_dir / "performance_curves.png"

    backtest_result.write_csv(backtest_csv)
    daily_decision.write_csv(daily_decision_csv)
    curve_data.write_csv(curve_csv)
    metrics.write_csv(metrics_csv)
    _plot_curves(curve_data, output_path=curves_png, freq=freq)

    log_info(
        f"频率 {freq} 输出完成: {backtest_csv.name}, {daily_decision_csv.name}, {curve_csv.name}, "
        f"{metrics_csv.name}, {curves_png.name}"
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