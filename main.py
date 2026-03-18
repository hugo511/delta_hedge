# from __future__ import  annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from core.backtest_runner import DeltaHedgeRunner
from core.config_shema import ExperimentConfig, load_config
from utils.logger import log_info
from utils.plot_utils import _build_curve_data, _compute_metrics, _plot_curves, _build_multi_freq_wide_data, _plot_multi_freq_curves_wide


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


def run_one_config(cfg: ExperimentConfig, run_dir: Path) -> None:
    freq = str(cfg.hedge.frequency)
    freq_dir = run_dir / f"freq_{freq}"
    freq_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"开始回测频率: {freq}")

    runner = DeltaHedgeRunner(cfg_exp=cfg, strategy_name=f"delta_hedge_{freq}")
    backtest_result = runner.run()
    broker_record = runner.broker_record
    daily_decision = runner.daily_decision
    curve_data = _build_curve_data(backtest_result)
    metrics = _compute_metrics(curve_data, freq=freq)

    backtest_csv = freq_dir / "backtest_detail.csv"
    broker_record_csv = freq_dir / "broker_record.csv"
    daily_decision_csv = freq_dir / "daily_decision.csv"
    curve_csv = freq_dir / "curve_data.csv"
    metrics_csv = freq_dir / "performance_metrics.csv"
    curves_png = freq_dir / "performance_curves.png"

    backtest_result.write_csv(backtest_csv)
    broker_record.write_csv(broker_record_csv)
    daily_decision.write_csv(daily_decision_csv)
    curve_data.write_csv(curve_csv)
    metrics.write_csv(metrics_csv)
    # _plot_curves(curve_data, output_path=curves_png, freq=freq)
    _plot_curves(curve_data, output_path=curves_png, freq=freq, plot_every_n_trading_days=2, sampling_mode="avg")

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

    # plot
    multi_freq_wide_df = _build_multi_freq_wide_data(run_dir=run_dir)
    if not multi_freq_wide_df.is_empty():
        _plot_multi_freq_curves_wide(
            multi_freq_wide_df=multi_freq_wide_df,
            output_path=run_dir / "performance_curves_multi_freq.png",
            plot_every_n_trading_days=2,
            sampling_mode="avg",
        )
        # multi_freq_wide_df.write_csv(run_dir / "multi_freq_curve.csv")
        # log_info(f"已保存多频率宽表: {run_dir / 'multi_freq_curve.csv'}")
    else:
        log_info("多频率宽表为空，跳过合并绘图")
    log_info(f"全部回测结束，输出目录: {run_dir}")



if __name__ == "__main__":
    main()