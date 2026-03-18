import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from utils.logger import log_info, log_warning
import numpy as np
import pandas as pd 
import matplotlib as mpl
from matplotlib.patches import Patch
import platform


def setup_chinese_font():
    """设置中文字体的辅助函数"""
    system = platform.system()
    
    if system == "Windows":
        # Windows 系统
        font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        font_names = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC', 'STHeiti']
    else:  # Linux
        font_names = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    # 查找可用的字体
    from matplotlib import font_manager
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    
    for font in font_names:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            log_info(f"使用字体: {font}")
            return
    
    # 如果都没有，使用默认字体但警告
    log_warning("警告: 未找到中文字体，中文可能显示为方框")
    



def _build_curve_data(backtest_result: pl.DataFrame) -> pl.DataFrame:
    if backtest_result.is_empty():
        return pl.DataFrame()
    # 统一使用新版回测输出口径：capital_used 代表资金占用（期货保证金）
    required_cols = {"bar_ts", "trade_date", "nav", "capital_used"}
    if not required_cols.issubset(set(backtest_result.columns)):
        log_info(f"backtest_result 缺少必要列: {required_cols}")
        return pl.DataFrame()

    return (
        backtest_result
        .sort("bar_ts")
        .with_columns(
            pl.col("margin_by_future").abs().alias("capital_used"),
            pl.col("future_close").alias("future_close"),
            (pl.col("iv_call_used") + pl.col("iv_put_used")).alias("iv_avg"),
            pl.col("pnl_future").cum_sum().alias("pnl_future"),
            pl.col("pnl_option").cum_sum().alias("pnl_option"),
            pl.col("pnl").cum_sum().alias("pnl_total"),
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
                "bar_ts", "trade_date", "nav", "capital_used", "capital_usage", "return_rate", "future_close", "iv_avg",
                "pnl_future", "pnl_option", "pnl_total"
            ]
            + (["name"] if "name" in backtest_result.columns else [])
        )
    )



def _plot_curves(
    curve_data: pl.DataFrame, 
    output_path: Path, 
    freq: str, 
    plot_every_n_trading_days: int = 1,
    sampling_mode: str = "last"
) -> None:
    """
    绘制净值曲线和收益率曲线
    
    Parameters:
    -----------
    curve_data : pl.DataFrame
        包含绘图数据的DataFrame
    output_path : Path
        输出图片路径
    freq : str
        数据频率（用于标题显示）
    plot_every_n_trading_days : int
        每隔多少个交易日画一个点，默认1表示每个交易日都画
        例如：plot_every_n_trading_days=5 表示每5个交易日画一个点
        无论原始数据是15min还是日频，都按交易日维度采样
    sampling_mode : str
        采样模式，可选：
        - "last": 取每个采样周期最后一个点的值（默认）
        - "avg": 取每个采样周期内所有bar的平均值
    """
    # 设置中文字体
    setup_chinese_font()
    
    if curve_data.is_empty():
        log_info("curve_data 为空，跳过绘图")
        return

    plot_data = curve_data
    if "name" in curve_data.columns:
        # close 记录对应平仓点，资金占用可能为0，默认不纳入曲线展示
        filtered = curve_data.filter(pl.col("name") != "close")
        if not filtered.is_empty():
            plot_data = filtered

    # 原始频率是 1d 还是 15min
    daily_plot_data = (
        plot_data
        .sort(["trade_date", "bar_ts"])
        .with_columns(pl.col("bar_ts").cast(pl.Datetime).alias("plot_ts"))
    ).sort("plot_ts")

    if daily_plot_data.is_empty():
        log_info("daily_plot_data 为空，跳过绘图")
        return

    # 转换为 pandas 并准备数据
    pdf = daily_plot_data.select(["plot_ts", "nav", "return_rate"]).to_pandas()
    pdf['plot_ts'] = pd.to_datetime(pdf['plot_ts'])
    pdf = pdf.sort_values('plot_ts').reset_index(drop=True)
    
    # ============== 按交易日维度进行降采样 ==============
    n_data_points = len(pdf)
    
    # 提取交易日日期（忽略时间）
    pdf['trade_date_only'] = pdf['plot_ts'].dt.date
    
    # 获取唯一的交易日列表
    unique_trading_days = sorted(pdf['trade_date_only'].unique())
    n_trading_days = len(unique_trading_days)
    
    log_info(f"原始数据共 {n_data_points} 个bar，涉及 {n_trading_days} 个交易日，采样模式: {sampling_mode}")
    
    # 按交易日分组，准备采样
    if plot_every_n_trading_days > 1:
        log_info(f"设置每隔 {plot_every_n_trading_days} 个交易日画一个点")
        
        # 将交易日分成连续的组，每组 plot_every_n_trading_days 个交易日
        trading_day_groups = []
        for i in range(0, len(unique_trading_days), plot_every_n_trading_days):
            group = unique_trading_days[i:i + plot_every_n_trading_days]
            trading_day_groups.append(group)
        
        # 对每个组进行采样
        sampled_data = []
        for group in trading_day_groups:
            # 获取该组内所有交易日的数据
            group_mask = pdf['trade_date_only'].isin(group)
            group_data = pdf[group_mask]
            
            if group_data.empty:
                continue
            
            if sampling_mode == "last":
                # 取组内最后一个数据点
                last_idx = group_data.index[-1]
                sampled_point = pdf.loc[last_idx:last_idx].copy()
                sampled_data.append(sampled_point)
            elif sampling_mode == "avg":
                # 取组内所有bar的平均值
                avg_point = pd.DataFrame({
                    'plot_ts': [pd.Timestamp(group[-1])],  # 用组最后一天作为时间戳
                    'nav': [group_data['nav'].mean()],
                    'return_rate': [group_data['return_rate'].mean()],
                    'trade_date_only': [group[-1]]
                })
                sampled_data.append(avg_point)
            else:
                raise ValueError(f"不支持的采样模式: {sampling_mode}")
        
        if sampled_data:
            pdf_plot = pd.concat(sampled_data, ignore_index=True)
        else:
            pdf_plot = pd.DataFrame()
        
        # 检查是否采样后点还是太多，如果是，再按max_points限制
        max_points = 2000
        if len(pdf_plot) > max_points:
            log_info(f"按交易日采样后仍有 {len(pdf_plot)} 个点，继续降采样至约 {max_points} 点")
            # 二次降采样（按时间均匀）
            sample_interval = len(pdf_plot) // max_points
            sampled_idx2 = pdf_plot.iloc[::sample_interval].index
            if sampled_idx2[-1] != pdf_plot.index[-1]:
                sampled_idx2 = sampled_idx2.append(pd.Index([pdf_plot.index[-1]]))
            pdf_plot = pdf_plot.loc[sampled_idx2].copy().reset_index(drop=True)
    else:
        # 如果 plot_every_n_trading_days = 1，检查是否需要按点数降采样
        max_points = 2000
        
        if n_data_points > max_points:
            log_info(f"数据点过多 ({n_data_points})，进行降采样至约 {max_points} 点")
            
            if sampling_mode == "last":
                # 按交易日采样：每个交易日取最后一个bar
                sampled_indices = []
                for trade_date in unique_trading_days:
                    day_data = pdf[pdf['trade_date_only'] == trade_date]
                    if not day_data.empty:
                        last_idx = day_data.index[-1]
                        sampled_indices.append(last_idx)
                
                pdf_sampled = pdf.loc[sampled_indices].copy().reset_index(drop=True)
            elif sampling_mode == "avg":
                # 按交易日采样：每个交易日取平均值
                avg_data = []
                for trade_date in unique_trading_days:
                    day_data = pdf[pdf['trade_date_only'] == trade_date]
                    if not day_data.empty:
                        avg_point = pd.DataFrame({
                            'plot_ts': [pd.Timestamp(trade_date)],
                            'nav': [day_data['nav'].mean()],
                            'return_rate': [day_data['return_rate'].mean()],
                            'trade_date_only': [trade_date]
                        })
                        avg_data.append(avg_point)
                
                if avg_data:
                    pdf_sampled = pd.concat(avg_data, ignore_index=True)
                else:
                    pdf_sampled = pd.DataFrame()
            else:
                raise ValueError(f"不支持的采样模式: {sampling_mode}")
            
            # 如果按交易日采样后还是太多，再按时间均匀降采样
            if len(pdf_sampled) > max_points:
                log_info(f"按交易日采样后仍有 {len(pdf_sampled)} 个点，继续降采样")
                sample_interval = len(pdf_sampled) // max_points
                sampled_idx2 = pdf_sampled.iloc[::sample_interval].index
                if sampled_idx2[-1] != pdf_sampled.index[-1]:
                    sampled_idx2 = sampled_idx2.append(pd.Index([pdf_sampled.index[-1]]))
                pdf_plot = pdf_sampled.loc[sampled_idx2].copy().reset_index(drop=True)
            else:
                pdf_plot = pdf_sampled
        else:
            pdf_plot = pdf.copy()
    
    if pdf_plot.empty:
        log_info("采样后数据为空，跳过绘图")
        return
    
    log_info(f"实际绘图点数: {len(pdf_plot)} (原始bar数: {n_data_points}, 原始交易日数: {n_trading_days}, 采样间隔: {plot_every_n_trading_days if plot_every_n_trading_days>1 else 'auto'}交易日, 模式: {sampling_mode})")
    
    # ============== 创建非线性X轴 ==============
    # 计算相邻交易日之间的实际天数
    pdf_plot['next_date'] = pdf_plot['plot_ts'].shift(-1)
    pdf_plot['days_to_next'] = (pdf_plot['next_date'] - pdf_plot['plot_ts']).dt.days.fillna(1)
    
    # 创建X轴位置序列
    x_positions = [0]
    for i in range(len(pdf_plot)-1):
        # 基础间距1，如果间隔>3天，额外增加间距
        gap = pdf_plot.iloc[i]['days_to_next']
        extra_space = max(0, (gap - 1) * 0.5)  # 每多一天增加0.5个单位
        x_positions.append(x_positions[-1] + 1 + extra_space)
    
    # ============== 创建图表 ==============
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
    # 绘制曲线（用更细的线）
    line_width = 1.2 if len(pdf_plot) > 500 else 1.5
    axes[0].plot(x_positions, pdf_plot["nav"], lw=line_width, color="#1f77b4")
    
    # 构建标题
    mode_text = "取最后一点" if sampling_mode == "last" else "取平均值"
    plot_info = f"共{n_trading_days}个交易日, {n_data_points}个bar"
    if plot_every_n_trading_days > 1:
        plot_info += f", 每{plot_every_n_trading_days}交易日{mode_text}"
    plot_info += f", 显示{len(pdf_plot)}个点"
    
    axes[0].set_title(f"NAV (freq={freq}, {plot_info})")
    axes[0].grid(alpha=0.25)
    
    axes[1].plot(x_positions, pdf_plot["return_rate"], lw=line_width, color="#2ca02c")
    axes[1].set_title(f"Return Rate (freq={freq})")
    axes[1].set_xlabel("Trade Date")
    axes[1].grid(alpha=0.25)
    
    # ============== 设置X轴标签 ==============
    # 动态选择刻度数量
    n_plot_points = len(pdf_plot)
    if n_plot_points <= 30:
        n_ticks = min(8, n_plot_points)
    elif n_plot_points <= 100:
        n_ticks = 10
    elif n_plot_points <= 300:
        n_ticks = 12
    else:
        n_ticks = 15
    
    # 生成均匀分布的刻度位置
    tick_pos = np.linspace(0, x_positions[-1], n_ticks)
    
    # 找到每个刻度位置对应的最近数据点
    tick_idx = []
    for pos in tick_pos:
        idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - pos))
        tick_idx.append(idx)
    
    # 去重并保持顺序
    unique_idx = []
    seen = set()
    for idx in tick_idx:
        if idx not in seen:
            seen.add(idx)
            unique_idx.append(idx)
    
    # 生成标签（根据时间跨度决定格式）
    time_span_days = (pdf_plot['plot_ts'].max() - pdf_plot['plot_ts'].min()).days
    if time_span_days > 365:  # 超过一年
        date_format = "%Y-%m"
    elif time_span_days > 180:  # 超过半年
        date_format = "%Y-%m-%d"
    else:
        date_format = "%m-%d"
    
    tick_labels = pdf_plot.iloc[unique_idx]["plot_ts"].dt.strftime(date_format)
    
    # 应用刻度
    axes[1].set_xticks([x_positions[i] for i in unique_idx])
    axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")
    
    # 移除上图的X轴标签
    axes[0].tick_params(axis="x", labelbottom=False)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    log_info(f"已保存走势图: {output_path}")
    
    # 打印统计信息
    max_gap = pdf_plot['days_to_next'].max()
    if max_gap > 3:
        log_info(f"检测到最长假期间隔: {int(max_gap)}天")
    
    # 记录采样信息
    if plot_every_n_trading_days > 1 or n_data_points > 2000:
        log_info(f"原始bar数: {n_data_points}, 原始交易日数: {n_trading_days}, 实际绘图: {len(pdf_plot)}")


def _build_multi_freq_wide_data(run_dir: Path) -> pl.DataFrame:
    freq_dirs = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("freq_")],
        key=lambda p: p.name,
    )
    if not freq_dirs:
        log_info(f"未找到频率目录: {run_dir}")
        return pl.DataFrame()

    wide_df: pl.DataFrame | None = None
    for freq_dir in freq_dirs:
        freq = freq_dir.name.replace("freq_", "", 1)
        backtest_csv = freq_dir / "backtest_detail.csv"
        if not backtest_csv.exists():
            log_info(f"缺少 backtest_detail.csv，跳过频率 {freq}: {backtest_csv}")
            continue

        backtest_df = pl.read_csv(backtest_csv, try_parse_dates=True)
        curve_data = _build_curve_data(backtest_df)
        if curve_data.is_empty():
            log_info(f"频率 {freq} 曲线数据为空，跳过")
            continue
        # 不再强制取每天最后一根bar，先按交易日求均值形成频率列
        # 之后再通过 plot_every_n_trading_days + sampling_mode 做降采样
        daily_avg = (
            curve_data
            .group_by("trade_date")
            .agg(
                pl.col("nav").last().alias(f"nav_{freq}"),
                pl.col("return_rate").last().alias(f"return_rate_{freq}"),
                pl.col("future_close").last().alias(f"future_close_{freq}"),
                pl.col("iv_avg").last().alias(f"iv_avg_{freq}"),
                pl.col("pnl_total").last().alias(f"pnl_total_{freq}"),
                pl.col("pnl_future").last().alias(f"pnl_future_{freq}"),
                pl.col("pnl_option").last().alias(f"pnl_option_{freq}"),
            )
            .sort("trade_date")
        )
        if wide_df is None:
            wide_df = daily_avg
        else:
            wide_df = wide_df.join(daily_avg, on="trade_date", how="full")
            if "trade_date_right" in wide_df.columns:
                wide_df = (
                    wide_df
                    .with_columns(pl.coalesce(["trade_date", "trade_date_right"]).alias("trade_date"))
                    .drop("trade_date_right")
                )

    if wide_df is None:
        return pl.DataFrame()
    return wide_df.sort("trade_date")


def _plot_multi_freq_curves_wide(
    multi_freq_wide_df: pl.DataFrame,
    output_path: Path,
    plot_every_n_trading_days: int = 1,
    sampling_mode: str = "avg",
) -> None:
    """
    多频率对比绘图（宽表输入）。

    Parameters
    ----------
    multi_freq_wide_df : pl.DataFrame
        交易日宽表，至少包含:
        - trade_date
        - nav_{freq}
        - return_rate_{freq}
    output_path : Path
        输出图片路径
    plot_every_n_trading_days : int
        每隔多少个交易日采样一个点
    sampling_mode : str
        采样模式: "last" 或 "avg"
    """
    setup_chinese_font()

    if multi_freq_wide_df.is_empty():
        log_info("multi_freq_wide_df 为空，跳过多频率绘图")
        return
    if "trade_date" not in multi_freq_wide_df.columns:
        log_info("multi_freq_wide_df 缺少 trade_date 列，跳过多频率绘图")
        return

    nav_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("nav_")])
    rr_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("return_rate_")])
    future_close_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("future_close_")])
    iv_avg_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("iv_avg_")])
    pnl_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("pnl_total_")])
    pnl_future_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("pnl_future_")])
    pnl_option_cols = sorted([c for c in multi_freq_wide_df.columns if c.startswith("pnl_option_")])
    freqs = sorted({c.replace("nav_", "", 1) for c in nav_cols if f"return_rate_{c.replace('nav_', '', 1)}" in rr_cols})
    if not freqs:
        log_info("未找到成对的 nav_{freq}/return_rate_{freq} 列，跳过多频率绘图")
        return

    pdf = multi_freq_wide_df.to_pandas()
    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])
    pdf = pdf.sort_values("trade_date").reset_index(drop=True)
    if pdf.empty:
        log_info("转换到 pandas 后为空，跳过多频率绘图")
        return

    if plot_every_n_trading_days > 1:
        sampled_rows = []
        for i in range(0, len(pdf), plot_every_n_trading_days):
            chunk = pdf.iloc[i:i + plot_every_n_trading_days]
            if chunk.empty:
                continue
            row: dict[str, object] = {"plot_ts": chunk["trade_date"].iloc[-1]}
            for freq in freqs:
                nav_col = f"nav_{freq}"
                rr_col = f"return_rate_{freq}"
                future_close_col = f"future_close_{freq}"
                iv_avg_col = f"iv_avg_{freq}"
                pnl_col = f"pnl_total_{freq}"
                pnl_future_col = f"pnl_future_{freq}"
                pnl_option_col = f"pnl_option_{freq}"
                
                nav_vals = chunk[nav_col].dropna() if nav_col in chunk.columns else pd.Series(dtype=float)
                rr_vals = chunk[rr_col].dropna() if rr_col in chunk.columns else pd.Series(dtype=float)
                future_close_vals = chunk[future_close_col].dropna() if future_close_col in chunk.columns else pd.Series(dtype=float)
                iv_avg_vals = chunk[iv_avg_col].dropna() if iv_avg_col in chunk.columns else pd.Series(dtype=float)
                pnl_vals = chunk[pnl_col].dropna() if pnl_col  in chunk.columns else pd.Series(dtype=float)
                pnl_future_vals = chunk[pnl_future_col].dropna() if pnl_future_col in chunk.columns else pd.Series(dtype=float)
                pnl_option_vals = chunk[pnl_option_col].dropna() if pnl_option_col in chunk.columns else pd.Series(dtype=float)
                
                if sampling_mode == "last":
                    row[nav_col] = float(nav_vals.iloc[-1]) if not nav_vals.empty else np.nan
                    row[rr_col] = float(rr_vals.iloc[-1]) if not rr_vals.empty else np.nan
                    row[future_close_col] = float(future_close_vals.iloc[-1]) if not future_close_vals.empty else np.nan
                    row[iv_avg_col] = float(iv_avg_vals.iloc[-1]) if not iv_avg_vals.empty else np.nan
                    row[pnl_col] = float(pnl_vals.iloc[-1]) if not pnl_vals.empty else np.nan
                    row[pnl_future_col] = float(pnl_future_vals.iloc[-1]) if not pnl_future_vals.empty else np.nan
                    row[pnl_option_cols] = float(pnl_option_vals.iloc[-1]) if not pnl_option_vals.empty else np.nan
                elif sampling_mode == "avg":
                    row[nav_col] = float(nav_vals.mean()) if not nav_vals.empty else np.nan
                    row[rr_col] = float(rr_vals.mean()) if not rr_vals.empty else np.nan
                    row[future_close_col] = float(future_close_vals.mean()) if not future_close_vals.empty else np.nan
                    row[iv_avg_col] = float(iv_avg_vals.mean()) if not iv_avg_vals.empty else np.nan
                    row[pnl_col] = float(pnl_vals.mean()) if not pnl_vals.empty else np.nan
                    row[pnl_future_col] = float(pnl_future_vals.mean()) if not pnl_future_vals.empty else np.nan
                    row[pnl_option_col] = float(pnl_option_vals.mean()) if not pnl_option_vals.empty else np.nan
                else:
                    raise ValueError(f"不支持的采样模式: {sampling_mode}")
            sampled_rows.append(row)
        pdf_plot = pd.DataFrame(sampled_rows)
    else:
        pdf_plot = pdf.rename(columns={"trade_date": "plot_ts"}).copy()

    if pdf_plot.empty:
        log_info("采样后数据为空，跳过多频率绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    # 设置同一种freq用同一种颜色
    color_map = plt.get_cmap('tab10')
    freq_color_dict = {freq: color_map(i % 10) for i, freq in enumerate(freqs)}
    
    for freq in freqs:
        nav_col = f"nav_{freq}"
        rr_col = f"return_rate_{freq}"
        future_close_col = f"future_close_{freq}"
        iv_avg_col = f"iv_avg_{freq}"
        pnl_col = f"pnl_total_{freq}"
        pnl_future_col = f"pnl_future_{freq}"
        pnl_option_col = f"pnl_option_{freq}"

        if rr_col in pdf_plot.columns:
            axes[0].plot(
                pdf_plot["plot_ts"], pdf_plot[rr_col], lw=1.4, label=f"freq={freq}", color=freq_color_dict[freq], linestyle='-'
            )
        # if nav_col in pdf_plot.columns:
        #     axes[0].plot(pdf_plot["plot_ts"], pdf_plot[nav_col], lw=1.6, label=f"NAV (freq={freq})", color=freq_color_dict[freq])
        if pnl_col in pdf_plot.columns:
            axes[1].plot(
                pdf_plot["plot_ts"], pdf_plot[pnl_col], lw=1.4, label=f"Pnl (freq={freq})", color=freq_color_dict[freq], linestyle='-'
            )
        if pnl_future_col in pdf_plot.columns:
            axes[1].plot(
                pdf_plot["plot_ts"], pdf_plot[pnl_future_col], lw=1.4, label=f"Future Pnl  (freq={freq})", color=freq_color_dict[freq], linestyle='--', alpha=0.2#, marker='o', markersize=4
            )
        if pnl_option_col in pdf_plot.columns:
            axes[1].plot(
                pdf_plot["plot_ts"], pdf_plot[pnl_option_col], lw=1.4, label=f"Option Pnl  (freq={freq})",  color=freq_color_dict[freq], linestyle='-.', alpha=0.2#, marker='s', markersize=4
            )

    mode_text = "取最后一点" if sampling_mode == "last" else "取平均值"
    axes[0].set_title(f"Return Rate Multi-Frequency (每{plot_every_n_trading_days}交易日{mode_text})")
    axes[0].set_xlabel("Trade Date")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[0].tick_params(axis="x", rotation=30)
    # 添加一个额外的图例说明颜色与频率的对应关系
    legend_elements = [Patch(facecolor=freq_color_dict[freq], label=f'Frequency: {freq}') for freq in freqs]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=8, title='Color Legend')

    axes[1].set_title(f"PnL Multi-Frequency ")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    # 用交易日真实日期坐标，保留节假日空档
    date_formatter = mdates.DateFormatter("%Y-%m-%d")
    for ax in axes:
        ax.xaxis.set_major_formatter(date_formatter)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    fig.tight_layout()
    # plt.show()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_info(f"已保存多频率对比图: {output_path}")

    fig2, ax2 = plt.subplots(figsize=(16, 4))
    # 选择第一个频率的 future_close 画线（有多个可以扩展逻辑）
    fc_col = "future_close_1d"
    label_fc = f"Future Close ({fc_col.replace('future_close_', '')})"
    ax2.plot(pdf_plot["plot_ts"], pdf_plot[fc_col], lw=1.4, color="tab:blue", label=label_fc)
    ax2.set_title("Future Close Curve")
    ax2.set_xlabel("Trade Date")
    ax2.set_ylabel("Future Close Price")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    ax2.xaxis.set_major_formatter(date_formatter)
    ax2.tick_params(axis="x", rotation=30)
    fig2.tight_layout()
    fc_path = output_path.parent / ("future_close.png")
    fig2.savefig(fc_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log_info(f"已保存 future close 走势图: {fc_path}")


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