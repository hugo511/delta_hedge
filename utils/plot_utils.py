import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from utils.logger import log_info
import numpy as np
import pandas as pd 
import matplotlib as mpl
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
            print(f"使用字体: {font}")
            return
    
    # 如果都没有，使用默认字体但警告
    print("警告: 未找到中文字体，中文可能显示为方框")
    



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
            ["bar_ts", "trade_date", "nav", "capital_used", "capital_usage", "return_rate"]
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


# def _plot_curves(curve_data: pl.DataFrame, output_path: Path, freq: str, plot_every_n_trading_days: int = 1) -> None:
#     """
#     绘制净值曲线和收益率曲线
    
#     Parameters:
#     -----------
#     curve_data : pl.DataFrame
#         包含绘图数据的DataFrame
#     output_path : Path
#         输出图片路径
#     freq : str
#         数据频率（用于标题显示）
#     plot_every_n_trading_days : int
#         每隔多少个交易日画一个点，默认1表示每个交易日都画
#         例如：plot_every_n_trading_days=5 表示每5个交易日画一个点
#         无论原始数据是15min还是日频，都按交易日维度采样
#     """
#     # 设置中文字体
#     # plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
#     # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     setup_chinese_font()
    
#     if curve_data.is_empty():
#         log_info("curve_data 为空，跳过绘图")
#         return

#     plot_data = curve_data
#     if "name" in curve_data.columns:
#         # close 记录对应平仓点，资金占用可能为0，默认不纳入曲线展示
#         filtered = curve_data.filter(pl.col("name") != "close")
#         if not filtered.is_empty():
#             plot_data = filtered

#     # 原始频率是 1d 还是 15min
#     daily_plot_data = (
#         plot_data
#         .sort(["trade_date", "bar_ts"])
#         .with_columns(pl.col("bar_ts").cast(pl.Datetime).alias("plot_ts"))
#     ).sort("plot_ts")

#     if daily_plot_data.is_empty():
#         log_info("daily_plot_data 为空，跳过绘图")
#         return

#     # 转换为 pandas 并准备数据
#     pdf = daily_plot_data.select(["plot_ts", "nav", "return_rate"]).to_pandas()
#     pdf['plot_ts'] = pd.to_datetime(pdf['plot_ts'])
#     pdf = pdf.sort_values('plot_ts').reset_index(drop=True)
    
#     # ============== 按交易日维度进行降采样 ==============
#     n_data_points = len(pdf)
    
#     # 提取交易日日期（忽略时间）
#     pdf['trade_date_only'] = pdf['plot_ts'].dt.date
    
#     # 获取唯一的交易日列表
#     unique_trading_days = pdf['trade_date_only'].unique()
#     n_trading_days = len(unique_trading_days)
    
#     log_info(f"原始数据共 {n_data_points} 个bar，涉及 {n_trading_days} 个交易日")
    
#     if plot_every_n_trading_days > 1:
#         log_info(f"设置每隔 {plot_every_n_trading_days} 个交易日画一个点")
        
#         # 对交易日进行采样
#         trading_days_sorted = sorted(unique_trading_days)
        
#         # 选择每隔 plot_every_n_trading_days 个交易日
#         sampled_trading_days = trading_days_sorted[::plot_every_n_trading_days]
        
#         # 确保包含最后一个交易日
#         if sampled_trading_days[-1] != trading_days_sorted[-1]:
#             sampled_trading_days.append(trading_days_sorted[-1])
        
#         # 对于每个被选中的交易日，选择该交易日最后一个bar的数据（收盘价）
#         sampled_indices = []
#         for trade_date in sampled_trading_days:
#             # 找到该交易日所有的数据点
#             day_data = pdf[pdf['trade_date_only'] == trade_date]
#             if not day_data.empty:
#                 # 取该交易日最后一个bar（收盘时刻）
#                 last_idx = day_data.index[-1]
#                 sampled_indices.append(last_idx)
        
#         # 创建采样后的DataFrame
#         pdf_plot = pdf.loc[sampled_indices].copy().reset_index(drop=True)
        
#         # 检查是否采样后点还是太多，如果是，再按max_points限制
#         max_points = 2000
#         if len(pdf_plot) > max_points:
#             log_info(f"按交易日采样后仍有 {len(pdf_plot)} 个点，继续降采样至约 {max_points} 点")
#             # 二次降采样（按时间均匀）
#             sample_interval = len(pdf_plot) // max_points
#             sampled_idx2 = pdf_plot.iloc[::sample_interval].index
#             if sampled_idx2[-1] != pdf_plot.index[-1]:
#                 sampled_idx2 = sampled_idx2.append(pd.Index([pdf_plot.index[-1]]))
#             pdf_plot = pdf_plot.loc[sampled_idx2].copy().reset_index(drop=True)
#     else:
#         # 如果 plot_every_n_trading_days = 1，检查是否需要按点数降采样
#         max_points = 2000
        
#         if n_data_points > max_points:
#             log_info(f"数据点过多 ({n_data_points})，进行降采样至约 {max_points} 点")
            
#             # 按交易日采样：每个交易日取最后一个bar
#             trading_days_sorted = sorted(unique_trading_days)
            
#             sampled_indices = []
#             for trade_date in trading_days_sorted:
#                 day_data = pdf[pdf['trade_date_only'] == trade_date]
#                 if not day_data.empty:
#                     last_idx = day_data.index[-1]
#                     sampled_indices.append(last_idx)
            
#             pdf_sampled = pdf.loc[sampled_indices].copy().reset_index(drop=True)
            
#             # 如果按交易日采样后还是太多，再按时间均匀降采样
#             if len(pdf_sampled) > max_points:
#                 log_info(f"按交易日采样后仍有 {len(pdf_sampled)} 个点，继续降采样")
#                 sample_interval = len(pdf_sampled) // max_points
#                 sampled_idx2 = pdf_sampled.iloc[::sample_interval].index
#                 if sampled_idx2[-1] != pdf_sampled.index[-1]:
#                     sampled_idx2 = sampled_idx2.append(pd.Index([pdf_sampled.index[-1]]))
#                 pdf_plot = pdf_sampled.loc[sampled_idx2].copy().reset_index(drop=True)
#             else:
#                 pdf_plot = pdf_sampled
#         else:
#             pdf_plot = pdf.copy()
    
#     log_info(f"实际绘图点数: {len(pdf_plot)} (原始bar数: {n_data_points}, 原始交易日数: {n_trading_days}, 采样间隔: {plot_every_n_trading_days if plot_every_n_trading_days>1 else 'auto'}交易日)")
    
#     # ============== 创建非线性X轴 ==============
#     # 计算相邻交易日之间的实际天数
#     pdf_plot['next_date'] = pdf_plot['plot_ts'].shift(-1)
#     pdf_plot['days_to_next'] = (pdf_plot['next_date'] - pdf_plot['plot_ts']).dt.days.fillna(1)
    
#     # 创建X轴位置序列
#     x_positions = [0]
#     for i in range(len(pdf_plot)-1):
#         # 基础间距1，如果间隔>3天，额外增加间距
#         gap = pdf_plot.iloc[i]['days_to_next']
#         extra_space = max(0, (gap - 1) * 0.5)  # 每多一天增加0.5个单位
#         x_positions.append(x_positions[-1] + 1 + extra_space)
    
#     # ============== 创建图表 ==============
#     fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
#     # 绘制曲线（用更细的线）
#     line_width = 1.2 if len(pdf_plot) > 500 else 1.5
#     axes[0].plot(x_positions, pdf_plot["nav"], lw=line_width, color="#1f77b4")
    
#     # 构建标题
#     plot_info = f"共{n_trading_days}个交易日, {n_data_points}个bar"
#     if plot_every_n_trading_days > 1:
#         plot_info += f", 每{plot_every_n_trading_days}交易日取1点"
#     plot_info += f", 显示{len(pdf_plot)}个点"
    
#     axes[0].set_title(f"NAV (freq={freq}, {plot_info})")
#     axes[0].grid(alpha=0.25)
    
#     axes[1].plot(x_positions, pdf_plot["return_rate"], lw=line_width, color="#2ca02c")
#     axes[1].set_title(f"Return Rate (freq={freq})")
#     axes[1].set_xlabel("Trade Date")
#     axes[1].grid(alpha=0.25)
    
#     # ============== 设置X轴标签 ==============
#     # 动态选择刻度数量
#     n_plot_points = len(pdf_plot)
#     if n_plot_points <= 30:
#         n_ticks = min(8, n_plot_points)
#     elif n_plot_points <= 100:
#         n_ticks = 10
#     elif n_plot_points <= 300:
#         n_ticks = 12
#     else:
#         n_ticks = 15
    
#     # 生成均匀分布的刻度位置
#     tick_pos = np.linspace(0, x_positions[-1], n_ticks)
    
#     # 找到每个刻度位置对应的最近数据点
#     tick_idx = []
#     for pos in tick_pos:
#         idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - pos))
#         tick_idx.append(idx)
    
#     # 去重并保持顺序
#     unique_idx = []
#     seen = set()
#     for idx in tick_idx:
#         if idx not in seen:
#             seen.add(idx)
#             unique_idx.append(idx)
    
#     # 生成标签（根据时间跨度决定格式）
#     time_span_days = (pdf_plot['plot_ts'].max() - pdf_plot['plot_ts'].min()).days
#     if time_span_days > 365:  # 超过一年
#         date_format = "%Y-%m"
#     elif time_span_days > 180:  # 超过半年
#         date_format = "%Y-%m-%d"
#     else:
#         date_format = "%m-%d"
    
#     tick_labels = pdf_plot.iloc[unique_idx]["plot_ts"].dt.strftime(date_format)
    
#     # 应用刻度
#     axes[1].set_xticks([x_positions[i] for i in unique_idx])
#     axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")
    
#     # 移除上图的X轴标签
#     axes[0].tick_params(axis="x", labelbottom=False)
    
#     # 调整布局
#     fig.tight_layout()
    
#     # 保存图片
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
    
#     log_info(f"已保存走势图: {output_path}")
    
#     # 打印统计信息
#     max_gap = pdf_plot['days_to_next'].max()
#     if max_gap > 3:
#         log_info(f"检测到最长假期间隔: {int(max_gap)}天")
    
#     # 记录采样信息
#     if plot_every_n_trading_days > 1 or n_data_points > 2000:
#         log_info(f"原始bar数: {n_data_points}, 原始交易日数: {n_trading_days}, 实际绘图: {len(pdf_plot)}")





# def _plot_curves(curve_data: pl.DataFrame, output_path: Path, freq: str, plot_every_n_bars: int = 1) -> None:
#     """
#     绘制净值曲线和收益率曲线
    
#     Parameters:
#     -----------
#     curve_data : pl.DataFrame
#         包含绘图数据的DataFrame
#     output_path : Path
#         输出图片路径
#     freq : str
#         数据频率（用于标题显示）
#     plot_every_n_bars : int
#         每隔多少个bar画一个点，默认1表示每个点都画
#         例如：plot_every_n_bars=10 表示每10个bar画一个点
#     """
#     if curve_data.is_empty():
#         log_info("curve_data 为空，跳过绘图")
#         return

#     plot_data = curve_data
#     if "name" in curve_data.columns:
#         # close 记录对应平仓点，资金占用可能为0，默认不纳入曲线展示
#         filtered = curve_data.filter(pl.col("name") != "close")
#         if not filtered.is_empty():
#             plot_data = filtered

#     # 原始频率是 1d 还是 15min
#     daily_plot_data = (
#         plot_data
#         .sort(["trade_date", "bar_ts"])
#         .with_columns(pl.col("bar_ts").cast(pl.Datetime).alias("plot_ts"))
#     ).sort("plot_ts")

#     if daily_plot_data.is_empty():
#         log_info("daily_plot_data 为空，跳过绘图")
#         return

#     # 转换为 pandas 并准备数据
#     pdf = daily_plot_data.select(["plot_ts", "nav", "return_rate"]).to_pandas()
#     pdf['plot_ts'] = pd.to_datetime(pdf['plot_ts'])
#     pdf = pdf.sort_values('plot_ts').reset_index(drop=True)
    
#     # ============== 根据 plot_every_n_bars 进行降采样 ==============
#     n_data_points = len(pdf)
    
#     if plot_every_n_bars > 1:
#         log_info(f"设置每隔 {plot_every_n_bars} 个bar画一个点")
        
#         # 选择每隔 plot_every_n_bars 的点
#         sampled_idx = pdf.iloc[::plot_every_n_bars].index
        
#         # 确保包含最后一个点，避免图表截断
#         if sampled_idx[-1] != pdf.index[-1]:
#             sampled_idx = sampled_idx.append(pd.Index([pdf.index[-1]]))
        
#         # 创建采样后的DataFrame
#         pdf_plot = pdf.loc[sampled_idx].copy()
        
#         # 检查是否降采样后点还是太多，如果是，再按max_points限制
#         if len(pdf_plot) > 2000:
#             log_info(f"降采样后仍有 {len(pdf_plot)} 个点，继续降采样至约2000点")
#             # 二次降采样
#             sample_interval = len(pdf_plot) // 2000
#             sampled_idx2 = pdf_plot.iloc[::sample_interval].index
#             if sampled_idx2[-1] != pdf_plot.index[-1]:
#                 sampled_idx2 = sampled_idx2.append(pd.Index([pdf_plot.index[-1]]))
#             pdf_plot = pdf_plot.loc[sampled_idx2].copy()
#     else:
#         # 如果 plot_every_n_bars = 1，用原来的降采样逻辑
#         max_points = 2000  # 设置最大点数，避免曲线太密
        
#         if n_data_points > max_points:
#             log_info(f"数据点过多 ({n_data_points})，进行降采样至约 {max_points} 点")
            
#             # 按时间均匀降采样
#             pdf['time_idx'] = range(len(pdf))
            
#             # 计算采样间隔
#             sample_interval = n_data_points // max_points
            
#             # 选择每隔 sample_interval 的点
#             sampled_idx = pdf.iloc[::sample_interval].index
            
#             # 确保包含最后一个点
#             if sampled_idx[-1] != pdf.index[-1]:
#                 sampled_idx = sampled_idx.append(pd.Index([pdf.index[-1]]))
            
#             # 创建采样后的DataFrame
#             pdf_sampled = pdf.loc[sampled_idx].copy()
            
#             # 检查是否是日频数据（相邻时间差平均 > 12小时）
#             time_diffs = pdf['plot_ts'].diff().dt.total_seconds().median() / 3600
#             if time_diffs > 12:  # 日频数据
#                 log_info("检测到日频数据，按周降采样作为备选")
#                 pdf['date'] = pdf['plot_ts'].dt.date
#                 weekly = pdf.groupby(pd.Grouper(key='plot_ts', freq='W')).agg({
#                     'nav': 'last',
#                     'return_rate': 'last'
#                 }).dropna().reset_index()
#                 # 如果周频采样后点数更少，用周频
#                 if len(weekly) < len(pdf_sampled) * 0.8:
#                     pdf_sampled = weekly
            
#             pdf_plot = pdf_sampled
#         else:
#             pdf_plot = pdf
    
#     log_info(f"实际绘图点数: {len(pdf_plot)} (原始点数: {n_data_points}, 采样间隔: {plot_every_n_bars if plot_every_n_bars>1 else 'auto'})")
    
#     # ============== 创建非线性X轴 ==============
#     # 计算相邻交易日之间的实际天数
#     pdf_plot['next_date'] = pdf_plot['plot_ts'].shift(-1)
#     pdf_plot['days_to_next'] = (pdf_plot['next_date'] - pdf_plot['plot_ts']).dt.days.fillna(1)
    
#     # 创建X轴位置序列
#     x_positions = [0]
#     for i in range(len(pdf_plot)-1):
#         # 基础间距1，如果间隔>3天，额外增加间距
#         gap = pdf_plot.iloc[i]['days_to_next']
#         extra_space = max(0, (gap - 1) * 0.5)  # 每多一天增加0.5个单位
#         x_positions.append(x_positions[-1] + 1 + extra_space)
    
#     # ============== 创建图表 ==============
#     fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
#     # 绘制曲线（用更细的线）
#     line_width = 1.2 if len(pdf_plot) > 500 else 1.5
#     axes[0].plot(x_positions, pdf_plot["nav"], lw=line_width, color="#1f77b4")
    
#     # 构建标题
#     plot_info = f"共{len(pdf)}个原始点"
#     if plot_every_n_bars > 1:
#         plot_info += f", 每{plot_every_n_bars}bar取1点"
#     plot_info += f", 显示{len(pdf_plot)}个点"
    
#     axes[0].set_title(f"NAV (freq={freq}, {plot_info})")
#     axes[0].grid(alpha=0.25)
    
#     axes[1].plot(x_positions, pdf_plot["return_rate"], lw=line_width, color="#2ca02c")
#     axes[1].set_title(f"Return Rate (freq={freq})")
#     axes[1].set_xlabel("Trade Date")
#     axes[1].grid(alpha=0.25)
    
#     # ============== 设置X轴标签 ==============
#     # 动态选择刻度数量
#     n_plot_points = len(pdf_plot)
#     if n_plot_points <= 30:
#         n_ticks = min(8, n_plot_points)
#     elif n_plot_points <= 100:
#         n_ticks = 10
#     elif n_plot_points <= 300:
#         n_ticks = 12
#     else:
#         n_ticks = 15
    
#     # 生成均匀分布的刻度位置
#     tick_pos = np.linspace(0, x_positions[-1], n_ticks)
    
#     # 找到每个刻度位置对应的最近数据点
#     tick_idx = []
#     for pos in tick_pos:
#         idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - pos))
#         tick_idx.append(idx)
    
#     # 去重并保持顺序
#     unique_idx = []
#     seen = set()
#     for idx in tick_idx:
#         if idx not in seen:
#             seen.add(idx)
#             unique_idx.append(idx)
    
#     # 生成标签（根据时间跨度决定格式）
#     time_span_days = (pdf_plot['plot_ts'].max() - pdf_plot['plot_ts'].min()).days
#     if time_span_days > 365:  # 超过一年
#         date_format = "%Y-%m"
#     elif time_span_days > 180:  # 超过半年
#         date_format = "%Y-%m-%d"
#     else:
#         date_format = "%m-%d"
    
#     tick_labels = pdf_plot.iloc[unique_idx]["plot_ts"].dt.strftime(date_format)
    
#     # 应用刻度
#     axes[1].set_xticks([x_positions[i] for i in unique_idx])
#     axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")
    
#     # # ============== 可选：标记长假期 ==============
#     # # 如果数据点不是太多，标记假期
#     # if len(pdf_plot) < 500:
#     #     for i, gap in enumerate(pdf_plot['days_to_next'].iloc[:-1]):
#     #         if gap > 5:  # 超过5天的假期才标记
#     #             mid_x = (x_positions[i] + x_positions[i+1]) / 2
#     #             ylim = axes[1].get_ylim()
#     #             y_pos = ylim[1] * 0.9
#     #             axes[1].text(mid_x, y_pos, f"{int(gap)}d", 
#     #                         fontsize=7, ha='center', color='red', alpha=0.7,
#     #                         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.5))
    
#     # 移除上图的X轴标签
#     axes[0].tick_params(axis="x", labelbottom=False)
    
#     # 调整布局
#     fig.tight_layout()
    
#     # 保存图片
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
    
#     log_info(f"已保存走势图: {output_path}")
    
#     # 打印统计信息
#     max_gap = pdf_plot['days_to_next'].max()
#     if max_gap > 3:
#         log_info(f"检测到最长假期间隔: {int(max_gap)}天")
    
#     # 记录降采样信息
#     if plot_every_n_bars > 1 or n_data_points > 2000:
#         log_info(f"原始数据点 {n_data_points}，实际绘图 {len(pdf_plot)}")













# def _plot_curves(curve_data: pl.DataFrame, output_path: Path, freq: str) -> None:
#     if curve_data.is_empty():
#         log_info("curve_data 为空，跳过绘图")
#         return

#     plot_data = curve_data
#     if "name" in curve_data.columns:
#         # close 记录对应平仓点，资金占用可能为0，默认不纳入曲线展示
#         filtered = curve_data.filter(pl.col("name") != "close")
#         if not filtered.is_empty():
#             plot_data = filtered

#     # 原始频率是 1d 还是 15min
#     daily_plot_data = (
#         plot_data
#         .sort(["trade_date", "bar_ts"])
#         .with_columns(pl.col("bar_ts").cast(pl.Datetime).alias("plot_ts"))
#     ).sort("plot_ts")

#     if daily_plot_data.is_empty():
#         log_info("daily_plot_data 为空，跳过绘图")
#         return

#     # 转换为 pandas 并准备数据
#     pdf = daily_plot_data.select(["plot_ts", "nav", "return_rate"]).to_pandas()
#     pdf['plot_ts'] = pd.to_datetime(pdf['plot_ts'])
#     pdf = pdf.sort_values('plot_ts').reset_index(drop=True)
    
#     # ============== 数据降采样：如果数据点太多，进行降采样 ==============
#     n_data_points = len(pdf)
#     max_points = 2000  # 设置最大点数，避免曲线太密
    
#     if n_data_points > max_points:
#         log_info(f"数据点过多 ({n_data_points})，进行降采样至约 {max_points} 点")
        
#         # 方法1：按时间均匀降采样（推荐）
#         # 创建时间索引
#         pdf['time_idx'] = range(len(pdf))
        
#         # 计算采样间隔
#         sample_interval = n_data_points // max_points
        
#         # 选择每隔 sample_interval 的点
#         sampled_idx = pdf.iloc[::sample_interval].index
        
#         # 确保包含最后一个点
#         if sampled_idx[-1] != pdf.index[-1]:
#             sampled_idx = sampled_idx.append(pd.Index([pdf.index[-1]]))
        
#         # 创建采样后的DataFrame
#         pdf_sampled = pdf.loc[sampled_idx].copy()
        
#         # 方法2：按日期重采样（如果是日频数据）
#         # 检查是否是日频数据（相邻时间差平均 > 12小时）
#         time_diffs = pdf['plot_ts'].diff().dt.total_seconds().median() / 3600
#         if time_diffs > 12:  # 日频数据
#             log_info("检测到日频数据，按周降采样")
#             pdf['date'] = pdf['plot_ts'].dt.date
#             weekly = pdf.groupby(pd.Grouper(key='plot_ts', freq='W')).agg({
#                 'nav': 'last',
#                 'return_rate': 'last'
#             }).dropna().reset_index()
#             pdf_sampled = weekly
        
#         # 使用降采样后的数据
#         pdf_plot = pdf_sampled
#     else:
#         pdf_plot = pdf
    
#     log_info(f"实际绘图点数: {len(pdf_plot)}")
    
#     # ============== 创建非线性X轴 ==============
#     # 计算相邻交易日之间的实际天数
#     pdf_plot['next_date'] = pdf_plot['plot_ts'].shift(-1)
#     pdf_plot['days_to_next'] = (pdf_plot['next_date'] - pdf_plot['plot_ts']).dt.days.fillna(1)
    
#     # 创建X轴位置序列
#     x_positions = [0]
#     for i in range(len(pdf_plot)-1):
#         # 基础间距1，如果间隔>3天，额外增加间距
#         gap = pdf_plot.iloc[i]['days_to_next']
#         extra_space = max(0, (gap - 1) * 0.5)  # 每多一天增加0.5个单位
#         x_positions.append(x_positions[-1] + 1 + extra_space)
    
#     # ============== 创建图表 ==============
#     fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)  # 稍微调大尺寸
    
#     # 绘制曲线
#     axes[0].plot(x_positions, pdf_plot["nav"], lw=1.5, color="#1f77b4")
#     axes[0].set_title(f"NAV (freq={freq}, 共{len(pdf)}个原始点, 显示{len(pdf_plot)}个)")
#     axes[0].grid(alpha=0.25)
    
#     axes[1].plot(x_positions, pdf_plot["return_rate"], lw=1.5, color="#2ca02c")
#     axes[1].set_title(f"Return Rate (freq={freq})")
#     axes[1].set_xlabel("Trade Date")
#     axes[1].grid(alpha=0.25)
    
#     # ============== 设置X轴标签 ==============
#     # 动态选择刻度数量
#     n_plot_points = len(pdf_plot)
#     if n_plot_points <= 30:
#         n_ticks = min(10, n_plot_points)
#     elif n_plot_points <= 100:
#         n_ticks = 12
#     elif n_plot_points <= 300:
#         n_ticks = 15
#     else:
#         n_ticks = 20
    
#     # 生成均匀分布的刻度位置
#     tick_pos = np.linspace(0, x_positions[-1], n_ticks)
    
#     # 找到每个刻度位置对应的最近数据点
#     tick_idx = []
#     for pos in tick_pos:
#         idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - pos))
#         tick_idx.append(idx)
    
#     # 去重并保持顺序
#     unique_idx = []
#     seen = set()
#     for idx in tick_idx:
#         if idx not in seen:
#             seen.add(idx)
#             unique_idx.append(idx)
    
#     # 生成标签（根据时间跨度决定格式）
#     time_span_days = (pdf_plot['plot_ts'].max() - pdf_plot['plot_ts'].min()).days
#     if time_span_days > 180:  # 超过半年
#         date_format = "%Y-%m"
#     else:
#         date_format = "%Y-%m-%d"
    
#     tick_labels = pdf_plot.iloc[unique_idx]["plot_ts"].dt.strftime(date_format)
    
#     # 应用刻度
#     axes[1].set_xticks([x_positions[i] for i in unique_idx])
#     axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")
    
#     # ============== 可选：标记长假期 ==============
#     # 如果数据点不是太多，标记假期
#     if len(pdf_plot) < 500:
#         for i, gap in enumerate(pdf_plot['days_to_next'].iloc[:-1]):
#             if gap > 5:  # 超过5天的假期才标记
#                 mid_x = (x_positions[i] + x_positions[i+1]) / 2
#                 ylim = axes[1].get_ylim()
#                 y_pos = ylim[1] * 0.9
#                 axes[1].text(mid_x, y_pos, f"{int(gap)}d", 
#                             fontsize=7, ha='center', color='red', alpha=0.7,
#                             bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.5))
    
#     # 移除上图的X轴标签
#     axes[0].tick_params(axis="x", labelbottom=False)
    
#     # 调整布局
#     fig.tight_layout()
    
#     # 保存图片
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
    
#     log_info(f"已保存走势图: {output_path}")
    
#     # 打印统计信息
#     max_gap = pdf_plot['days_to_next'].max()
#     if max_gap > 3:
#         log_info(f"检测到最长假期间隔: {int(max_gap)}天")
    
#     # 记录降采样信息
#     if n_data_points > max_points:
#         log_info(f"原始数据点 {n_data_points}，降采样后 {len(pdf_plot)}")








# def _plot_curves(curve_data: pl.DataFrame, output_path: Path, freq: str) -> None:
#     if curve_data.is_empty():
#         log_info("curve_data 为空，跳过绘图")
#         return

#     plot_data = curve_data
#     if "name" in curve_data.columns:
#         # close 记录对应平仓点，资金占用可能为0，默认不纳入曲线展示
#         filtered = curve_data.filter(pl.col("name") != "close")
#         if not filtered.is_empty():
#             plot_data = filtered

#     # 原始频率是 1d 还是 15min
#     daily_plot_data = (
#         plot_data
#         .sort(["trade_date", "bar_ts"])
#         .with_columns(pl.col("bar_ts").cast(pl.Datetime).alias("plot_ts"))
#     ).sort("plot_ts")

#     if daily_plot_data.is_empty():
#         log_info("daily_plot_data 为空，跳过绘图")
#         return

#     # 转换为 pandas 并准备数据
#     pdf = daily_plot_data.select(["plot_ts", "nav", "return_rate"]).to_pandas()
#     pdf['plot_ts'] = pd.to_datetime(pdf['plot_ts'])
#     pdf = pdf.sort_values('plot_ts').reset_index(drop=True)
    
#     # 创建非线性X轴：让假期间隔有更多空间
#     # 计算相邻交易日之间的实际天数
#     pdf['next_date'] = pdf['plot_ts'].shift(-1)
#     pdf['days_to_next'] = (pdf['next_date'] - pdf['plot_ts']).dt.days.fillna(1)
    
#     # 创建X轴位置序列
#     x_positions = [0]
#     for i in range(len(pdf)-1):
#         # 基础间距1，如果间隔>3天，额外增加间距
#         gap = pdf.iloc[i]['days_to_next']
#         extra_space = max(0, (gap - 1) * 0.5)  # 每多一天增加0.5个单位
#         x_positions.append(x_positions[-1] + 1 + extra_space)
    
#     # 创建图表
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
#     # 绘制曲线
#     axes[0].plot(x_positions, pdf["nav"], lw=1.8, color="#1f77b4")
#     axes[0].set_title(f"NAV (freq={freq})")
#     axes[0].grid(alpha=0.25)
    
#     axes[1].plot(x_positions, pdf["return_rate"], lw=1.6, color="#2ca02c")
#     axes[1].set_title(f"Return Rate (freq={freq})")
#     axes[1].set_xlabel("Trade Date")
#     axes[1].grid(alpha=0.25)
    
#     # 设置X轴标签 - 动态选择刻度数量
#     n_data_points = len(pdf)
#     if n_data_points <= 30:
#         n_ticks = min(10, n_data_points)  # 数据点少时减少刻度
#     elif n_data_points <= 100:
#         n_ticks = 15
#     elif n_data_points <= 500:
#         n_ticks = 20
#     else:
#         n_ticks = 25
    
#     # 生成均匀分布的刻度位置
#     tick_pos = np.linspace(0, x_positions[-1], n_ticks)
    
#     # 找到每个刻度位置对应的最近数据点
#     tick_idx = []
#     for pos in tick_pos:
#         # 找到x_positions中最接近pos的索引
#         idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - pos))
#         tick_idx.append(idx)
    
#     # 去重并保持顺序
#     unique_idx = []
#     seen = set()
#     for idx in tick_idx:
#         if idx not in seen:
#             seen.add(idx)
#             unique_idx.append(idx)
    
#     # 生成标签
#     tick_labels = pdf.iloc[unique_idx]["plot_ts"].dt.strftime("%Y-%m-%d")
    
#     # 应用刻度
#     axes[1].set_xticks([x_positions[i] for i in unique_idx])
#     axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")
    
#     # # 标记长假期（间隔 > 3天）
#     # for i, gap in enumerate(pdf['days_to_next'].iloc[:-1]):
#     #     if gap > 3:
#     #         mid_x = (x_positions[i] + x_positions[i+1]) / 2
#     #         # 获取当前y轴范围
#     #         ylim = axes[1].get_ylim()
#     #         y_pos = ylim[1] * 0.9  # 放在顶部90%位置
#     #         axes[1].text(mid_x, y_pos, f"{int(gap)}d", 
#     #                     fontsize=8, ha='center', color='red', 
#     #                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            
#     #         # 可选：添加垂直虚线标记假期开始
#     #         axes[1].axvline(x=x_positions[i], color='red', linestyle=':', alpha=0.3, linewidth=0.5)
    
#     # 移除上图的X轴标签（因为它们共享X轴）
#     axes[0].tick_params(axis="x", labelbottom=False)
    
#     # 调整布局
#     fig.tight_layout()
    
#     # 保存图片
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
    
#     log_info(f"已保存走势图: {output_path}")
    
#     # 可选：打印统计信息
#     max_gap = pdf['days_to_next'].max()
#     if max_gap > 3:
#         log_info(f"检测到最长假期间隔: {int(max_gap)}天")



























# def _plot_curves(curve_data: pl.DataFrame, output_path: Path, freq: str) -> None:
#     if curve_data.is_empty():
#         log_info("curve_data 为空，跳过绘图")
#         return

#     plot_data = curve_data
#     if "name" in curve_data.columns:
#         # close 记录对应平仓点，资金占用可能为0，默认不纳入曲线展示
#         filtered = curve_data.filter(pl.col("name") != "close")
#         if not filtered.is_empty():
#             plot_data = filtered

#     # 原始频率是 1d 还是 15minr”
#     daily_plot_data = (
#         plot_data
#         .sort(["trade_date", "bar_ts"])
#         .with_columns(pl.col("bar_ts").cast(pl.Datetime).alias("plot_ts"))
#     ).sort("plot_ts")

#     if daily_plot_data.is_empty():
#         log_info("daily_plot_data 为空，跳过绘图")
#         return

#     pdf = daily_plot_data.select(["plot_ts", "nav", "return_rate"]).to_pandas()
#     fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

#     axes[0].plot(pdf["plot_ts"], pdf["nav"], lw=1.8, color="#1f77b4")
#     axes[0].set_title(f"NAV (freq={freq})")
#     axes[0].grid(alpha=0.25)

#     axes[1].plot(pdf["plot_ts"], pdf["return_rate"], lw=1.6, color="#2ca02c")
#     axes[1].set_title(f"Return Rate (freq={freq})")
#     axes[1].set_xlabel("Trade Date")
#     axes[1].grid(alpha=0.25)
#     axes[1].tick_params(axis="x", rotation=30)

#     x_formatter = mdates.DateFormatter("%Y-%m-%d")
#     for ax in axes: 
#         ax.xaxis.set_major_formatter(x_formatter)

#     for ax in axes[:-1]:
#         ax.tick_params(axis="x", labelbottom=False)
#     fig.tight_layout()
#     # plt.show()

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=150)
#     plt.close(fig)
#     log_info(f"已保存走势图: {output_path}")














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