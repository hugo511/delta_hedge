# Delta Hedge 回测项目

## 1. 环境准备

- Python 版本：`>=3.12`
- 推荐包管理工具：`uv`

安装依赖：

```bash
uv sync
```

## 2. 配置 TuShare Token

项目会从根目录 `.env` 读取 `TUSHARE_TOKEN`。

可先复制示例文件：

```bash
cp .env.example .env
```

然后在 `.env` 中填写：

```env
TUSHARE_TOKEN=your_tushare_token_here
```

## 3. 配置回测参数

默认配置文件是 `config/shfe_ag_demo.yaml`，可按需要修改：

- `backtest`：回测起止日期
- `future`：品种与交易所信息
- `hedge`：对冲参数（`frequency`、`roll_days_before_maturity`、`straddle_size` 等）

说明：

- `hedge.frequency` 支持列表，例如 `["1d", "15min"]`
- 主程序会自动按每个频率分别运行并分别输出结果

## 4. 运行回测

```bash
python main.py
```

运行结果会输出到：

```text
outputs/<run_timestamp>/
```

每个频率会在独立目录下保存：

- `backtest_detail.csv`：逐 bar 明细（持仓、交易、PnL、NAV）
- `daily_result.csv`：逐日指标
- `daily_position.csv`：逐日持仓合约与仓位
- `summary.csv`：汇总结果
- `daily_pnl.png`、`strategy_curves.png`：图表
- `run_config.json`：本次运行使用的配置快照

## 5. 本地数据库目录 `local_db`

行情与合约数据会缓存到 `local_db/`（`parquet` 文件）。

典型目录结构：

```text
local_db/
  contract_info/
    future_basic/
    option_basic/
  future_price_daily/
    SHFE_AG/
  future_price_minute/
    SHFE_AG/
  option_price_daily/
    SHFE_AG/
  option_price_minute/
    SHFE_AG/
  market_data/
```

注意：

- `local_db` 下的数据文件（如 `*.parquet`）不提交到 Git
- 仓库仅保留目录结构占位文件（`.gitkeep`）


