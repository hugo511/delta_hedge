import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path


from data_fetcher.option_loader import OptionLoader
from core.config_shema import ExperimentConfig
from utils.logger import log_info


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _year_fraction(expiry_date: date, trade_date: date) -> float:
    return max((expiry_date - trade_date).days, 1) / 365.0


def plot_daily_pnl(
    daily_result: pl.DataFrame,
    output_path: str = "outputs/daily_pnl.png",
    show: bool = False,
) -> None:
    """Plot NAV ratio, strategy return, capital usage and daily PnL."""
    if daily_result.is_empty():
        log_info("daily_result 为空，跳过绘图")
        return
    required_cols = {"trade_date", "daily_pnl", "nav_ratio", "strategy_return", "capital_usage"}
    if not required_cols.issubset(set(daily_result.columns)):
        log_info(f"daily_result 缺少必要列: {required_cols}")
        return

    df = daily_result.sort("trade_date")
    pdf = df.select(["trade_date", "daily_pnl", "nav_ratio", "strategy_return", "capital_usage"]).to_pandas()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(pdf["trade_date"], pdf["nav_ratio"], lw=1.8, color="#1f77b4")
    axes[0].set_title("NAV Ratio (NAV / Init Cash)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(pdf["trade_date"], pdf["strategy_return"], lw=1.6, color="#2ca02c")
    axes[1].set_title("Strategy Return (CumPnL / Capital Used)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(pdf["trade_date"], pdf["capital_usage"], lw=1.6, color="#d62728")
    axes[2].set_title("Capital Usage (Capital Used / NAV)")
    axes[2].grid(alpha=0.25)

    axes[3].bar(pdf["trade_date"], pdf["daily_pnl"], color="#ff7f0e", alpha=0.8)
    axes[3].set_title("Daily PnL")
    axes[3].grid(alpha=0.25)
    axes[3].tick_params(axis="x", rotation=30)
    axes[3].set_xlabel("Trade Date")

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    log_info(f"已保存PnL图: {out}")


@dataclass
class SelectedContracts:
    future_ts_code: str
    call_ts_code: str
    put_ts_code: str
    future_expiry_date: date
    option_expiry_date: date
    strike: float
    option_per_unit: float
    future_per_unit: float


TENOR_TO_YEAR = {
    "on": 1 / 365,
    "1w": 7 / 365,
    "2w": 14 / 365,
    "1m": 1 / 12,
    "3m": 3 / 12,
    "6m": 6 / 12,
    "9m": 9 / 12,
    "1y": 1.0,
}



class CrrModel:
    @staticmethod
    def price(
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        ttm: float,
        option_type: str,
        steps: int = 80,
    ) -> float:
        if spot <= 0 or strike <= 0 or ttm <= 0 or vol <= 0:
            return max(spot - strike, 0.0) if option_type == "C" else max(strike - spot, 0.0)

        dt = ttm / steps
        up = math.exp(vol * math.sqrt(dt))
        dn = 1.0 / up
        disc = math.exp(-rate * dt)
        p = (math.exp(rate * dt) - dn) / (up - dn)
        p = min(max(p, 1e-8), 1 - 1e-8)

        values = []
        for j in range(steps + 1):
            s_t = spot * (up**j) * (dn ** (steps - j))
            payoff = max(s_t - strike, 0.0) if option_type == "C" else max(strike - s_t, 0.0)
            values.append(payoff)

        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                values[j] = disc * (p * values[j + 1] + (1 - p) * values[j])
        return values[0]

    @classmethod
    def delta(
        cls,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        ttm: float,
        option_type: str,
        bump: float = 0.001,
        steps: int = 80,
    ) -> float:
        if spot <= 0:
            return 0.0
        up_spot = spot * (1 + bump)
        dn_spot = max(spot * (1 - bump), 1e-6)
        v_up = cls.price(up_spot, strike, rate, vol, ttm, option_type, steps=steps)
        v_dn = cls.price(dn_spot, strike, rate, vol, ttm, option_type, steps=steps)
        return (v_up - v_dn) / (up_spot - dn_spot)

    @classmethod
    def implied_vol(
        cls,
        market_price: float,
        spot: float,
        strike: float,
        rate: float,
        ttm: float,
        option_type: str,
        vol_low: float = 0.01,
        vol_high: float = 2.0,
        steps: int = 80,
        tol: float = 1e-5,
        max_iter: int = 80,
    ) -> float:
        if market_price <= 0 or spot <= 0 or strike <= 0 or ttm <= 0:
            return 0.2

        lo, hi = vol_low, vol_high
        if market_price <= cls.price(spot, strike, rate, lo, ttm, option_type, steps=steps):
            return lo
        if market_price >= cls.price(spot, strike, rate, hi, ttm, option_type, steps=steps):
            return hi

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            mid_price = cls.price(spot, strike, rate, mid, ttm, option_type, steps=steps)
            if abs(mid_price - market_price) < tol:
                return mid
            if mid_price > market_price:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


class ContractSelector:
    def __init__(
        self, 
        cfg_exp: ExperimentConfig, 
        fut_basic: pl.DataFrame, 
        opt_basic: pl.DataFrame,
        fut_bars: pl.DataFrame,
        opt_bars: pl.DataFrame,
    ):
        self.cfg_exp = cfg_exp
        self.fut_basic = fut_basic
        self.opt_basic = opt_basic
        self.fut_bars = fut_bars
        self.opt_bars = opt_bars
        self.selected_contract: SelectedContracts | None = None
    
    def select_contract(self, trade_date: date | datetime) -> SelectedContracts | None:
        td = trade_date.date() if isinstance(trade_date, datetime) else trade_date
        # 思路：非换月日沿用旧合约；换月日重新选“次月期货 + 次月期权ATM跨式”
        if not self._should_roll(td):
            return self.selected_contract

        # 期货合约信息
        fut_row = self._select_next_month_future(td)
        if fut_row.is_empty():
            return self.selected_contract

        future_info = fut_row.row(0, named=True)
        future_ts_code = future_info["ts_code"]
        future_per_unit = future_info.get("per_unit", 1.0)
        future_expiry_date = future_info.get("delist_date")

        spot = self._latest_close(self.fut_bars, future_ts_code, td)
        if spot <= 0:
            return self.selected_contract

        # 期货期权合约信息
        opts = self._active_option_pool(td, future_ts_code)
        if opts.is_empty():
            return self.selected_contract

        option_expiry_date = self._pick_next_month_option_expiry(opts)
        if option_expiry_date is None:
            return self.selected_contract
        opts = opts.filter(pl.col("delist_date") == option_expiry_date)
        if opts.is_empty():
            return self.selected_contract

        opts = opts.with_columns(
            (pl.col("exercise_price").cast(pl.Float64, strict=False) - spot).abs().alias("distance")
        )
        atm_strike_row = opts.sort("distance").row(0, named=True)
        if not atm_strike_row:
            return self.selected_contract
        atm_strike = float(atm_strike_row["exercise_price"])

        call_opts = opts.filter(
            (pl.col("exercise_price") == atm_strike)
            & (pl.col("call_put") == "C")
        )
        put_opts = opts.filter(
            (pl.col("exercise_price") == atm_strike)
            & (pl.col("call_put") == "P")
        )
        if call_opts.is_empty() or put_opts.is_empty():
            return self.selected_contract

        call = call_opts.row(0, named=True)
        put = put_opts.row(0, named=True)

        # 最小入侵修复：
        # 若当前持仓仍有效，且新候选仅是同到期月下的ATM执行价变化（同future+同expiry），
        # 则不触发换仓，避免在roll窗口内频繁“同月换strike”。
        if self.selected_contract is not None:
            current_active = self.opt_basic.filter(
                (pl.col("ts_code").is_in([self.selected_contract.call_ts_code, self.selected_contract.put_ts_code]))
                & (pl.col("list_date") <= td)
                & (pl.col("delist_date") >= td)
            ).height >= 2
            same_future = future_ts_code == self.selected_contract.future_ts_code
            same_option_expiry = option_expiry_date == self.selected_contract.option_expiry_date
            if current_active and same_future and same_option_expiry:
                return self.selected_contract

        self.selected_contract = SelectedContracts(
            future_ts_code=future_ts_code,
            call_ts_code=call["ts_code"],
            put_ts_code=put["ts_code"],
            future_expiry_date=future_expiry_date,
            option_expiry_date=option_expiry_date,
            strike=atm_strike,
            option_per_unit=call.get("per_unit", 1.0),
            future_per_unit=future_per_unit,
        )
        return self.selected_contract

    def _should_roll(self, trade_date: date) -> bool:
        if self.selected_contract is None:
            return True

        roll_days = self.cfg_exp.hedge.roll_days_before_maturity
        days_to_option_expiry = (self.selected_contract.option_expiry_date - trade_date).days
        # 按期货期权 delist_date 触发换月（期权通常早于期货）
        if days_to_option_expiry <= roll_days:
            return True

        active_opts = self.opt_basic.filter(
            (pl.col("ts_code").is_in([self.selected_contract.call_ts_code, self.selected_contract.put_ts_code]))
            & (pl.col("list_date") <= trade_date)
            & (pl.col("delist_date") >= trade_date)
        )
        return active_opts.height < 2

    def _select_next_month_future(self, trade_date: date) -> pl.DataFrame:
        active = self.fut_basic.filter(
            (pl.col("list_date") <= trade_date)
            & (pl.col("delist_date") >= trade_date)
        ).sort("delist_date")
        if active.is_empty():
            return pl.DataFrame()
        # 永远优先选次月（第2近）合约；若不足两个，则回退最近月。
        idx = 1 if active.height > 1 else 0
        return active.slice(idx, 1)

    def _active_option_pool(self, trade_date: date, future_ts_code: str) -> pl.DataFrame:
        opts = self.opt_basic.filter(
            (pl.col("list_date") <= trade_date)
            & (pl.col("delist_date") >= trade_date)
            & (pl.col("call_put").is_in(["C", "P"]))
        )
        if "opt_code" in opts.columns:
            by_code = opts.filter(pl.col("opt_code") == f"OP{future_ts_code}")
            if not by_code.is_empty():
                return by_code
        return opts

    @staticmethod
    def _pick_next_month_option_expiry(opts: pl.DataFrame) -> date | None:
        maturities = opts.select("delist_date").drop_nulls().unique().sort("delist_date")
        if maturities.is_empty():
            return None
        idx = 1 if maturities.height > 1 else 0
        return maturities.row(idx)[0]

    @staticmethod
    def _latest_close(df: pl.DataFrame, ts_code: str, trade_date: date) -> float:
        rows = (
            df.filter((pl.col("ts_code") == ts_code) & (pl.col("trade_date") == trade_date))
            .sort("bar_ts")
            .select("close")
        )
        if rows.is_empty():
            return 0.0
        return float(rows.tail(1).item())

    def build_trade_dates(self) -> list[date]:
        return (
            self.fut_bars.select("trade_date")
            .unique()
            .sort("trade_date")
            .to_series()
            .to_list()
        )
    
    @staticmethod
    def _with_bar_timeline(df: pl.DataFrame) -> pl.DataFrame:
        if "trade_time" in df.columns:
            return df.with_columns(pl.col("trade_time").alias("bar_ts"))
        return df.with_columns(
            pl.datetime(
                pl.col("trade_date").dt.year(),
                pl.col("trade_date").dt.month(),
                pl.col("trade_date").dt.day(),
                pl.lit(15),
                pl.lit(0),
                pl.lit(0),
            ).alias("bar_ts")
        )



class BacktestDataModule:
    """Data module: update local DB and provide aligned bars."""

    def __init__(self, cfg_exp: ExperimentConfig):
        self.cfg_exp = cfg_exp
        self.cfg_backtest = cfg_exp.backtest
        self.cfg_hedge = cfg_exp.hedge
        self.loader = OptionLoader(cfg_exp=cfg_exp)

        self.fut_basic = pl.DataFrame()
        self.opt_basic = pl.DataFrame()
        self.fut_bars = pl.DataFrame()
        self.opt_bars = pl.DataFrame()
        self.shibor_daily = pl.DataFrame()
        self.trade_dates: list[date] = []

    def adddata(self) -> None:
        log_info(f"开始准备{self.cfg_hedge.frequency}频率的期货、期权、市场利率数据")
        self.loader.update_future_price_to_local_db()
        self.loader.update_option_price_to_local_db()
        log_info("期货、期权、市场利率数据准备完成")

        self.fut_basic = self.loader.load_future_basic(transform_date=True)
        self.opt_basic = self.loader.load_option_basic(transform_date=True)
        self.fut_bars, self.opt_bars = self._load_bar_data()
        self.shibor_daily = self.loader.load_shibor_daily()
        self.trade_dates = (
            self.fut_bars.select("trade_date").unique().sort("trade_date").to_series().to_list()
            if not self.fut_bars.is_empty()
            else []
        )

    def _load_bar_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        freq = self.cfg_hedge.frequency
        fut_contracts = self.fut_basic.filter(
            (pl.col("list_date") <= self.cfg_backtest.end_date)
            & (pl.col("delist_date") >= self.cfg_backtest.start_date)
        )
        opt_contracts = self.opt_basic.filter(
            (pl.col("list_date") <= self.cfg_backtest.end_date)
            & (pl.col("delist_date") >= self.cfg_backtest.start_date)
            & (pl.col("call_put").is_in(["C", "P"]))
        )

        fut_frames: list[pl.DataFrame] = []
        for row in fut_contracts.iter_rows(named=True):
            ts_code = row["ts_code"]
            try:
                df = self.loader.load_future_bar(ts_code=ts_code, freq=freq).with_columns(pl.lit(ts_code).alias("ts_code"))
            except FileNotFoundError:
                continue
            fut_frames.append(df)
        fut_bars = pl.concat(fut_frames, how="vertical_relaxed") if fut_frames else pl.DataFrame()

        opt_frames: list[pl.DataFrame] = []
        for row in opt_contracts.iter_rows(named=True):
            ts_code = row["ts_code"]
            try:
                df = self.loader.load_option_bar(ts_code=ts_code, freq=freq).with_columns(pl.lit(ts_code).alias("ts_code"))
            except FileNotFoundError:
                continue
            opt_frames.append(df)
        opt_bars = pl.concat(opt_frames, how="vertical_relaxed") if opt_frames else pl.DataFrame()

        if not fut_bars.is_empty():
            fut_bars = fut_bars.filter(
                (pl.col("trade_date") >= self.cfg_backtest.start_date)
                & (pl.col("trade_date") <= self.cfg_backtest.end_date)
            )
            fut_bars = ContractSelector._with_bar_timeline(fut_bars)
        if not opt_bars.is_empty():
            opt_bars = opt_bars.filter(
                (pl.col("trade_date") >= self.cfg_backtest.start_date)
                & (pl.col("trade_date") <= self.cfg_backtest.end_date)
            )
            opt_bars = ContractSelector._with_bar_timeline(opt_bars)

        log_info(f"加载{freq}频率的期货、期权行情数据完成")
        return fut_bars, opt_bars

    def get_day_bars(self, selected: SelectedContracts, trade_date: date) -> pl.DataFrame:
        fut_day = (
            self.fut_bars.filter((pl.col("ts_code") == selected.future_ts_code) & (pl.col("trade_date") == trade_date))
            .select(["bar_ts", "open", "close"])
            .rename({"open": "future_open", "close": "future_close"})
        )
        call_day = (
            self.opt_bars.filter((pl.col("ts_code") == selected.call_ts_code) & (pl.col("trade_date") == trade_date))
            .select(["bar_ts", "close"])
            .rename({"close": "call_close"})
        )
        put_day = (
            self.opt_bars.filter((pl.col("ts_code") == selected.put_ts_code) & (pl.col("trade_date") == trade_date))
            .select(["bar_ts", "close"])
            .rename({"close": "put_close"})
        )
        if fut_day.is_empty() or call_day.is_empty() or put_day.is_empty():
            return pl.DataFrame()
        return fut_day.join(call_day, on="bar_ts", how="inner").join(put_day, on="bar_ts", how="inner").sort("bar_ts")


class DeltaHedgeStrategy:
    """Strategy module: produce target hedge position from bars."""

    def __init__(self, cfg_exp: ExperimentConfig):
        self.cfg_hedge = cfg_exp.hedge
        self.iv_cache_by_option: dict[str, float] = {}

    def on_bar(
        self,
        selected: SelectedContracts,
        trade_date: date,
        bar: dict,
        rate: float,
    ) -> dict | None:
        fut_open = _safe_float(bar["future_open"])
        fut_px = _safe_float(bar["future_close"])
        call_px = _safe_float(bar["call_close"])
        put_px = _safe_float(bar["put_close"])
        if fut_open <= 0:
            fut_open = fut_px
        if fut_px <= 0 or fut_open <= 0:
            return None

        ttm = _year_fraction(selected.option_expiry_date, trade_date)
        iv_call = self.iv_cache_by_option.get(selected.call_ts_code, 0.2)
        iv_put = self.iv_cache_by_option.get(selected.put_ts_code, 0.2)
        # 对冲目标用bar open的期货价格计算，匹配开仓对冲时点
        delta_call = CrrModel.delta(fut_open, selected.strike, rate, iv_call, ttm, "C")
        delta_put = CrrModel.delta(fut_open, selected.strike, rate, iv_put, ttm, "P")
        combo_delta = delta_call + delta_put

        scale = 1.0
        if self.cfg_hedge.use_contract_unit:
            scale = selected.option_per_unit / max(selected.future_per_unit, 1e-9)
        target_pos = -combo_delta * scale

        return {
            "trade_date": trade_date,
            "bar_ts": bar["bar_ts"],
            "future_open": fut_open,
            "future_close": fut_px,
            "call_close": call_px,
            "put_close": put_px,
            "rate": rate,
            "ttm": ttm,
            "iv_call_used": iv_call,
            "iv_put_used": iv_put,
            "delta_call": delta_call,
            "delta_put": delta_put,
            "combo_delta": combo_delta,
            "target_future_position": target_pos,
        }

    def on_day_close(self, selected: SelectedContracts, trade_date: date, last_bar: dict, rate: float) -> None:
        spot = _safe_float(last_bar["future_close"])
        if spot <= 0:
            return
        ttm = _year_fraction(selected.option_expiry_date, trade_date)
        call_px = _safe_float(last_bar["call_close"])
        put_px = _safe_float(last_bar["put_close"])
        self.iv_cache_by_option[selected.call_ts_code] = CrrModel.implied_vol(call_px, spot, selected.strike, rate, ttm, "C")
        self.iv_cache_by_option[selected.put_ts_code] = CrrModel.implied_vol(put_px, spot, selected.strike, rate, ttm, "P")


class BrokerEngine:
    """Execution & book module: convert target position into trades."""

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        fee_rate: float = 0.00005,
        slippage_bps: float = 1.0,
        straddle_size: float = 1.0,
        margin_rate: float = 0.15, 
    ):
        self.init_cash = init_cash
        self.cash = init_cash
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.straddle_size = straddle_size
        self.margin_rate = margin_rate
        # futures
        self.prev_fut_px: float = 0.0
        self.current_future_code: str | None = None
        self.current_future_per_unit: float = 1.0
        self.current_future_position = 0.0
        # options
        self.option_position_call = 0.0
        self.option_position_put = 0.0
        self.current_call_code: str | None = None
        self.current_put_code: str | None = None
        self.prev_call_px: float | None = None
        self.prev_put_px: float | None = None

        self.records: list[dict] = []
        self.prev_nav: float | None = None

    def on_decision(self, selected: SelectedContracts, decision: dict) -> None:
        fut_open = _safe_float(decision["future_open"])
        fut_px = _safe_float(decision["future_close"])
        if fut_open <= 0:
            fut_open = fut_px
        call_px = _safe_float(decision["call_close"])
        put_px = _safe_float(decision["put_close"])
        option_unit = selected.option_per_unit

        # =========================
        # 1 FUTURES OVERNIGHT MTM TO BAR OPEN
        # =========================
        future_contract_changed = (
            self.current_future_code is not None
            and self.current_future_code != selected.future_ts_code
        )
        future_overnight_pnl = 0.0
        if self.prev_fut_px != 0 and not future_contract_changed:
            future_overnight_pnl = (
                self.current_future_position
                * (fut_open - self.prev_fut_px)
                * selected.future_per_unit
            )
            self.cash += future_overnight_pnl

        # 合约切换时先平旧期货仓位，避免跨合约价格直接做隔夜PnL
        roll_close_qty = 0.0
        roll_close_price = 0.0
        roll_close_fee = 0.0
        if future_contract_changed and self.current_future_position != 0:
            roll_close_qty = -self.current_future_position
            roll_close_price = self.prev_fut_px if self.prev_fut_px > 0 else fut_open
            roll_close_fee = (
                abs(roll_close_qty)
                * roll_close_price
                * self.current_future_per_unit
                * self.fee_rate
            )
            self.cash -= roll_close_fee
            self.current_future_position = 0.0
        
        # =========================
        # 2 OPTION LEG PNL (mark-to-market, no pnl on roll switch)
        # =========================
        call_leg_pnl = 0.0
        put_leg_pnl = 0.0
        if self.current_call_code == selected.call_ts_code and self.prev_call_px is not None:
            call_leg_pnl = (
                (call_px - self.prev_call_px)
                * self.option_position_call
                * option_unit
            )
        if self.current_put_code == selected.put_ts_code and self.prev_put_px is not None:
            put_leg_pnl = (
                (put_px - self.prev_put_px)
                * self.option_position_put
                * option_unit
            )
        option_combo_pnl = call_leg_pnl + put_leg_pnl

        # =========================
        # 3 OPTION ROLL CHECK
        # =========================
        option_changed = (
            self.current_call_code != selected.call_ts_code
            or self.current_put_code != selected.put_ts_code
        )
        if option_changed:
            # close old options
            if self.option_position_call != 0 and self.prev_call_px is not None:
                self.cash += (
                    self.prev_call_px
                    * self.option_position_call
                    * option_unit
                )
            if self.option_position_put != 0 and self.prev_put_px is not None:
                self.cash += (
                    self.prev_put_px
                    * self.option_position_put
                    * option_unit
                )
            # open new options (1 call + 1 put)
            size = self.straddle_size
            self.cash -= call_px * size * option_unit
            self.cash -= put_px * size * option_unit
            self.option_position_call = size
            self.option_position_put = size
            self.current_call_code = selected.call_ts_code
            self.current_put_code = selected.put_ts_code

        # =========================
        # 4 FUTURE HEDGE TRADE
        # =========================
        target_pos = _safe_float(decision["target_future_position"]) * self.straddle_size
        trade_qty = target_pos - self.current_future_position
        trade_sign = 1.0 if trade_qty > 0 else -1.0 if trade_qty < 0 else 0.0
        # 对冲交易发生在bar open
        exec_price = fut_open * (1 + trade_sign * self.slippage_bps / 10000.0)
        trade_notional = trade_qty * exec_price * selected.future_per_unit
        fee = abs(trade_qty) * exec_price * selected.future_per_unit * self.fee_rate
        slippage_cost = trade_qty * (exec_price - fut_open) * selected.future_per_unit

        self.cash -= slippage_cost
        self.cash -= fee
        self.current_future_position = target_pos
        self.current_future_code = selected.future_ts_code
        self.current_future_per_unit = selected.future_per_unit

        # bar内期货持仓从open持有到close的PnL
        future_intrabar_pnl = (
            self.current_future_position
            * (fut_px - fut_open)
            * selected.future_per_unit
        )
        self.cash += future_intrabar_pnl
        future_leg_pnl = future_overnight_pnl + future_intrabar_pnl

        # =========================
        # 5 OPTION MARKET VALUE
        # =========================
        option_market_value = (
            call_px * self.option_position_call * option_unit
            + put_px * self.option_position_put * option_unit
        )

        # =========================
        # FUTURE MARGIN
        # =========================
        future_margin = (
            abs(self.current_future_position)
            * fut_px
            * selected.future_per_unit
            * self.margin_rate
        )
        # =========================
        # CAPITAL USED
        # =========================
        capital_used = option_market_value + future_margin

        # =========================
        # 6 NAV
        # =========================
        # futures already marked to market
        nav = self.cash + option_market_value
        pnl = 0.0 if self.prev_nav is None else nav - self.prev_nav
        self.prev_nav = nav
        trading_cost = fee + slippage_cost + roll_close_fee

        # 更新价格缓存
        self.prev_fut_px = fut_px
        self.prev_call_px = call_px
        self.prev_put_px = put_px

        # =================================
        # 7 RECORD
        # =================================
        future_notional = (
            self.current_future_position
            * fut_px
            * selected.future_per_unit
        )

        self.records.append(
            {
                **decision,
                # contract info
                "future_ts_code": selected.future_ts_code,
                "call_ts_code": selected.call_ts_code,
                "put_ts_code": selected.put_ts_code,
                # position info
                "current_future_position": self.current_future_position,
                "option_position_call": self.option_position_call,
                "option_position_put": self.option_position_put,
                # price info
                "future_open": fut_open,
                "future_close": fut_px,
                "call_close": call_px,
                "put_close": put_px,
                # pnl info
                "cash": self.cash,
                "nav": nav,
                "pnl": pnl,
                "future_pnl": future_leg_pnl,
                "future_overnight_pnl": future_overnight_pnl,
                "future_intrabar_pnl": future_intrabar_pnl,
                "option_pnl": option_combo_pnl,
                "call_pnl": call_leg_pnl,
                "put_pnl": put_leg_pnl,
                "slippage_cost": slippage_cost,
                "trading_cost": trading_cost,
                "explained_pnl": future_leg_pnl + option_combo_pnl - trading_cost,                
                # trade info
                "trade_qty": trade_qty,
                "exec_price": exec_price,
                "trade_notional": trade_notional,
                "fee": fee,
                "future_contract_changed": future_contract_changed,
                "roll_close_qty": roll_close_qty,
                "roll_close_price": roll_close_price,
                "roll_close_fee": roll_close_fee,
                # notionals
                "future_market_value": future_notional,
                "option_market_value": option_market_value,
                "capital_used": capital_used,
            }
        )

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.records) if self.records else pl.DataFrame()

    def to_daily_frame(self) -> pl.DataFrame:
        detail = self.to_frame()
        if detail.is_empty():
            return pl.DataFrame()
        daily = (
            detail.sort("bar_ts")
            .group_by("trade_date")
            .agg(
                # contract info
                pl.col("future_ts_code").last().alias("future_ts_code"),
                pl.col("call_ts_code").last().alias("call_ts_code"),
                pl.col("put_ts_code").last().alias("put_ts_code"),
                pl.col("nav").last().alias("nav"),
                pl.col("pnl").sum().alias("daily_pnl"),
                pl.col("fee").sum().alias("daily_fee"),
                pl.col("trade_qty").abs().sum().alias("daily_turnover_qty"),
                pl.col("cash").last().alias("cash"),
                pl.col("future_market_value").last().alias("future_market_value"),
                pl.col("option_market_value").last().alias("option_market_value"),
                pl.col("capital_used").last().alias("capital_used"),
                pl.col("current_future_position").last().alias("future_position"),
                # pnl aliases
                pl.col("future_pnl").sum().alias("daily_future_pnl"),
                pl.col("call_pnl").sum().alias("daily_call_pnl"),
                pl.col("put_pnl").sum().alias("daily_put_pnl"),
                pl.col("option_pnl").sum().alias("daily_option_pnl"),
                pl.col("slippage_cost").sum().alias("daily_slippage_cost"),
                pl.col("trading_cost").sum().alias("daily_trading_cost"),
                pl.col("explained_pnl").sum().alias("daily_explained_pnl"),
            )
            .sort("trade_date")
            .with_columns(
                ((pl.col("nav") / pl.lit(self.init_cash))).alias("nav_ratio"),
                (
                    pl.col("daily_pnl").cum_sum()
                    / pl.col("capital_used").clip(lower_bound=1e-6)
                ).alias("strategy_return"),
                (pl.col("capital_used") / pl.col("nav").clip(lower_bound=1e-6)).alias("capital_usage"),
            )
        )
        return daily


class DeltaHedgeRunner:
    """Orchestrator: data -> strategy -> broker."""

    def __init__(self, cfg_exp: ExperimentConfig, strategy_name: str):
        self.strategy_name = strategy_name
        self.cfg_exp = cfg_exp
        self.data_module = BacktestDataModule(cfg_exp)
        self.strategy = DeltaHedgeStrategy(cfg_exp)
        self.broker = BrokerEngine(straddle_size=cfg_exp.hedge.straddle_size)
        self.backtest_result = pl.DataFrame()
        self.daily_result = pl.DataFrame()

    def run(self) -> pl.DataFrame:
        self.data_module.adddata()
        risk_rater = RiskFreeRate(interest_df=self.data_module.shibor_daily)
        selector = ContractSelector(
            cfg_exp=self.cfg_exp,
            fut_basic=self.data_module.fut_basic,
            opt_basic=self.data_module.opt_basic,
            fut_bars=self.data_module.fut_bars,
            opt_bars=self.data_module.opt_bars,
        )

        log_info("开始回测")
        for trade_date in self.data_module.trade_dates:
            selected = selector.select_contract(trade_date)
            
            if selected is None:
                continue
            day_bars = self.data_module.get_day_bars(selected, trade_date)
            if day_bars.is_empty():
                continue

            rate = risk_rater.get_rate(trade_date, _year_fraction(selected.option_expiry_date, trade_date))
            for bar in day_bars.iter_rows(named=True):
                decision = self.strategy.on_bar(selected, trade_date, bar, rate)
                if decision is None:
                    continue
                self.broker.on_decision(selected, decision)

            self.strategy.on_day_close(selected, trade_date, day_bars.tail(1).row(0, named=True), rate)

        self.backtest_result = self.broker.to_frame()
        self.daily_result = self.broker.to_daily_frame()
        log_info("回测完成")
        return self.backtest_result


class RiskFreeRate:
    def __init__(self, interest_df: pl.DataFrame):
        self.interest_rate_df = interest_df
    # =========================================================
    # 利率曲线（插值）
    # =========================================================
    def _get_rate_curve(self, trade_date: date):
        try:
            # 过滤有效行
            rate_rows = self.interest_rate_df.filter(pl.col("trade_date") == trade_date).to_dicts()
            if len(rate_rows) == 0:
                return np.array([]), np.array([])
            
            row = rate_rows[0]
            maturities = []
            rates = []

            for tenor, T in TENOR_TO_YEAR.items():
                if tenor in row and row[tenor] is not None and isinstance(row[tenor], (int, float)):
                    maturities.append(T)
                    rates.append(row[tenor] / 100)  # 转换为小数（如 2.5% → 0.025）

            return np.array(maturities), np.array(rates)
        except Exception:
            return np.array([]), np.array([])

    def _interpolate_rate(self, trade_date: date, T: float) -> float:
        maturities, rates = self._get_rate_curve(trade_date)
        # 处理空值/无效值
        if len(maturities) == 0 or len(rates) == 0:
            return 0.02  # 默认利率（可根据需求调整）
        if T <= maturities.min():
            return rates[0]
        if T >= maturities.max():
            return rates[-1]
        r = np.interp(T, maturities, rates)
        return float(r)

    def get_rate(self, trade_date: date, T: float) -> float:
        return self._interpolate_rate(trade_date, T)




if __name__ == '__main__':
    from core.config_shema import load_config
    cfg_exp = load_config('config/shfe_ag_demo.yaml')[0]
    delta_hedge_runner = DeltaHedgeRunner(cfg_exp=cfg_exp, strategy_name='delta_hedge')
    # 先看选约逻辑:
    # delta_hedge_runner.demo_contract_selector(hedge_qty=0.0, max_days=30)
    # 再跑完整三模块回测:
    backtest_result = delta_hedge_runner.run()
    print(backtest_result)
    print(delta_hedge_runner.daily_result)
    daily_position = (backtest_result
        .sort("bar_ts")
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
    print(daily_position)
    plot_daily_pnl(
        delta_hedge_runner.daily_result, 
        output_path=f"outputs/daily_pnl_{cfg_exp.backtest.start_date}_{cfg_exp.backtest.end_date}.png", 
        show=False)