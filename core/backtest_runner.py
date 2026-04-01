


from typing import Literal
import polars as pl
from dataclasses import asdict, dataclass
from datetime import date
from datetime import datetime, time, timedelta
import numpy as np
import math

from core.config_shema import ExperimentConfig
from data_fetcher.option_loader import OptionLoader
from utils.logger import log_info
from utils.tools import _safe_float, _year_fraction


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

@dataclass
class StrategyBarHedgeSignal:
    trade_date: date
    bar_ts: datetime
    future_ts_code: str
    future_close: float
    future_per_unit: float
    call_ts_code: str
    call_close: float
    call_settle: float
    put_ts_code: str
    put_close: float
    put_settle: float
    option_per_unit: float
    rate: float
    ttm: float
    iv_call_used: float
    iv_put_used: float
    delta_call: float
    delta_put: float
    combo_delta: float
    target_future_position: float
    gamma_call: float
    gamma_put: float
    vega_call: float
    vega_put: float
    theta_call: float
    theta_put: float

@dataclass
class BrokerRecord:
    trade_date: date
    bar_ts: datetime
    nav: float
    margin_by_future: float
    cash: float
    pnl: float
    pnl_future: float
    pnl_call: float
    pnl_put: float
    pnl_option: float
    cash_future: float
    cash_call: float
    cash_put: float
    cash_option: float
    nav_future: float
    nav_call: float
    nav_put: float
    nav_option: float
    # capital_used
    capital_used: float
    capital_future: float
    capital_call: float
    capital_put: float
    capital_option: float
    # future
    future_ts_code: str
    # option
    call_ts_code: str
    put_ts_code: str
    # fee
    fee: float
    fee_future: float
    fee_call: float
    fee_put: float
    fee_option: float
    slippage: float
    theta_loss: float
    name: str=Literal["open", "mtm", "close"]


@dataclass
class BacktestRecord:
    bar_signal: StrategyBarHedgeSignal
    broker_record: BrokerRecord

    def flatten(self) -> dict:
        d = {}
        decision_dict = asdict(self.bar_signal)
        broker_dict = asdict(self.broker_record)

        d.update(decision_dict)
        d.update(broker_dict)
        
        return d



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
        self.fut_bars, self.opt_bars = self.load_bar_data()
        self.fut_settle_bars, self.opt_settle_bars = self._load_bar_data_by_freq(freq="1d")
        self.shibor_daily = self.loader.load_shibor_daily()
        self.trade_dates = (
            self.fut_bars.select("trade_date").unique().sort("trade_date").to_series().to_list()
            if not self.fut_bars.is_empty()
            else []
        )

    def get_day_bars(self, selected: SelectedContracts, trade_date: date, use_settle_col: bool = False) -> pl.DataFrame:
        if use_settle_col:
            # 使用日线结算价数据
            fut_day = (
                self.fut_settle_bars.filter(
                    (pl.col("ts_code") == selected.future_ts_code) & (pl.col("trade_date") == trade_date)
                )
                .select(["bar_ts", "settle"])
                .rename({"settle": "future_settle"})
            )
            call_day = (
                self.opt_settle_bars.filter(
                    (pl.col("ts_code") == selected.call_ts_code) & (pl.col("trade_date") == trade_date)
                )
                .select(["bar_ts", "settle"])
                .rename({"settle": "call_settle"})
            )
            put_day = (
                self.opt_settle_bars.filter(
                    (pl.col("ts_code") == selected.put_ts_code) & (pl.col("trade_date") == trade_date)
                )
                .select(["bar_ts", "settle"])
                .rename({"settle": "put_settle"})
            )
        else:
            # 使用高频行情数据
            close_col = "close"
            fut_day = (
                self.fut_bars.filter(
                    (pl.col("ts_code") == selected.future_ts_code) & (pl.col("trade_date") == trade_date)
                )
                .select(["bar_ts", "open", close_col])
                .rename({"open": "future_open", close_col: "future_close"})
            )
            call_day = (
                self.opt_bars.filter(
                    (pl.col("ts_code") == selected.call_ts_code) & (pl.col("trade_date") == trade_date)
                )
                .select(["bar_ts", close_col])
                .rename({close_col: "call_close"})
            )
            put_day = (
                self.opt_bars.filter(
                    (pl.col("ts_code") == selected.put_ts_code) & (pl.col("trade_date") == trade_date)
                )
                .select(["bar_ts", close_col])
                .rename({close_col: "put_close"})
            )

        if fut_day.is_empty() or call_day.is_empty() or put_day.is_empty():
            return pl.DataFrame()
        
        return fut_day.join(call_day, on="bar_ts", how="inner").join(put_day, on="bar_ts", how="inner").sort("bar_ts")

    # 输入ts_code和trade_date，返回期货或者call或者put的一根bar
    def get_bar(
        self, 
        ts_code: str, 
        bar_ts: datetime, 
        instrument_type: Literal["future", "call", "put"]
    ) -> pl.DataFrame:

        if instrument_type == "future":
            bar_ts = self.fut_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).select(
                "bar_ts", "open", "close"
            ).sort("bar_ts")
        elif instrument_type == "call":
            # return self.opt_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).sort("bar_ts")
            bar_ts = self.opt_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).select(
                "bar_ts", "close"
            ).sort("bar_ts")
        elif instrument_type == "put":
            # return self.opt_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).sort("bar_ts")
            bar_ts = self.opt_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).select(
                "bar_ts", "close"
            ).sort("bar_ts")
        else:
            raise ValueError(f"Invalid instrument type: {instrument_type}")
        return bar_ts

    def load_bar_data(self):
        freq = self.cfg_hedge.frequency
        fut_bars, opt_bars = self._load_bar_data_by_freq(freq=freq)
        return fut_bars, opt_bars

    def _load_bar_data_by_freq(self, freq=Literal["1d", "1min", "5min", "15min", "30min", "60min"]) -> tuple[pl.DataFrame, pl.DataFrame]:
        
        start_filter = self.cfg_backtest.start_date - timedelta(days=5)
        
        fut_contracts = self.fut_basic.filter(
            (pl.col("list_date") <= self.cfg_backtest.end_date)
            & (pl.col("delist_date") >= start_filter)
        )
        opt_contracts = self.opt_basic.filter(
            (pl.col("list_date") <= self.cfg_backtest.end_date)
            & (pl.col("delist_date") >= start_filter)
            & (pl.col("call_put").is_in(["C", "P"]))
        )
        # load future bars
        fut_frames: list[pl.DataFrame] = []
        for row in fut_contracts.iter_rows(named=True):
            ts_code = row["ts_code"]
            try:
                df = self.loader.load_future_bar(ts_code=ts_code, freq=freq).with_columns(pl.lit(ts_code).alias("ts_code"))
            except FileNotFoundError:
                continue
            fut_frames.append(df)
        fut_bars = pl.concat(fut_frames, how="vertical_relaxed") if fut_frames else pl.DataFrame()

        if not fut_bars.is_empty():
            fut_bars = fut_bars.filter(
                (pl.col("trade_date") >= start_filter)
                & (pl.col("trade_date") <= self.cfg_backtest.end_date)
            )
            fut_bars = ContractSelector._with_bar_timeline(fut_bars)
        
        # load option bars
        opt_frames: list[pl.DataFrame] = []
        for row in opt_contracts.iter_rows(named=True):
            ts_code = row["ts_code"]
            try:
                df = self.loader.load_option_bar(ts_code=ts_code, freq=freq).with_columns(pl.lit(ts_code).alias("ts_code"))
            except FileNotFoundError:
                continue
            opt_frames.append(df)
        opt_bars = pl.concat(opt_frames, how="vertical_relaxed") if opt_frames else pl.DataFrame()

        if not opt_bars.is_empty():
            opt_bars = opt_bars.filter(
                (pl.col("trade_date") >= start_filter)
                & (pl.col("trade_date") <= self.cfg_backtest.end_date)
            )
            opt_bars = ContractSelector._with_bar_timeline(opt_bars)

        log_info(f"加载{freq}频率的期货、期权行情数据完成")
        return fut_bars, opt_bars


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
        steps: int = 100,
        tol: float = 1e-6,
        max_iter: int = 100,
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

    @classmethod
    def gamma(
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
        # Gamma = d^2V/dS^2, centered difference approx
        if spot <= 0:
            return 0.0
        up_spot = spot * (1 + bump)
        dn_spot = max(spot * (1 - bump), 1e-6)
        v_up = cls.price(up_spot, strike, rate, vol, ttm, option_type, steps=steps)
        v_mid = cls.price(spot, strike, rate, vol, ttm, option_type, steps=steps)
        v_dn = cls.price(dn_spot, strike, rate, vol, ttm, option_type, steps=steps)
        return (v_up - 2 * v_mid + v_dn) / ((up_spot - spot) * (spot - dn_spot))

    @classmethod
    def vega(
        cls,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        ttm: float,
        option_type: str,
        bump: float = 0.001,  # 相对变动
        steps: int = 80,
        unit: str = "percent"  # "percent" 或 "unit"
    ) -> float:
        """
        Calculate vega (dV/dσ).
        
        Parameters
        ----------
        unit : str
            "percent" - return vega per 1% change in volatility (industry standard)
            "unit" - return vega per 100% change in volatility (raw derivative)
        """
        if vol <= 0 or spot <= 0:
            return 0.0
        
        up_vol = vol * (1 + bump)
        dn_vol = max(vol * (1 - bump), 1e-6)
        v_up = cls.price(spot, strike, rate, up_vol, ttm, option_type, steps=steps)
        v_dn = cls.price(spot, strike, rate, dn_vol, ttm, option_type, steps=steps)
        
        # 原始vega：每100%波动率变化的价格变化
        vega_raw = (v_up - v_dn) / (up_vol - dn_vol)
        
        # 根据单位转换
        if unit.lower() == "percent":
            return vega_raw * 0.01  # 转换为每1%波动率变化
        else:  # "unit"
            return vega_raw

    @classmethod
    def theta(
        cls,
        spot: float,
        strike: float,
        rate: float,
        vol: float,
        ttm: float,
        option_type: str,
        days_bump: float = 1 / 252,
        steps: int = 80,
    ) -> float:
        # Theta = -dV/dt, dt in years. Default days_bump=1/252 for "per day"
        if ttm <= 0 or spot <= 0:
            return 0.0
        ttm_minus = max(ttm - days_bump, 1e-6)
        v_now = cls.price(spot, strike, rate, vol, ttm, option_type, steps=steps)
        v_next = cls.price(spot, strike, rate, vol, ttm_minus, option_type, steps=steps)
        return v_next - v_now


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
    ) -> StrategyBarHedgeSignal | None:
        fut_px = _safe_float(bar["future_close"])
        call_px = _safe_float(bar["call_close"])
        put_px = _safe_float(bar["put_close"])
        if fut_px <= 0:
            return None

        option_expiry_datetime = datetime.combine(selected.option_expiry_date, time(15, 0, 0))
        ttm = _year_fraction(option_expiry_datetime, bar["bar_ts"])       
        iv_call = self.iv_cache_by_option.get(selected.call_ts_code, 0.2)
        iv_put = self.iv_cache_by_option.get(selected.put_ts_code, 0.2)
        # 对冲目标用bar open的期货价格计算，匹配开仓对冲时点
        delta_call = CrrModel.delta(fut_px, selected.strike, rate, iv_call, ttm, "C")
        gamma_call = CrrModel.gamma(fut_px, selected.strike, rate, iv_call, ttm, "C")
        vega_call = CrrModel.vega(fut_px, selected.strike, rate, iv_call, ttm, "C")
        theta_call = CrrModel.theta(fut_px, selected.strike, rate, iv_call, ttm, "C")
        delta_put = CrrModel.delta(fut_px, selected.strike, rate, iv_put, ttm, "P")
        gamma_put = CrrModel.gamma(fut_px, selected.strike, rate, iv_put, ttm, "P")
        vega_put = CrrModel.vega(fut_px, selected.strike, rate, iv_put, ttm, "P")
        theta_put = CrrModel.theta(fut_px, selected.strike, rate, iv_put, ttm, "P")
        combo_delta = delta_call + delta_put

        # option settle price 
        call_settle = CrrModel.price(fut_px, selected.strike, rate, iv_call, ttm, "C")
        put_settle = CrrModel.price(fut_px, selected.strike, rate, iv_put, ttm, "P")

        scale = 1.0
        if self.cfg_hedge.use_contract_unit:
            scale = selected.option_per_unit / max(selected.future_per_unit, 1e-9)
        target_pos = -combo_delta * scale

        strategy_hedge_signal = StrategyBarHedgeSignal(
            trade_date=trade_date,
            bar_ts=bar["bar_ts"],
            future_ts_code=selected.future_ts_code,
            future_close=fut_px,
            future_per_unit=selected.future_per_unit,
            call_ts_code=selected.call_ts_code,
            call_close=call_px,
            call_settle=call_settle,
            put_ts_code=selected.put_ts_code,
            put_close=put_px,
            put_settle=put_settle,
            option_per_unit=selected.option_per_unit,
            rate=rate,
            ttm=ttm,
            iv_call_used=iv_call,
            iv_put_used=iv_put,
            delta_call=delta_call,
            delta_put=delta_put,
            combo_delta=combo_delta,
            target_future_position=target_pos,
            gamma_call=gamma_call,
            gamma_put=gamma_put,
            vega_call=vega_call,
            vega_put=vega_put,
            theta_call=theta_call,
            theta_put=theta_put,
        )
        return strategy_hedge_signal

    def on_day_close(self, selected: SelectedContracts, trade_date: date, last_bar: dict, rate: float) -> None:
        spot = _safe_float(last_bar["future_settle"])
        if spot <= 0:
            return
        ttm = _year_fraction(selected.option_expiry_date, trade_date)
        call_px = _safe_float(last_bar["call_settle"])
        put_px = _safe_float(last_bar["put_settle"])
        self.iv_cache_by_option[selected.call_ts_code] = CrrModel.implied_vol(call_px, spot, selected.strike, rate, ttm, "C")
        self.iv_cache_by_option[selected.put_ts_code] = CrrModel.implied_vol(put_px, spot, selected.strike, rate, ttm, "P")


class BrokerEngine:
    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        fee_rate: float = 0.0005,
        slippage_bps: float = 1.0,
        straddle_size: float = 1.0,
        margin_rate: float = 0.15,
        data_loader: BacktestDataModule = None,
        strategy: DeltaHedgeStrategy = None,
    ):
        self.init_cash = init_cash
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.straddle_size = straddle_size
        self.margin_rate = margin_rate
        self.cash = init_cash
        self.nav = init_cash
        self.data_loader = data_loader
        self.strategy = strategy
        # futures
        self.current_future_code: str | None = None
        self.future_per_unit: float = 15.0 
        self.current_future_position = 0.0 
        self.margin_by_future: float = 0.0
        # options 
        self.option_position_call = 0.0
        self.option_position_put = 0.0
        self.option_per_unit: float = 15.0
        self.current_call_close: float | None = None
        self.current_put_close: float | None = None
        # records
        self.records: list[BrokerRecord] = []

        self.prev_decision: StrategyBarHedgeSignal | None = None
        self.current_decision: StrategyBarHedgeSignal | None = None
        self.prev_record: BrokerRecord | None = None

    def on_decision(self, decision: StrategyBarHedgeSignal, prev_decision_update_to_current_bar_ts: StrategyBarHedgeSignal=None) -> None:
        self.current_decision = decision
        if self.prev_decision is None:
            record = self.on_decision_with_open_pos(self.current_decision)
        elif (
            self.prev_decision.future_ts_code != self.current_decision.future_ts_code and
            self.prev_decision.call_ts_code != self.current_decision.call_ts_code and
            self.prev_decision.put_ts_code != self.current_decision.put_ts_code
        ):
            record = self.on_decision_with_close_pos(prev_decision_update_to_current_bar_ts)
            record = self.on_decision_with_open_pos(self.current_decision)
        else:
            record = self.on_decision_with_mtm(self.current_decision)
        self.prev_decision = decision
        self.prev_record = record

    def on_decision_with_open_pos(self, decision: StrategyBarHedgeSignal) -> BrokerRecord:
        self.current_future_code = decision.future_ts_code
        self.future_per_unit = decision.future_per_unit
        # ======================== open futures ========================
        fut_px = _safe_float(decision.future_close)
        if fut_px <= 0:
            return None
        target_pos = _safe_float(decision.target_future_position) * self.straddle_size
        trade_qty = target_pos - self.current_future_position
        trade_sign = 1.0 if trade_qty > 0 else -1.0 if trade_qty < 0 else 0.0
        exec_price = fut_px * (1 + trade_sign * self.slippage_bps / 10000.0)
        
        # future margin 
        future_margin = abs(trade_qty) * exec_price * self.future_per_unit * self.margin_rate 
        self.cash -= future_margin
        self.margin_by_future = future_margin

        # future fee 
        fut_fee = abs(trade_qty) * exec_price * self.future_per_unit * self.fee_rate
        self.cash -= fut_fee

        # future slippage_cost only for record
        slippage_cost = abs(trade_qty) * abs(exec_price - fut_px) * self.future_per_unit

        future_delta_nav = - fut_fee        
        self.current_future_position = target_pos

        # ======================== open call ========================
        call_px = _safe_float(decision.call_settle)
        call_premium = call_px * self.straddle_size * self.option_per_unit
        call_fee = call_premium * self.fee_rate 
        # open long pos
        self.cash += - call_premium - call_fee 
        call_delta_nav = -call_fee

        self.option_position_call = self.straddle_size
        self.current_call_close = call_px

        # ======================== open put ========================
        put_px = _safe_float(decision.put_settle)
        put_premium = put_px * self.straddle_size * self.option_per_unit
        put_fee = put_premium * self.fee_rate
        # open long pos 
        self.cash += - put_premium - put_fee
        put_delta_nav = -put_fee

        self.option_position_put = self.straddle_size
        self.current_put_close = put_px

        self.nav += future_delta_nav + call_delta_nav + put_delta_nav

        record = BrokerRecord(
            trade_date=decision.trade_date,
            bar_ts=decision.bar_ts,
            nav=self.nav,
            margin_by_future=self.margin_by_future,
            cash=self.cash,
            pnl=0.0,
            pnl_future=0.0,
            pnl_call=0.0,
            pnl_put=0.0,
            pnl_option=0.0,
            cash_future=-future_margin - fut_fee,
            cash_call=-call_premium - call_fee,
            cash_put=-put_premium - put_fee,
            cash_option=-(call_premium + call_fee + put_premium + put_fee),
            nav_future=future_delta_nav,
            nav_call=call_delta_nav,
            nav_put=put_delta_nav,
            nav_option=call_delta_nav + put_delta_nav,
            # capital_used
            capital_used=future_margin + call_premium + put_premium,
            capital_future=future_margin,
            capital_call=call_premium,
            capital_put=put_premium,
            capital_option=call_premium + put_premium,
            future_ts_code=decision.future_ts_code,
            call_ts_code=decision.call_ts_code,
            put_ts_code=decision.put_ts_code,
            fee=fut_fee + call_fee + put_fee,
            fee_future=fut_fee,
            fee_call=call_fee,
            fee_put=put_fee,
            fee_option=call_fee + put_fee,
            slippage=slippage_cost,
            theta_loss=0.0,
            name="open",
        )
        self.records.append(record)
        return record
    
    def on_decision_with_close_pos(self, decision: StrategyBarHedgeSignal) -> BrokerRecord:
        # ======================== close futures ========================
        prev_fut_current_bar = self.data_loader.get_bar(
            ts_code=decision.future_ts_code,
            bar_ts=self.current_decision.bar_ts,
            instrument_type="future",
        )
        fut_px = prev_fut_current_bar["close"].item()
        prev_fut_px = self.prev_decision.future_close
        if fut_px <= 0:
            return None
        
        trade_qty = - self.current_future_position
        trade_sign = 1.0 if trade_qty > 0 else -1.0 if trade_qty < 0 else 0.0
        exec_price = fut_px * (1 + trade_sign * self.slippage_bps / 10000.0)
        
        # future_pnl
        fut_pnl = (fut_px - prev_fut_px) * self.current_future_position * self.future_per_unit
        self.cash += fut_pnl

        # future margin 
        margin_future = self.margin_by_future
        self.cash += margin_future

        # future fee 
        fut_fee = abs(trade_qty) * exec_price * self.future_per_unit * self.fee_rate
        self.cash -= fut_fee

        # future slippage_cost only for record
        slippage_cost = abs(trade_qty) * abs(exec_price - fut_px) * self.future_per_unit

        future_delta_nav = fut_pnl - fut_fee        
        self.current_future_position = 0.0
        self.current_future_code = None
        self.future_per_unit = None
        self.margin_by_future = 0.0
   
        # ======================== close call ========================
        call_px = decision.call_settle
        prev_call_px = self.prev_decision.call_settle

        # call pnl
        call_pnl = (call_px - prev_call_px) * self.option_position_call * self.option_per_unit

        # close long pos
        call_premium = call_px * self.straddle_size * self.option_per_unit
        call_fee = call_premium * self.fee_rate
        self.cash += call_premium - call_fee
        call_delta_nav = call_pnl - call_fee

        self.option_position_call = 0.0
        self.current_call_close = None

        # ======================== close put ========================
        put_px = decision.put_settle
        prev_put_px = self.prev_decision.put_settle

        # put pnl
        put_pnl = (put_px - prev_put_px) * self.option_position_put * self.option_per_unit

        # close long pos
        put_premium = put_px * self.straddle_size * self.option_per_unit
        put_fee = put_premium * self.fee_rate
        self.cash += put_premium - put_fee
        put_delta_nav = put_pnl - put_fee

        self.option_position_put = 0.0
        self.current_put_close = None

        self.nav += future_delta_nav + call_delta_nav + put_delta_nav

        # 计算从上一决策到当前bar的theta损失
        if self.prev_decision is not None:
            dt_days = (decision.bar_ts - self.prev_decision.bar_ts).total_seconds() / (24 * 3600)
            theta_total = (self.prev_decision.theta_call + self.prev_decision.theta_put)
            theta_loss = theta_total * dt_days * self.straddle_size * self.option_per_unit
        else:
            theta_loss = 0.0
        
        record = BrokerRecord(
            trade_date=self.current_decision.trade_date,
            bar_ts=self.current_decision.bar_ts,
            nav=self.nav,
            margin_by_future=self.margin_by_future,
            cash=self.cash,
            pnl=fut_pnl + call_pnl + put_pnl,
            pnl_future=fut_pnl,
            pnl_call=call_pnl,
            pnl_put=put_pnl,
            pnl_option=call_pnl + put_pnl,
            cash_future=fut_pnl - fut_fee + margin_future,
            cash_call=call_premium - call_fee,
            cash_put=put_premium - put_fee,
            cash_option=call_premium - call_fee + put_premium - put_fee,
            nav_future=future_delta_nav,
            nav_call=call_delta_nav,
            nav_put=put_delta_nav,
            nav_option=call_delta_nav + put_delta_nav,
            # capital_used
            capital_used=None,
            capital_future=None,
            capital_call=None,
            capital_put=None,
            capital_option=None,
            future_ts_code=decision.future_ts_code,
            call_ts_code=decision.call_ts_code,
            put_ts_code=decision.put_ts_code,
            fee=fut_fee + call_fee + put_fee,
            fee_future=fut_fee,
            fee_call=call_fee,
            fee_put=put_fee,
            fee_option=call_fee + put_fee,
            slippage=slippage_cost,
            theta_loss=theta_loss,
            name="close",
        )
        self.records.append(record)
        return record

    def on_decision_with_mtm(self, decision: StrategyBarHedgeSignal) -> BrokerRecord:
        # ======================== mtm futures ========================
        fut_px = _safe_float(decision.future_close)
        prev_fut_px = self.prev_decision.future_close

        # future pnl 
        fut_pnl = (fut_px - prev_fut_px) * self.current_future_position * self.future_per_unit
        self.cash += fut_pnl

        # -------------- future delta hedge --------------
        target_future_position = _safe_float(decision.target_future_position) * self.straddle_size
        trade_qty = target_future_position - self.current_future_position
        trade_sign = 1.0 if trade_qty > 0 else -1.0 if trade_qty < 0 else 0.0
        exec_price = fut_px * (1 + trade_sign * self.slippage_bps / 10000.0)

        # future_fee 
        fut_fee = abs(trade_qty) * exec_price * self.future_per_unit * self.fee_rate
        self.cash -= fut_fee

        # future slippage_cost only for record
        slippage_cost = abs(trade_qty) * abs(exec_price - fut_px) * self.future_per_unit
        self.current_future_position = target_future_position

        # future margin 
        future_margin = abs(self.current_future_position) * exec_price * self.future_per_unit * self.margin_rate
        future_delta_margin = self.margin_by_future - future_margin
        self.cash += future_delta_margin
        self.margin_by_future = future_margin

        future_delta_nav = fut_pnl - fut_fee
        self.current_future_position = target_future_position

        # ======================== mtm call ========================
        call_px = _safe_float(decision.call_settle)
        prev_call_px = self.prev_decision.call_settle

        call_mtm_pnl = (call_px - prev_call_px) * self.option_position_call * self.option_per_unit
        call_delta_nav = call_mtm_pnl

        # ======================== mtm put ========================
        put_px = _safe_float(decision.put_settle)
        prev_put_px = self.prev_decision.put_settle

        put_mtm_pnl = (put_px - prev_put_px) * self.option_position_put * self.option_per_unit
        put_delta_nav = put_mtm_pnl
        

        self.nav += future_delta_nav + call_delta_nav + put_delta_nav

        # --- 新增：计算该bar的theta损失 ---
        # 获取时间间隔（天数）
        if self.prev_decision is not None:
            dt_days = (decision.bar_ts - self.prev_decision.bar_ts).total_seconds() / (24 * 3600)
            # theta_call 和 theta_put 已经是日度变化（即一天的价格变化）
            theta_total = (self.prev_decision.theta_call + self.prev_decision.theta_put)
            # 乘以持仓数量、合约乘数，得到现金损失
            theta_loss = theta_total * dt_days * self.straddle_size * self.option_per_unit
        else:
            theta_loss = 0.0

        record = BrokerRecord(
            trade_date=decision.trade_date,
            bar_ts=decision.bar_ts,
            nav=self.nav,
            cash=self.cash,
            margin_by_future=self.margin_by_future,
            pnl=fut_pnl + call_mtm_pnl + put_mtm_pnl,
            pnl_future=fut_pnl,
            pnl_call=call_mtm_pnl,
            pnl_put=put_mtm_pnl,
            pnl_option=call_mtm_pnl + put_mtm_pnl,
            cash_future=fut_pnl - fut_fee + future_delta_margin,
            cash_call=0.0,
            cash_put=0.0,
            cash_option=0.0,
            nav_future=future_delta_nav,
            nav_call=call_delta_nav,
            nav_put=put_delta_nav,
            nav_option=call_delta_nav + put_delta_nav,
            # capital_used
            capital_used=None,
            capital_future=future_margin,
            capital_call=call_px * self.option_position_call * self.option_per_unit,
            capital_put=put_px * self.option_position_put * self.option_per_unit,
            capital_option=call_px * self.option_position_call * self.option_per_unit + put_px * self.option_position_put * self.option_per_unit,
            future_ts_code=decision.future_ts_code,
            call_ts_code=decision.call_ts_code,
            put_ts_code=decision.put_ts_code,
            fee=fut_fee,
            fee_future=fut_fee,
            fee_call=0.0,
            fee_put=0.0,
            fee_option=0.0,
            slippage=slippage_cost,
            theta_loss=theta_loss,
            name='mtm'
        )
        self.records.append(record)
        return record

    
    def to_frame(self) -> pl.DataFrame:
        pass 
    
    def to_daily_frame(self) -> pl.DataFrame:
        pass






class DeltaHedgeRunner:
    "add data -> rebalance -> broker"

    def __init__(self, cfg_exp: ExperimentConfig, strategy_name: str) -> None:
        self.strategy_name = strategy_name
        self.cfg_exp = cfg_exp
        self.data_module = BacktestDataModule(cfg_exp)
        self.data_module.adddata()
        self.strategy = DeltaHedgeStrategy(cfg_exp=cfg_exp)
        self.risk_rater = RiskFreeRate(interest_df=self.data_module.shibor_daily)
        self.selector = ContractSelector(
            cfg_exp=self.cfg_exp,
            fut_basic=self.data_module.fut_basic,
            opt_basic=self.data_module.opt_basic,
            fut_bars=self.data_module.fut_bars,
            opt_bars=self.data_module.opt_bars,
        )
        self.broker = BrokerEngine(
            straddle_size=cfg_exp.hedge.straddle_size, 
            data_loader=self.data_module, 
            strategy=self.strategy,
        )
        self.daily_decision: list[StrategyBarHedgeSignal] = []
        self.backtest_record: list[BacktestRecord] = []
        self.backtest_result = pl.DataFrame()

    def decide_roll(self, prev_bar_signal: StrategyBarHedgeSignal, bar_signal: StrategyBarHedgeSignal) -> bool:
        return (
            prev_bar_signal is not None
            and (prev_bar_signal.future_ts_code != bar_signal.future_ts_code
            or prev_bar_signal.call_ts_code != bar_signal.call_ts_code
            or prev_bar_signal.put_ts_code != bar_signal.put_ts_code)
        )
    
    def update_bar_signal(self, prev_bar_signal: StrategyBarHedgeSignal, bar_signal: StrategyBarHedgeSignal) -> dict:
        prev_bar_fut_close_update_to_current_bar_ts = self.data_module.get_bar(
            ts_code=prev_bar_signal.future_ts_code,
            bar_ts=bar_signal.bar_ts,
            instrument_type="future",
        )
        prev_bar_call_close_update_to_current_bar_ts = self.data_module.get_bar(
            ts_code=prev_bar_signal.call_ts_code,
            bar_ts=bar_signal.bar_ts,
            instrument_type="call",
        )
        if prev_bar_call_close_update_to_current_bar_ts.is_empty():
            prev_bar_call_close_update_to_current_bar_ts = self.data_module.get_bar(
                ts_code=prev_bar_signal.call_ts_code,
                bar_ts=prev_bar_signal.bar_ts,
                instrument_type="call",
            )
        prev_bar_put_close_update_to_current_bar_ts = self.data_module.get_bar(
            ts_code=prev_bar_signal.put_ts_code,
            bar_ts=bar_signal.bar_ts,
            instrument_type="put",
        )
        if prev_bar_put_close_update_to_current_bar_ts.is_empty():
            prev_bar_put_close_update_to_current_bar_ts = self.data_module.get_bar(
                ts_code=prev_bar_signal.put_ts_code,
                bar_ts=prev_bar_signal.bar_ts,
                instrument_type="put",
            )
        prev_bar_update_to_current_bar_ts = {
            "bar_ts": bar_signal.bar_ts,
            "future_open": prev_bar_fut_close_update_to_current_bar_ts["open"].item(),
            "future_close": prev_bar_fut_close_update_to_current_bar_ts["close"].item(),
            "call_close": prev_bar_call_close_update_to_current_bar_ts["close"].item(),
            "put_close": prev_bar_put_close_update_to_current_bar_ts["close"].item(),
        }
        return prev_bar_update_to_current_bar_ts

    def run(self) -> pl.DataFrame:

        log_info("开始回测")
        prev_bar_signal = None
        prev_selected = None
        last_trade_date = None

        for trade_date in self.data_module.trade_dates:

            current_selected = self.selector.select_contract(trade_date)
            if current_selected is None:
                continue

            # ---- 新增：为当前合约初始化 IV（如果尚未缓存）----
            if (current_selected.call_ts_code not in self.strategy.iv_cache_by_option or
                current_selected.put_ts_code not in self.strategy.iv_cache_by_option):
                self._init_contract_iv(current_selected, None)

            date_close_bars = self.data_module.get_day_bars(current_selected, trade_date, use_settle_col=False)
            if date_close_bars.is_empty():
                continue

            rate = self.risk_rater.get_rate(
                trade_date, _year_fraction(current_selected.option_expiry_date, trade_date)
            )

            for bar in date_close_bars.iter_rows(named=True):
                bar_signal = self.strategy.on_bar(current_selected, trade_date, bar, rate)
                
                is_roll = self.decide_roll(prev_bar_signal, bar_signal)
                if is_roll:
                    # 1. 旧合约的 IV 更新（基于昨日数据）
                    self._init_contract_iv(prev_selected, last_trade_date)

                    # 2. 新合约的 IV 初始化（同样基于昨日数据）
                    self._init_contract_iv(current_selected, last_trade_date)

                    # 3. 生成旧合约在当前 bar 的平仓信号
                    pre_bar_update = self.update_bar_signal(prev_bar_signal, bar_signal)
                    prev_bar_signal_update = self.strategy.on_bar(prev_selected, trade_date, pre_bar_update, rate)

                    # 4. 平仓操作（注意设置 current_decision）
                    self.broker.current_decision = prev_bar_signal_update   # 重要：让平仓方法能获取正确 bar_ts
                    close_record = self.broker.on_decision_with_close_pos(prev_bar_signal_update)
                    if close_record:
                        self.backtest_record.append(BacktestRecord(
                            bar_signal=prev_bar_signal_update, broker_record=close_record
                        ))

                    # 5. 生成新合约的开仓信号（此时 IV 已缓存）
                    bar_signal = self.strategy.on_bar(current_selected, trade_date, bar, rate)

                    # # 5. 调用 broker 处理换仓（先平旧仓，后开新仓）
                    # self.broker.on_decision(bar_signal, prev_bar_signal_update)

                    # 6. 开仓操作
                    self.broker.current_decision = bar_signal
                    open_record = self.broker.on_decision_with_open_pos(bar_signal)
                    if open_record:
                        self.backtest_record.append(BacktestRecord(
                            bar_signal=bar_signal, broker_record=open_record
                        ))
                    
                    # 7. 更新 broker 的内部状态
                    self.broker.prev_decision = bar_signal
                    self.broker.prev_record = open_record
                else:
                    # 非换仓：只进行 MTM 操作
                    self.broker.on_decision(bar_signal)
                    # 记录回测数据
                    backtest_record = BacktestRecord(
                        bar_signal=bar_signal, 
                        broker_record=self.broker.prev_record
                    )
                    self.backtest_record.append(backtest_record)
                prev_bar_signal = bar_signal
            prev_selected = current_selected
            last_trade_date = trade_date
            
            self.daily_decision.append(bar_signal)
            # 更新iv
            date_settle_bars = self.data_module.get_day_bars(current_selected, trade_date, use_settle_col=True)
            if date_settle_bars.is_empty():
                continue
            self.strategy.on_day_close(current_selected, trade_date, date_settle_bars.tail(1).row(0, named=True), rate)
            
        # 回测结束，平仓所有持仓
        if self.broker.prev_decision is not None:
            # 平仓时需要使用最后一个决策信号，因为我们需要获取当前持仓合约的价格
            final_close = self.broker.on_decision_with_close_pos(self.broker.prev_decision)
            if final_close:
                self.backtest_record.append(BacktestRecord(
                    bar_signal=self.broker.prev_decision, broker_record=final_close
                ))
        # 循环结束后，将broker.records转为pl.DataFrame
        self.broker_record = pl.DataFrame([r.__dict__ for r in self.broker.records])
        self.daily_decision = pl.DataFrame([d.__dict__ for d in self.daily_decision])
        backtest_result = pl.DataFrame([b.flatten() for b in self.backtest_record])
        self.backtest_result = backtest_result.sort("bar_ts").with_columns(
            pl.col("pnl").cum_sum().alias("pnl_cumsum"),
            pl.col("pnl_future").cum_sum().alias("pnl_future_cumsum"),
            pl.col("pnl_call").cum_sum().alias("pnl_call_cumsum"),
            pl.col("pnl_put").cum_sum().alias("pnl_put_cumsum"),
            pl.col("pnl_option").cum_sum().alias("pnl_option_cumsum"),
            pl.col("future_close").pct_change().alias("future_return_rate"),
            pl.col("call_settle").pct_change().alias("call_return_rate"),
            pl.col("put_settle").pct_change().alias("put_return_rate"),
        )
        return self.backtest_result

    def _init_contract_iv(self, selected: SelectedContracts, trade_date: date | None) -> None:
        """为指定合约加载前一日结算价，计算并缓存 IV"""
        if trade_date is None:
            trade_date = self.cfg_exp.backtest.start_date - timedelta(days=1)

        max_look_back = 22
        found_date = None
        for i in range(max_look_back):
            check_date = trade_date - timedelta(days=i)
            settle_bars = self.data_module.get_day_bars(selected, check_date, use_settle_col=True)
            if not settle_bars.is_empty():
                found_date = check_date
                break

        if found_date is None:
            self.strategy.iv_cache_by_option[selected.call_ts_code] = 0.2
            self.strategy.iv_cache_by_option[selected.put_ts_code] = 0.2
            return

        settle_bars = self.data_module.get_day_bars(selected, found_date, use_settle_col=True)
        last_bar = settle_bars.tail(1).row(0, named=True)
        rate = self.risk_rater.get_rate(found_date, _year_fraction(selected.option_expiry_date, found_date))
        self.strategy.on_day_close(selected, found_date, last_bar, rate)











if __name__ == "__main__":
    from core.config_shema import load_confg
    cfg = load_confg("config/shfe_ag_demo.yaml")
    runner = DeltaHedgeRunner(cfg_exp=cfg[0], strategy_name="delta_hedge_self")
    runner.run()