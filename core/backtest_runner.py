


from typing import Literal
import polars as pl
from dataclasses import asdict, dataclass
from datetime import date
from datetime import datetime, time
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
class StrategyDecision:
    trade_date: date
    bar_ts: datetime
    future_ts_code: str
    future_close: float
    future_per_unit: float
    call_ts_code: str
    call_close: float
    put_ts_code: str
    put_close: float
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
    name: str=Literal["open", "mtm", "close"]


@dataclass
class BacktestRecord:
    decision: StrategyDecision
    broker_record: BrokerRecord

    def flatten(self) -> dict:
        d = {}
        decision_dict = asdict(self.decision)
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
        self.fut_bars, self.opt_bars = self._load_bar_data()
        self.shibor_daily = self.loader.load_shibor_daily()
        self.trade_dates = (
            self.fut_bars.select("trade_date").unique().sort("trade_date").to_series().to_list()
            if not self.fut_bars.is_empty()
            else []
        )

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

    # 输入ts_code和trade_date，返回期货或者call或者put的一根bar
    def get_bar(
        self, 
        ts_code: str, 
        bar_ts: datetime, 
        instrument_type: Literal["future", "call", "put"]
    ) -> pl.DataFrame:

        if instrument_type == "future":
            return self.fut_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).sort("bar_ts")
        elif instrument_type == "call":
            return self.opt_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).sort("bar_ts")
        elif instrument_type == "put":
            return self.opt_bars.filter((pl.col("ts_code") == ts_code) & (pl.col("bar_ts") == bar_ts)).sort("bar_ts")
        else:
            raise ValueError(f"Invalid instrument type: {instrument_type}")
        # TODO columns name

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
                (pl.col("trade_date") >= self.cfg_backtest.start_date)
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
                (pl.col("trade_date") >= self.cfg_backtest.start_date)
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
    ) -> StrategyDecision | None:
        fut_open = _safe_float(bar["future_open"])
        fut_px = _safe_float(bar["future_close"])
        call_px = _safe_float(bar["call_close"])
        put_px = _safe_float(bar["put_close"])
        if fut_open <= 0:
            fut_open = fut_px
        if fut_px <= 0 or fut_open <= 0:
            return None

        option_expiry_datetime = datetime.combine(selected.option_expiry_date, time(15, 0, 0))
        ttm = _year_fraction(option_expiry_datetime, bar["bar_ts"])       
        iv_call = self.iv_cache_by_option.get(selected.call_ts_code, 0.2)
        iv_put = self.iv_cache_by_option.get(selected.put_ts_code, 0.2)
        # 对冲目标用bar open的期货价格计算，匹配开仓对冲时点
        delta_call = CrrModel.delta(fut_open, selected.strike, rate, iv_call, ttm, "C")
        gamma_call = CrrModel.gamma(fut_open, selected.strike, rate, iv_call, ttm, "C")
        vega_call = CrrModel.vega(fut_open, selected.strike, rate, iv_call, ttm, "C")
        theta_call = CrrModel.theta(fut_open, selected.strike, rate, iv_call, ttm, "C")
        delta_put = CrrModel.delta(fut_open, selected.strike, rate, iv_put, ttm, "P")
        gamma_put = CrrModel.gamma(fut_open, selected.strike, rate, iv_put, ttm, "P")
        vega_put = CrrModel.vega(fut_open, selected.strike, rate, iv_put, ttm, "P")
        theta_put = CrrModel.theta(fut_open, selected.strike, rate, iv_put, ttm, "P")
        combo_delta = delta_call + delta_put

        scale = 1.0
        if self.cfg_hedge.use_contract_unit:
            scale = selected.option_per_unit / max(selected.future_per_unit, 1e-9)
        target_pos = -combo_delta * scale

        strategy_decision = StrategyDecision(
            trade_date=trade_date,
            bar_ts=bar["bar_ts"],
            future_ts_code=selected.future_ts_code,
            future_close=fut_px,
            future_per_unit=selected.future_per_unit,
            call_ts_code=selected.call_ts_code,
            call_close=call_px,
            put_ts_code=selected.put_ts_code,
            put_close=put_px,
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
        return strategy_decision

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
    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        fee_rate: float = 0.0005,
        slippage_bps: float = 1.0,
        straddle_size: float = 1.0,
        margin_rate: float = 0.15,
        data_loader: BacktestDataModule = None,
    ):

        self.init_cash = init_cash
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.straddle_size = straddle_size
        self.margin_rate = margin_rate
        self.cash = init_cash
        self.nav = init_cash
        self.data_loader = data_loader
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

        self.prev_decision: StrategyDecision | None = None
        self.current_decision: StrategyDecision | None = None
        self.prev_record: BrokerRecord | None = None

    def on_decision(self, decision: dict) -> None:
        self.current_decision = decision
        if self.prev_decision is None:
            record = self.on_decision_with_open_pos(self.current_decision)
        elif (
            self.prev_decision.future_ts_code != self.current_decision.future_ts_code and
            self.prev_decision.call_ts_code != self.current_decision.call_ts_code and
            self.prev_decision.put_ts_code != self.current_decision.put_ts_code
        ):
            record = self.on_decision_with_close_pos(self.prev_decision)
            record = self.on_decision_with_open_pos(self.current_decision)
        else:
            record = self.on_decision_with_mtm(self.current_decision)
        self.prev_decision = decision
        self.prev_record = record

    def on_decision_with_open_pos(self, decision: StrategyDecision) -> BrokerRecord:
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
        call_px = _safe_float(decision.call_close)
        call_premium = call_px * self.straddle_size * self.option_per_unit
        call_fee = call_premium * self.fee_rate 
        # open long pos
        self.cash += - call_premium - call_fee 
        call_delta_nav = -call_fee

        self.option_position_call = self.straddle_size
        self.current_call_close = call_px

        # ======================== open put ========================
        put_px = _safe_float(decision.put_close)
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
            name="open",
        )
        self.records.append(record)
        return record
    
    def on_decision_with_close_pos(self, decision: StrategyDecision) -> BrokerRecord:
        # ======================== close futures ========================
        fut_px = self.data_loader.get_bar(
            ts_code=decision.future_ts_code,
            bar_ts=self.current_decision.bar_ts,
            instrument_type="future",
        )["close"].item()
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
        call_px = self.data_loader.get_bar(
            ts_code=decision.call_ts_code,
            bar_ts=self.current_decision.bar_ts,
            instrument_type="call",
        )["close"].item()
        prev_call_px = self.prev_decision.call_close

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
        put_px = self.data_loader.get_bar(
                    ts_code=decision.put_ts_code,
                    bar_ts=self.current_decision.bar_ts,
                    instrument_type="put",
                )["close"].item()
        prev_put_px = self.prev_decision.put_close

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
            name="close",
        )
        self.records.append(record)
        return record

    def on_decision_with_mtm(self, decision: dict) -> BrokerRecord:
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
        call_px = _safe_float(decision.call_close)
        prev_call_px = self.prev_decision.call_close

        call_mtm_pnl = (call_px - prev_call_px) * self.option_position_call * self.option_per_unit
        call_delta_nav = call_mtm_pnl

        # ======================== mtm put ========================
        put_px = _safe_float(decision.put_close)
        prev_put_px = self.prev_decision.put_close

        put_mtm_pnl = (put_px - prev_put_px) * self.option_position_put * self.option_per_unit
        put_delta_nav = put_mtm_pnl
        

        self.nav += future_delta_nav + call_delta_nav + put_delta_nav

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
        self.broker = BrokerEngine(straddle_size=cfg_exp.hedge.straddle_size, data_loader=self.data_module)
        self.daily_decision: list[StrategyDecision] = []
        self.backtest_record: list[BacktestRecord] = []
        self.backtest_result = pl.DataFrame()

    def run(self) -> pl.DataFrame:

        log_info("开始回测")
        for trade_date in self.data_module.trade_dates:
            selected = self.selector.select_contract(trade_date)
            if selected is None:
                continue
            day_bars = self.data_module.get_day_bars(selected, trade_date)
            if day_bars.is_empty():
                continue

            rate = self.risk_rater.get_rate(
                trade_date, _year_fraction(selected.option_expiry_date, trade_date)
            )

            for bar in day_bars.iter_rows(named=True):
                decision = self.strategy.on_bar(selected, trade_date, bar, rate)
                if decision is None:
                    continue
                # update broker record
                self.broker.on_decision(decision)
                # update backtest record
                backtest_record = BacktestRecord(
                    decision=decision, 
                    broker_record=self.broker.prev_record
                )
                self.backtest_record.append(backtest_record)
            
            self.strategy.on_day_close(selected, trade_date, day_bars.tail(1).row(0, named=True), rate)
            self.daily_decision.append(decision)
        self.broker.on_decision_with_close_pos(decision)
        # 循环结束后，将broker.records转为pl.DataFrame
        self.broker_record = pl.DataFrame([r.__dict__ for r in self.broker.records])
        self.daily_decision = pl.DataFrame([d.__dict__ for d in self.daily_decision])
        self.backtest_result = pl.DataFrame([b.flatten() for b in self.backtest_record])
        return self.backtest_result












if __name__ == "__main__":
    from core.config_shema import load_confg
    cfg = load_confg("config/shfe_ag_demo.yaml")
    runner = DeltaHedgeRunner(cfg_exp=cfg[0], strategy_name="delta_hedge_self")
    runner.run()