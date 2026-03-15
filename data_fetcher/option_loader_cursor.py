from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

from core.config_shema import FutureConfig, HedgeConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_ROOT / "local_db"

Freq = Literal["1d", "15min"]


def _to_py_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return date.fromisoformat(text) if fmt == "%Y-%m-%d" else date(
                int(text[:4]), int(text[4:6]), int(text[6:8])
            )
        except ValueError:
            continue
    return None


class OptionLoader:
    """Load local future/option data with contract rolling support."""

    def __init__(self, future_config: FutureConfig, hedge_config: HedgeConfig):
        self.future_config = future_config
        self.hedge_config = hedge_config

    @property
    def _fut_basic_path(self) -> Path:
        return (
            LOCAL_DATA_DIR
            / "contract_info"
            / "future_basic"
            / f"{self.future_config.exchange}_{self.future_config.fut_code}_fut_basic.parquet"
        )

    @property
    def _opt_basic_path(self) -> Path:
        return (
            LOCAL_DATA_DIR
            / "contract_info"
            / "option_basic"
            / f"{self.future_config.exchange}_{self.future_config.fut_code}_opt_basic.parquet"
        )

    def _parse_trade_date(self, trade_date: date | str) -> date:
        parsed = _to_py_date(trade_date)
        if parsed is None:
            raise ValueError(f"Invalid trade_date: {trade_date!r}")
        return parsed

    def _normalize_contract_dates(self, df_contract: pl.DataFrame) -> pl.DataFrame:
        date_cols = ["list_date", "delist_date", "maturity_date", "last_ddate"]
        exprs: list[pl.Expr] = []
        for col in date_cols:
            if col not in df_contract.columns:
                continue
            exprs.append(
                pl.coalesce(
                    [
                        pl.col(col).cast(pl.Date, strict=False),
                        pl.col(col)
                        .cast(pl.Utf8, strict=False)
                        .str.strptime(pl.Date, "%Y%m%d", strict=False),
                        pl.col(col)
                        .cast(pl.Utf8, strict=False)
                        .str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                    ]
                ).alias(col)
            )
        return df_contract.with_columns(exprs) if exprs else df_contract

    def load_future_basic(self) -> pl.DataFrame:
        if not self._fut_basic_path.exists():
            raise FileNotFoundError(f"Future basic file not found: {self._fut_basic_path}")
        return self._normalize_contract_dates(pl.read_parquet(self._fut_basic_path))

    def load_option_basic(self) -> pl.DataFrame:
        if not self._opt_basic_path.exists():
            raise FileNotFoundError(f"Option basic file not found: {self._opt_basic_path}")
        return self._normalize_contract_dates(pl.read_parquet(self._opt_basic_path))

    def _active_on_trade_date(self, df_contract: pl.DataFrame, trade_date: date) -> pl.DataFrame:
        if "list_date" in df_contract.columns:
            df_contract = df_contract.filter(
                pl.col("list_date").is_null() | (pl.col("list_date") <= trade_date)
            )
        if "delist_date" in df_contract.columns:
            df_contract = df_contract.filter(
                pl.col("delist_date").is_null() | (pl.col("delist_date") >= trade_date)
            )
        return df_contract

    def _pick_front_two_maturities(
        self, df_contract: pl.DataFrame, trade_date: date
    ) -> pl.DataFrame:
        if df_contract.is_empty():
            return df_contract

        if "maturity_date" in df_contract.columns:
            maturity_expr = pl.col("maturity_date")
        elif "last_ddate" in df_contract.columns:
            maturity_expr = pl.col("last_ddate")
        elif "delist_date" in df_contract.columns:
            maturity_expr = pl.col("delist_date")
        else:
            raise ValueError("Contract dataframe missing maturity fields.")

        df_active = (
            df_contract.with_columns(maturity_expr.alias("_maturity"))
            .filter(pl.col("_maturity").is_not_null() & (pl.col("_maturity") >= trade_date))
            .with_columns((pl.col("_maturity") - pl.lit(trade_date)).dt.total_days().alias("_ttm_days"))
        )
        if df_active.is_empty():
            return df_active

        maturity_vals = (
            df_active.select(pl.col("_maturity"))
            .unique()
            .sort("_maturity")
            .get_column("_maturity")
            .to_list()
        )
        if not maturity_vals:
            return df_active.clear()

        front_days = (maturity_vals[0] - trade_date).days
        start_idx = 1 if front_days <= self.hedge_config.roll_days_before_maturity else 0
        chosen = maturity_vals[start_idx : start_idx + 2]
        return df_active.filter(pl.col("_maturity").is_in(chosen))

    def select_future_contracts(self, trade_date: date | str) -> pl.DataFrame:
        td = self._parse_trade_date(trade_date)
        df = self._active_on_trade_date(self.load_future_basic(), td)
        return self._pick_front_two_maturities(df, td).drop(["_maturity", "_ttm_days"], strict=False)

    def select_option_contracts(
        self,
        trade_date: date | str,
        call_put: str | None = None,
    ) -> pl.DataFrame:
        td = self._parse_trade_date(trade_date)
        cp = call_put or self.future_config.call_put
        df = self._active_on_trade_date(self.load_option_basic(), td)
        if cp and "call_put" in df.columns:
            df = df.filter(pl.col("call_put") == cp)
        return self._pick_front_two_maturities(df, td).drop(["_maturity", "_ttm_days"], strict=False)

    @staticmethod
    def _normalize_price_date(df: pl.DataFrame) -> pl.DataFrame:
        if "trade_date" in df.columns:
            df = df.with_columns(
                pl.coalesce(
                    [
                        pl.col("trade_date").cast(pl.Date, strict=False),
                        pl.col("trade_date")
                        .cast(pl.Utf8, strict=False)
                        .str.strptime(pl.Date, "%Y%m%d", strict=False),
                        pl.col("trade_date")
                        .cast(pl.Utf8, strict=False)
                        .str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                    ]
                ).alias("trade_date")
            )
        if "trade_time" in df.columns:
            df = df.with_columns(
                pl.col("trade_time")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                .alias("trade_time")
            )
            if "trade_date" not in df.columns:
                df = df.with_columns(pl.col("trade_time").dt.date().alias("trade_date"))
        return df

    def _load_price_one_contract(
        self, ts_code: str, trade_date: date, asset: Literal["future", "option"], freq: Freq
    ) -> pl.DataFrame:
        root = "future" if asset == "future" else "option"
        suffix = "fut" if asset == "future" else "opt"
        freq_tag = "daily" if freq == "1d" else "minute"
        folder = (
            LOCAL_DATA_DIR
            / f"{root}_price_{freq_tag}"
            / f"{self.future_config.exchange}_{self.future_config.fut_code}"
        )
        file_path = folder / f"{ts_code}_{suffix}_{freq_tag}.parquet"
        if not file_path.exists():
            return pl.DataFrame()

        df = self._normalize_price_date(pl.read_parquet(file_path))
        if "trade_date" not in df.columns:
            return pl.DataFrame()
        return df.filter(pl.col("trade_date") == trade_date)

    def load_future_price(self, trade_date: date | str, freq: Freq = "1d") -> pl.DataFrame:
        td = self._parse_trade_date(trade_date)
        df_contract = self.select_future_contracts(td)
        if df_contract.is_empty():
            return pl.DataFrame()

        rows: list[pl.DataFrame] = []
        for ts_code in df_contract.get_column("ts_code").to_list():
            df_px = self._load_price_one_contract(ts_code=ts_code, trade_date=td, asset="future", freq=freq)
            if df_px.is_empty():
                continue
            rows.append(df_px.with_columns(pl.lit(ts_code).alias("selected_ts_code")))
        return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()

    def load_option_price(
        self,
        trade_date: date | str,
        freq: Freq = "1d",
        call_put: str | None = None,
    ) -> pl.DataFrame:
        td = self._parse_trade_date(trade_date)
        df_contract = self.select_option_contracts(td, call_put=call_put)
        if df_contract.is_empty():
            return pl.DataFrame()

        rows: list[pl.DataFrame] = []
        for ts_code in df_contract.get_column("ts_code").to_list():
            df_px = self._load_price_one_contract(ts_code=ts_code, trade_date=td, asset="option", freq=freq)
            if df_px.is_empty():
                continue
            rows.append(df_px.with_columns(pl.lit(ts_code).alias("selected_ts_code")))
        return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()
