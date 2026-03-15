from typing import List, Optional, Literal
import polars as pl
import os
from pathlib import Path
from datetime import datetime, date


from data_fetcher.tusharedb import TuShare
from core.config_shema import FutureConfig
from utils.logger import log_info


PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "local_db"



class UpdateLocalDB:
    @staticmethod
    def get_last_update_date(file: Path) -> Optional[pl.Date]:
        """
        获取数据库文件中的最大交易日期。

        :param file: Parquet 文件路径
        :return: 最大日期或 None
        """
        if file.exists():
            df = pl.read_parquet(file)
            return df["trade_date"].max()
        return None

    @staticmethod
    def get_first_update_date(file: Path) -> Optional[pl.Date]:
        """
        获取数据库文件中的最早交易日期。

        :param file: Parquet 文件路径
        :return: 最大日期或 None
        """
        if file.exists():
            df = pl.read_parquet(file)
            return df["trade_date"].min()
        return None

    @staticmethod
    def get_instrument_id(file: Path) -> List[str]:
        """获取数据库中当前的所有instrument_id"""
        if file.exists():
            df = pl.read_parquet(file)
            unique_values = set(df["instrument_id"].unique())
            return unique_values
        else:
            return None

    @staticmethod
    def update_old(
        file: Path,
        df: pl.DataFrame,
        unique_subset: List[str] = ["trade_date", "instrument_id"],
    ):
        """
        合并新数据并写入本地数据库。

        :param file: Parquet 文件路径
        :param df: 新数据，需包含 trade_date 和 instrument_id
        :param unique_subset: 用于去重的列名列表
        """
        file.parent.mkdir(parents=True, exist_ok=True)

        if file.exists():
            df_old = pl.read_parquet(file)
            # log_info(f"本地数据库 {file.name} 已存在，原数据行数为{df_old.shape[0]}")
            df = df.cast(df_old.schema, strict=False)
            df = pl.concat([df_old, df])
            df = df.unique(subset=unique_subset)

        df.sort(by=unique_subset).write_parquet(file)
        # log_info(f"已更新本地数据库 {file.name}，当前数据行数为{df.shape[0]}")

    @staticmethod
    def get_max_rows(file: Path) -> int:
        if file.exists():
            df = pl.read_parquet(file)
            return df.shape[0]
        return 0

    @staticmethod
    def get_update_range(
        file: Path,
        start: pl.Date,
        end: pl.Date,
        list_date: pl.Date | None = None,
        delist_date: pl.Date | None = None
    ) -> list[tuple[pl.Date, pl.Date]]:
        
        if start > end:
            raise ValueError(f"起始日期 {start} 不能晚于结束日期 {end}")

        # 上市日前不更新，退市日后不更新
        if list_date is not None:
            start = max(start, list_date)
        if delist_date is not None:
            end = min(end, delist_date)
        if start > end:
            return []

        if not file.exists():
            return [(start, end)]

        local_first = UpdateLocalDB.get_first_update_date(file)
        local_last = UpdateLocalDB.get_last_update_date(file)

        if local_first is None or local_last is None:
            return [(start, end)]

        # 如果已经完全覆盖
        if local_first <= start and local_last >= end:
            return []

        update_ranges = []

        # 请求起始 < 本地起始 → 补充 [start, local_first)
        if start < local_first:
            update_ranges.append((start, local_first))

        # 请求结束 > 本地结束 → 补充 [local_last, end]
        if end > local_last:
            update_ranges.append((local_last, end))

        update_ranges = [(s, e) for s, e in update_ranges if s <= e]

        return update_ranges



# 获取期货合约信息、日频/分钟频kline
class FutureFetcher:
    def __init__(self, future_config: FutureConfig):
        self.future_config = future_config
        self.ts = TuShare()
        self.update_db = UpdateLocalDB()

    @staticmethod
    def _normalize_trade_date(df: pl.DataFrame) -> pl.DataFrame:
        if "trade_date" not in df.columns:
            return df
        return df.with_columns(
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

    def update_future_basic(self, fut_code: str, update_basic: bool = False) -> pl.DataFrame:

        file_path = (LOCAL_DATA_DIR 
        / "contract_info" / "future_basic" 
        / f"{self.future_config.exchange}_{self.future_config.fut_code}_fut_basic.parquet")

        if update_basic or not file_path.exists():
            _df = self.ts.get_future_basic(
                exchange=self.future_config.exchange,
                fut_code=fut_code,
            )
            df = pl.from_pandas(_df)
            self.update_db.update_old(
                file=file_path,
                df=df,
                unique_subset=["ts_code", "symbol", "fut_code"],
            )
        else:
            df = pl.read_parquet(file_path)

        return df
    

    def fetch_future_kline_daily(
        self,
        ts_code: str,
        start: date,
        end: date,
        list_date: date | None = None,
        delist_date: date | None = None,
        fields: list[str] | None = None,
    ):

        data_file = (
            LOCAL_DATA_DIR 
            / "future_price_daily" 
            / f"{self.future_config.exchange}_{self.future_config.fut_code}"
            / f"{ts_code}_fut_daily.parquet")

        update_ranges = self.update_db.get_update_range(
            file=data_file,
            start=start,
            end=end,
            list_date=list_date,
            delist_date=delist_date
        )
        for _start, _end in update_ranges:
            _df_kline = self.ts.get_future_kline_daily(
                ts_code=ts_code,
                start=_start,
                end=_end,
                fields=fields,
            )
            df_kline = pl.from_arrow(_df_kline)
            df_kline = self._normalize_trade_date(df_kline)
            df_kline = df_kline.filter(pl.col("trade_date").is_not_null())
            self.update_db.update_old(
                file=data_file,
                df=df_kline,
                unique_subset=["trade_date", "ts_code"],
            )
            log_info(f"增量更新期货合约 {ts_code} 的日线数据，更新范围 {_start} 至 {_end}，更新行数 {_df_kline.shape[0]}")
        
        return
    
    def fetch_future_kline_minute(
        self,
        ts_code: str,
        start: date,
        end: date,
        freq: Literal["1min", "5min", "15min", "30min", "60min"],
        list_date: date | None = None,
        delist_date: date | None = None,
        start_time: str = "00:00:00",
        end_time: str = "23:59:59",
    ):
        data_file = (
            LOCAL_DATA_DIR 
            / "future_price_minute" 
            / f"{self.future_config.exchange}_{self.future_config.fut_code}"
            / f"{ts_code}_fut_minute.parquet")
        update_ranges = self.update_db.get_update_range(
            file=data_file,
            start=start,
            end=end,
            list_date=list_date,
            delist_date=delist_date
        )
        for _start, _end in update_ranges:
            _df_kline = self.ts.get_future_kline_minute(
                ts_code=ts_code,
                freq=freq,
                start=_start,
                end=_end,
                start_time=start_time,
                end_time=end_time,
            )
            df_kline = pl.from_arrow(_df_kline)
            df_kline = self._normalize_trade_date(df_kline)
            df_kline = df_kline.with_columns(
                pl.col("trade_time")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                .alias("trade_time")
            ).with_columns(pl.col("trade_time").dt.date().alias("trade_date"))
            df_kline = df_kline.filter(pl.col("trade_time").is_not_null())

            self.update_db.update_old(
                file=data_file,
                df=df_kline,
                unique_subset=["trade_time", "ts_code"],
            )
            log_info(f"增量更新期货合约 {ts_code} 的分钟线数据，更新范围 {_start} 至 {_end}，更新行数 {_df_kline.shape[0]}")
        return



class OptionFetcher:
    def __init__(self, future_config: FutureConfig):
        self.future_config = future_config
        self.ts = TuShare()
        self.update_db = UpdateLocalDB()
    
    @staticmethod
    def _normalize_trade_date(df: pl.DataFrame) -> pl.DataFrame:
        if "trade_date" not in df.columns:
            return df
        return df.with_columns(
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

    def update_option_basic(self, opt_code: str, update_basic: bool = False) -> pl.DataFrame:
        file_path = (LOCAL_DATA_DIR 
            / "contract_info" 
            / "option_basic" 
            / f"{self.future_config.exchange}_{self.future_config.fut_code}_opt_basic.parquet"
        )
        if update_basic or not file_path.exists():
            _df = self.ts.get_option_basic(
                opt_code=opt_code,
            )
            df = pl.from_pandas(_df)
            self.update_db.update_old(
                file=file_path,
                df=df,
                unique_subset=["opt_code", "maturity_date", "ts_code"],
            )
        else:
            df = pl.read_parquet(file_path)
        return df
    
    def fetch_option_kline_daily(
        self,
        ts_code: str,
        start: date,
        end: date,
        list_date: date | None = None,
        delist_date: date | None = None,
        fields: list[str] | None = None,
    ):
        data_file = (LOCAL_DATA_DIR 
            / "option_price_daily" 
            / f"{self.future_config.exchange}_{self.future_config.fut_code}"
            / f"{ts_code}_opt_daily.parquet")
        
        update_ranges = self.update_db.get_update_range(
            file=data_file,
            start=start,
            end=end,
            list_date=list_date,
            delist_date=delist_date
        )
        for _start, _end in update_ranges:
            _df_kline = self.ts.get_option_daily(
                ts_code=ts_code,
                start=_start,
                end=_end,
                fields=fields,
            )
            df_kline = pl.from_arrow(_df_kline)
            df_kline = self._normalize_trade_date(df_kline)
            df_kline = df_kline.filter(pl.col("trade_date").is_not_null())

            self.update_db.update_old(
                file=data_file,
                df=df_kline,
                unique_subset=["trade_date", "ts_code"],
            )
            log_info(f"增量更新期权合约 {ts_code} 的日线数据，更新范围 {_start} 至 {_end}，更新行数 {_df_kline.shape[0]}")
        
        return

    def fetch_option_kline_minute(
        self,
        ts_code: str,
        start: date,
        end: date,
        freq: Literal["1min", "5min", "15min", "30min", "60min"],
        list_date: date | None = None,
        delist_date: date | None = None,
        start_time: str = "00:00:00",
        end_time: str = "23:59:59",
    ):
        data_file = (LOCAL_DATA_DIR 
            / "option_price_minute" 
            / f"{self.future_config.exchange}_{self.future_config.fut_code}"
            / f"{ts_code}_opt_minute.parquet")
        
        update_ranges = self.update_db.get_update_range(
            file=data_file,
            start=start,
            end=end,
            list_date=list_date,
            delist_date=delist_date
        )
        for _start, _end in update_ranges:
            _df_kline = self.ts.get_option_minute(
                ts_code=ts_code,
                freq=freq,
                start=_start,
                end=_end,
                start_time=start_time,
                end_time=end_time,
            )
            df_kline = pl.from_arrow(_df_kline).with_columns(
                # 1. 解析 trade_time 为 Datetime 类型（覆盖原列，补全正确格式）
                pl.col("trade_time").str.strptime(
                    dtype=pl.Datetime,
                    format="%Y-%m-%d %H:%M:%S", 
                    strict=False,                # 容错：无效值返回 null，不崩溃
                    exact=True                   # 严格匹配格式
                ).alias("trade_time")
            ).with_columns(
                # 2. 添加 trade_date 列
                pl.col("trade_time").dt.date().alias("trade_date")
            )
            self.update_db.update_old(
                file=data_file,
                df=df_kline,
                unique_subset=["trade_time", "ts_code"],
            )
            log_info(f"增量更新期权合约 {ts_code} 的分钟线数据，更新范围 {_start} 至 {_end}，更新行数 {df_kline.shape[0]}")
        return


class MarketDataFetcher:
    def __init__(self):
        self.ts = TuShare()
        self.update_db = UpdateLocalDB()

    @staticmethod
    def _normalize_trade_date(df: pl.DataFrame) -> pl.DataFrame:
        if "trade_date" not in df.columns:
            return df
        return df.with_columns(
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
    
    def fetch_shibor_daily(self, start: date, end: date):
        file_path = LOCAL_DATA_DIR / "market_data" / f"shibor_daily.parquet"

        update_ranges = self.update_db.get_update_range(
            file=file_path,
            start=start,
            end=end,
        )

        for _start, _end in update_ranges:
            _df = self.ts.get_shibor_daily(start=_start, end=_end)
            df = pl.from_arrow(_df)
            if df.height == 0:
                continue
            df = self._normalize_trade_date(df)
            df = df.filter(pl.col("trade_date").is_not_null())
            if df.height == 0:
                continue
            self.update_db.update_old(
                file=file_path,
                df=df,
                unique_subset=["trade_date"],
            )
   
        return




if __name__ == "__main__":

    from core.config_shema import load_confg
    cfg = load_confg("config/shfe_ag_demo.yaml")[0]

    future_fetcher = FutureFetcher(cfg.future)
    future_fetcher.update_future_basic(fut_code=cfg.future.fut_code)
    ts_code = 'AG2603.SHF'
    future_fetcher.fetch_future_kline_daily(
        ts_code=ts_code,
        start=cfg.backtest.start_date,
        end=cfg.backtest.end_date,
    )
    future_fetcher.fetch_future_kline_minute(
        ts_code=ts_code,
        start=cfg.backtest.start_date,
        end=cfg.backtest.end_date,
        freq='15min',
    )

    opt_code = "OPAG2612.SHF"
    ts_code = 'AG2612C30400.SHF'
    option_fetcher = OptionFetcher(cfg.future)
    option_fetcher.update_option_basic(opt_code=opt_code)
    option_fetcher.fetch_option_kline_daily(
        ts_code=ts_code,
        start=cfg.backtest.start_date,
        end=cfg.backtest.end_date,
    )
    option_fetcher.fetch_option_kline_minute(
        ts_code=ts_code,
        start=cfg.backtest.start_date,
        end=cfg.backtest.end_date,
        freq='15min',
    )

    market_data_fetcher = MarketDataFetcher()
    market_data_fetcher.fetch_shibor_daily(
        start=cfg.backtest.start_date,
        end=cfg.backtest.end_date,
    )