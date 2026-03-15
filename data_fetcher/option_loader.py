from typing import Literal
import polars as pl


from core.config_shema import FutureConfig, BacktestConfig, HedgeConfig, ExperimentConfig
from data_fetcher.option_fetcher import FutureFetcher, OptionFetcher, MarketDataFetcher
from data_fetcher.tusharedb import LOCAL_DATA_DIR
from utils.logger import log_info






class OptionLoader(object):
    def __init__(self, cfg_exp: ExperimentConfig):
        self.cfg_exp = cfg_exp
        self.cfg_future = cfg_exp.future
        self.cfg_backtest = cfg_exp.backtest
        self.cfg_hedge = cfg_exp.hedge
        self.fut_fetcher = FutureFetcher(self.cfg_future)
        self.opt_fetcher = OptionFetcher(self.cfg_future)
        self.market_fetcher = MarketDataFetcher()
    
    def load_shibor_daily(self):
        file_path = LOCAL_DATA_DIR / "market_data" / f"shibor_daily.parquet"
        return pl.read_parquet(file_path)

    def load_future_basic(self, transform_date: bool = False) -> pl.DataFrame:
        df_fut_basic = self.fut_fetcher.update_future_basic(
            self.cfg_future.fut_code, self.cfg_future.update_basic
        )
        if transform_date:
            df_fut_basic = df_fut_basic.with_columns(
                pl.col("list_date").str.strptime(pl.Date, '%Y%m%d'),
                pl.col("delist_date").str.strptime(pl.Date, '%Y%m%d'),
                pl.col('d_month').str.strptime(pl.Date, '%Y%m'),
                pl.col('last_ddate').str.strptime(pl.Date, '%Y%m%d'),
            )
        return df_fut_basic
    
    def load_future_bar(self, ts_code:str, freq: Literal['1d', '15min']):
        if freq == '1d':
            data_file = (
                LOCAL_DATA_DIR 
                / "future_price_daily" 
                / f"{self.cfg_future.exchange}_{self.cfg_future.fut_code}"
                / f"{ts_code}_fut_daily.parquet")
        elif freq == '15min':
            data_file = (
                LOCAL_DATA_DIR 
                / "future_price_minute" 
                / f"{self.cfg_future.exchange}_{self.cfg_future.fut_code}"
                / f"{ts_code}_fut_minute.parquet")
        else:
            raise ValueError("目前路径存有的行情数据频率为1d和15min, 请检查freq参数")
        return pl.read_parquet(data_file)
    
    def load_option_bar(self, ts_code:str, freq: Literal['1d', '15min']):
        if freq == '1d':
            data_file = (LOCAL_DATA_DIR 
                / "option_price_daily" 
                / f"{self.cfg_future.exchange}_{self.cfg_future.fut_code}"
                / f"{ts_code}_opt_daily.parquet"
            )
        elif freq == '15min':
            data_file = (LOCAL_DATA_DIR 
                / "option_price_minute" 
                / f"{self.cfg_future.exchange}_{self.cfg_future.fut_code}"
                / f"{ts_code}_opt_minute.parquet"
            )
        else:
            raise ValueError("目前路径存有的行情数据频率为1d和15min, 请检查freq参数")
        return pl.read_parquet(data_file)

    def load_option_basic(self, transform_date: bool = False) -> pl.DataFrame:
        file_path = (LOCAL_DATA_DIR 
            / "contract_info" 
            / "option_basic" 
            / f"{self.cfg_future.exchange}_{self.cfg_future.fut_code}_opt_basic.parquet"
        )
        df_opt_basic = pl.read_parquet(file_path)
        if transform_date:
            df_opt_basic = df_opt_basic.with_columns(
                pl.col("maturity_date").str.strptime(pl.Date, '%Y%m%d'),
                pl.col("list_date").str.strptime(pl.Date, '%Y%m%d'),
                pl.col("delist_date").str.strptime(pl.Date, '%Y%m%d'),
                pl.col('s_month').str.strptime(pl.Date, '%Y%m'),
                pl.col('last_ddate').str.strptime(pl.Date, '%Y%m%d'),
            )
        return df_opt_basic
    
    def update_future_price_to_local_db(self):
        df_fut_basic = self.load_future_basic(transform_date=True)
        df_fut_contract = df_fut_basic.filter(
            (pl.col("list_date") <= self.cfg_backtest.end_date) & 
            (pl.col("delist_date") >= self.cfg_backtest.start_date)
        )

        need_daily = self.cfg_hedge.frequency == '1d'
        need_minute = self.cfg_hedge.frequency == '15min'

        for i, row in enumerate(df_fut_contract.iter_rows(named=True)):
            ts_code = row['ts_code']

            if need_daily:
                self.fut_fetcher.fetch_future_kline_daily(
                    ts_code=ts_code,
                    start=self.cfg_backtest.start_date,
                    end=self.cfg_backtest.end_date,
                    list_date=row['list_date'],
                    delist_date=row['delist_date'],
                )
            if need_minute:
                self.fut_fetcher.fetch_future_kline_minute(
                    ts_code=ts_code,
                    start=self.cfg_backtest.start_date,
                    end=self.cfg_backtest.end_date,
                    freq='15min',
                    list_date=row['list_date'],
                    delist_date=row['delist_date'],
                )
        self.market_fetcher.fetch_shibor_daily(
            start=self.cfg_backtest.start_date,
            end=self.cfg_backtest.end_date,
        )

        return     

    def update_option_price_to_local_db(self):
        # 挑出backtest start_date 到 end_date之间的有效期货合约
        df_fut_basic = self.load_future_basic(transform_date=True)
        df_fut_contract = df_fut_basic.filter(
            (pl.col("list_date") <= self.cfg_backtest.end_date) & 
            (pl.col("delist_date") >= self.cfg_backtest.start_date)
        )

        # 更新期权合约信息
        for i, ts_code in enumerate(df_fut_contract['ts_code']):
            opt_code = f'OP{ts_code}'
            self.opt_fetcher.update_option_basic(
                opt_code=opt_code,
                update_basic=self.cfg_future.update_basic,
            )

        need_daily = self.cfg_hedge.frequency == '1d'
        need_minute = self.cfg_hedge.frequency == '15min'
        
        df_opt_basic = self.load_option_basic(transform_date=True)
        for i, row in enumerate(df_opt_basic.iter_rows(named=True)):
            ts_code = row['ts_code']
            if need_daily:
                self.opt_fetcher.fetch_option_kline_daily(
                    ts_code=ts_code,
                    start=self.cfg_backtest.start_date,
                    end=self.cfg_backtest.end_date,
                    list_date=row['list_date'],
                    delist_date=row['delist_date'],
                )
            if need_minute:
                self.opt_fetcher.fetch_option_kline_minute(
                    ts_code=ts_code,
                    start=self.cfg_backtest.start_date,
                    end=self.cfg_backtest.end_date,
                    freq='15min',
                    list_date=row['list_date'],
                    delist_date=row['delist_date']
                )
            
            if i % 500 == 0:
                log_info(f'已更新{i}个期权合约， 共{df_opt_basic.shape[0]}个期权合约')
        
        return
                







if __name__ == '__main__':

    from core.config_shema import load_config
    cfg_exp = load_config('config/shfe_ag_demo.yaml')[0]
    option_loader = OptionLoader(cfg_exp)
    # option_loader.update_future_price_to_local_db()
    option_loader.update_option_price_to_local_db()
