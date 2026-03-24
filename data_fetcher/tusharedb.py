import os
import time
from datetime import date, timedelta, datetime
from typing import Literal
from pathlib import Path
import functools

import pandas as pd
import requests
import tushare as ts
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "local_db"


def _load_env_file(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env

def retry_on_exception(decorated):
    def wrapper(*args, **kwargs):
        retries = 0
        while True:
            try:
                return decorated(*args, **kwargs)
            except (IOError, requests.HTTPError) as e:
                print(f"Tushare 发生I/O错误正在重试({retries}/10) {repr(e)}")
                retries += 1
                if retries > 10:
                    raise
                time.sleep(10)

    return wrapper

def retry_on_tushare_limit(max_retry: int = 1000, sleep_seconds: int = 60):
    """
    当tushare触发每分钟访问限制时自动等待并重试
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            for attempt in range(max_retry):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e)
                    if "每分钟最多访问" in err_msg or "rate limit" in err_msg:
                        print(
                            f"Tushare接口限速，等待 {sleep_seconds}s 后重试 "
                            f"({attempt+1}/{max_retry})"
                        )
                        time.sleep(sleep_seconds)
                    else:
                        raise e
            raise RuntimeError("达到最大重试次数")
        return wrapper

    return decorator



class TuShare(object):

    def __init__(self):
        env_values = _load_env_file(PROJECT_ROOT / ".env")
        token = os.getenv("TUSHARE_TOKEN") or env_values.get("TUSHARE_TOKEN")
        if not token:
            raise ValueError(
                "未找到 TUSHARE_TOKEN。请在项目根目录 .env 中配置 "
                "`TUSHARE_TOKEN=your_token`（可参考 .env.example）。"
            )
        ts.set_token(token)
        self.ts = ts
        self.pro = ts.pro_api()
        # self.pro = ts.pro_api(token)
        # self.pro._DataApi__token = token
        # self.pro._DataApi__http_url = 'http://118.25.178.42:5000' 

    def parse_date_to_str(self, date_obj: date | str) -> str:
        """转换date类型， ‘2026-03-02’ 至日期字符 ‘20260302’

        :param date | str date_obj:
        :return str:
        """
        if isinstance(date_obj, date):
            date_str = date_obj.strftime("%Y%m%d")

        if isinstance(date_obj, str):
            if "-" in date_obj:
                date_str = datetime.strptime(date_obj, "%Y-%m-%d").strftime("%Y%m%d")
            if len(date_obj) == 8 and date_obj.isdigit():
                date_str = date_obj

        return date_str
    
    @retry_on_exception
    def get_future_basic(
        self,
        exchange: str = None,
        fut_code: str = None,
        fields: list[str] = None,
    ) -> pd.DataFrame:
        """
        获取期货合约信息
        :param str exchange: 交易所代码 （包括上交所SSE等交易所）
        :param str fut_code: 期货合约代码
        :param list[str] fields: 字段列表
        :return pd.DataFrame:
        """
        if fields is None:
            fields = [  
                "ts_code",
                "symbol",
                "exchange",
                "name",
                "fut_code",
                "multiplier",
                "trade_unit",
                "per_unit",
                "quote_unit",
                "quote_unit_desc",
                "d_mode_desc",
                "list_date",
                "delist_date",
                "d_month",
                "last_ddate",
                "trade_time_desc",
            ]
        return self.pro.fut_basic(exchange=exchange, fut_code=fut_code, fields=fields)

    @retry_on_exception
    def get_option_basic(
        self,
        ts_code: str = None,
        exchange: str = None,
        opt_code: str = None,
        call_put: str = None,
        fileds: list[str] = None,
    ) -> pd.DataFrame:
        """获取期权合约信息

        :param str ts_code: TS期权代码
        :param str exchange: 交易所代码 （包括上交所SSE等交易所）
        :param str opt_code: 标准合约代码，OP+期货合约TS_CODE，如棕榈油2207合约，输入OPP2207.DCE
        :param str call_put: 期权类型
        :return pd.DataFrame:
        """
        if fileds is None:
            fileds = [
                "ts_code",
                "exchange",
                "name",
                "per_unit",
                "opt_code",
                "opt_type",
                "call_put",
                "exercise_type",
                "exercise_price",
                "s_month",
                "maturity_date",
                "list_date",
                "delist_date",
                "last_ddate",
            ]

        df = self.pro.opt_basic(
            ts_code=ts_code,
            exchange=exchange,
            opt_code=opt_code,
            call_put=call_put,
            fields=",".join(fileds),
        )

        return df

    @retry_on_tushare_limit()
    def get_option_daily(
        self,
        ts_code: str,
        start: date,
        end: date,
        fields: list[str] = None,
    ) -> pd.DataFrame:
        """获取期权日线行情

        :param str ts_code: _description_
        :param date start: _description_
        :param date end: _description_
        :param list[str] fields: _description_, defaults to None
        :return pd.DataFrame: _description_
        """

        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)

        if fields is None:
            fields = [
                "ts_code",
                "trade_date",
                "exchange",
                "pre_settle",
                "pre_close",
                "open",
                "high",
                "low",
                "close",
                "settle",
                "vol",
                "amount",
                "oi",
            ]

        df = self.pro.opt_daily(
            ts_code=ts_code, start_date=start, end_date=end, fields=fields
        )

        return df

    @retry_on_tushare_limit()
    def get_option_minute(
        self,
        ts_code: str,
        freq: Literal["1min", "5min", "15min", "30min", "60min"],
        start: date,
        end: date,
        start_time: str = "00:00:00",
        end_time: str = "23:59:59",
    ) -> pd.DataFrame:
        """获取期权分钟线行情

        :param str ts_code: _description_
        :param Literal["1min", "5min", "15min", "30min", "60min"] freq: _description_
        :param date start: _description_
        :param date end: _description_
        :param _type_ start_time: _description_, defaults to "00:00:00"
        :param _type_ end_time: _description_, defaults to "23:59:59"
        :return pd.DataFrame: _description_
        """

        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)
        start_str = f"{start} {start_time}"
        end_str = f"{end} {end_time}"

        df = self.pro.opt_mins(
            ts_code=ts_code, freq=freq, start_date=start_str, end_date=end_str
        )

        return df
    
    @retry_on_exception
    def get_index_kline_daily(
        self,
        ts_code: str,
        start: date,
        end: date,
        fields: list[str] = None,
    ) -> pd.DataFrame:
        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)
        if fields is None:
            fields = [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pct_chg",
            ]
        
        df = self.pro.index_daily(
            ts_code=ts_code, start_date=start, end_date=end, fields=fields
        )

        # 涨跌幅 % -> 涨跌幅 decimal
        if "pct_chg" in df.columns:
            df["pct_chg"] = df["pct_chg"] / 100

        return df

    @retry_on_exception
    def get_index_kline_minute(
        self,
        ts_code: str,
        freq: Literal["1min", "5min", "15min", "30min", "60min"],
        start: date,
        end: date,
        start_time: str = "00:00:00",
        end_time: str = "23:59:59",
    ) -> pd.DataFrame:
        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)
        start_str = f"{start} {start_time}"
        end_str = f"{end} {end_time}"

        df = self.pro.stk_mins(
            ts_code=ts_code, freq=freq, start_date=start_str, end_date=end_str
        )

         # 涨跌幅 % -> 涨跌幅 decimal
        if "pct_chg" in df.columns:
            df["pct_chg"] = df["pct_chg"] / 100
        
        return df

    @retry_on_exception
    def get_future_kline_daily(
        self,
        ts_code: str,
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)
        if fields is None:
            fields = [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "settle",
                "change1",
                "change2",
                "vol",
                "amount",
                "oi",
                "oi_chg",
                "delv_settle",
            ]
        df = self.pro.fut_daily(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields=",".join(fields),
        )
        if "pct_chg" in df.columns:
            df["pct_chg"] = df["pct_chg"] / 100
        return df

    @retry_on_exception
    def get_future_kline_minute(
        self,
        ts_code: str,
        freq: Literal["1min", "5min", "15min", "30min", "60min"],
        start: date,
        end: date,
        start_time: str = "00:00:00",
        end_time: str = "23:59:59",
    ) -> pd.DataFrame:
        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)
        start_str = f"{start} {start_time}"
        end_str = f"{end} {end_time}"

        return self.pro.ft_mins(
            ts_code=ts_code,
            freq=freq,
            start_date=start_str,
            end_date=end_str,
        )
    
    @retry_on_exception
    def get_shibor_daily(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        start = self.parse_date_to_str(start)
        end = self.parse_date_to_str(end)
        
        df = self.pro.shibor(
            start_date=start, end_date=end
        )

        rate_cols = ['on', '1w', '2w', '1m', '3m', '6m', '9m', '1y']
        df[rate_cols] = df[rate_cols] / 100
        df.rename(columns={"date": "trade_date"}, inplace=True)

        return df



def fetch_option_basic():
    opt_basic_all = ts.get_option_basic()
    if not LOCAL_DATA_DIR.exists():
        LOCAL_DATA_DIR.mkdir()
    opt_basic_all.to_csv(LOCAL_DATA_DIR / "opt_basic_all.csv", index=False)

    opt_basic_index_option = opt_basic_all[opt_basic_all['opt_type'] == '指数期权']
    opt_basic_index_option.to_csv(LOCAL_DATA_DIR / "opt_basic_index_option.csv", index=False)

    opt_basic_CFFEX = ts.get_option_basic(exchange='CFFEX')
    opt_basic_CFFEX.to_csv(LOCAL_DATA_DIR / "opt_basic_CFFEX.csv", index=False)
    return


def main():

    ts = TuShare()
    start_date = date(2026, 3, 1)
    end_date = date(2026, 3, 6)

    # fetch_option_basic()
    ts.pro.ft_mins(ts_code='CU2310.SHF', freq='15min', start_date='2023-08-25 00:00:00', end_date='2023-08-25 19:00:00')
    ts.pro.idx_mins(ts_code='000001.SH', freq='1min', start_date='2023-08-25 00:00:00', end_date='2023-08-25 19:00:00')
    ts.pro.pro_bar(ts_code='000001.SZ', adj='hfq', start_date='20180101', end_date='20181011')
    ts.pro.stk_mins(ts_code='000300.SH', freq='15min', start_date='2023-08-25 00:00:00', end_date='2023-08-25 19:00:00')
    ts.pro.shibor(start_date='20260301', end_date='20260306')

    df_opt_basic = ts.pro.opt_basic(opt_code='OPAG')
    ts.pro.fut_basic(exchange='SHFE', fut_code='AG')
    
    df_opt_basic.to_csv(LOCAL_DATA_DIR / "opt_basic_SHFE.csv", index=False)

    ts.get_option_basic()

    opt_kline = ts.get_option_minute(
        ts_code='IO2606-C-3400.CFX',
        freq='15min',
        start=start_date,
        end=end_date,
    )

    return

















if __name__ == "__main__":
    main()



