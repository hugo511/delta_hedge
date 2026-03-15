from loguru import logger
import os
import sys
import time
from functools import partial, wraps
from typing import Callable, Any, Optional, Literal

LOGURU_LEVEL = os.getenv("LOGURU_LEVEL", "INFO")
os.environ["LOGURU_LEVEL"] = LOGURU_LEVEL


def timer_decorator(func: Callable) -> Callable:
    """
     简单计时装饰器，记录函数执行时间，并打印函数的传参值

    :param func: 被测试函数
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # 记录开始时间
        start_time = int(time.time())

        # 打印传参信息
        args_str = ", ".join(repr(a) for a in args)
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())

        # 执行被装饰的函数
        result = func(*args, **kwargs)
        end_time = int(time.time())
        # 计算耗时
        elapsed_time = end_time - start_time
        logger.info(
            f"调用函数 {func.__name__} | 参数: args=({args_str}), kwargs={{ {kwargs_str} }},{func.__name__},"
            f"开始时间:{start_time},结束时间{end_time}, 操作耗时 {elapsed_time:.3f} 秒"
        )

        return result

    return wrapper


def filter_log_level(record, level):
    return record["level"].name == level


class Logger(object):
    def __init__(
            self, name: str,
            log_dir: Optional[str] = None,
            retention: int = 5,
            monitor_type: Literal["Feishu"] = "Feishu"
    ):
        """
        日志记录器

        :param name: logger名称
        :param log_dir: 输出目录, 无输出目录时不会记录log到文件
        :param retention: 保留天数
        :param monitor_type: 是否启用监控报警
        """
        if log_dir and not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir
        self.retention = retention
        self.name = name
        self.log_format = f"<green>[{{time}}]</green> <yellow>[{{process}}]</yellow> <level>[{{level}}]</level> <blue>[{{file}}]</blue> <magenta>[{{line}}]</magenta> <cyan>[{name}]</cyan> <level>{{message}}</level>"
        self.trace = logger.trace
        self.debug = logger.debug
        self.info = logger.info
        self.warning = logger.warning
        self.error = logger.error
        self.exception = logger.exception
        self.min_level = LOGURU_LEVEL
        self.min_level_no = logger.level(LOGURU_LEVEL).no  # 获取最小级别的数值
        self.monitor_type = monitor_type
        self.monitor = None

        # 定义支持的日志级别
        levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
        handlers = []

        # 为每个符合条件的级别添加文件处理器
        if self.log_dir:
            for level in levels:
                if logger.level(level).no >= self.min_level_no:
                    handlers.append({
                        "sink": os.path.join(self.log_dir, f"{level.lower()}.log"),
                        "level": level,
                        "format": self.log_format,
                        "rotation": "1 days",
                        "enqueue": True,
                        "filter": partial(filter_log_level, level=level),
                        "retention": f"{self.retention} days",
                    })

        # 添加控制台处理器，显示 min_level 及以上级别的所有日志
        handlers.append({
            "sink": sys.stdout,
            "level": self.min_level,
            "format": self.log_format,
            "enqueue": True,
            # 无 filter，记录 min_level 及以上所有消息
        })

        logger.configure(handlers=handlers)


def log_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"------------Begin: {func.__name__}------------")
        try:
            rsp = func(*args, **kwargs)
            logger.info(f"result: {rsp}")
            end = time.time()
            logger.info(f"Time consuming: {end - start}s")
            logger.info(f"------------{func.__name__}------------")
            return rsp
        except Exception as e:
            logger.error(repr(e))
            raise e

    return wrapper


log_debug = logger.debug
log_info = logger.info
log_warning = logger.warning
log_error = logger.error

dir_path = os.path.join('./', "logs")
logger = Logger('delta_hedge', dir_path, retention=30)


if __name__ == '__main__':
    log_debug('这是debug')
    log_info('这是info')
    log_warning('这是warning')
    log_error('这是error')

