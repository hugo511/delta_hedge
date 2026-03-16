



import os
import glob
import polars as pl

def check_option_price_nulls(dir_path, threshold=0.05):
    """
    检查指定路径下所有parquet文件，若'ts_code', 'trade_date', 'close'中任一列的空值比例高于threshold，则记录该文件路径
    :param dir_path: 检查的文件夹路径
    :param threshold: 空值比例阈值，默认0.05（5%）
    :return: 问题文件列表
    """
    parquet_files = glob.glob(os.path.join(dir_path, "*.parquet"))
    bad_files = []

    for file in parquet_files:
        try:
            df = pl.read_parquet(file)
            n_rows = df.height
            if n_rows == 0:
                # 文件无数据也算一个问题
                bad_files.append(file)
                continue

            for col in ["ts_code", "trade_date", "close"]:
                if col in df.columns:
                    null_count = df[col].null_count()
                    null_ratio = null_count / n_rows
                    if null_ratio > threshold:
                        bad_files.append(file)
                        break
                else:
                    # 缺失必需列也记为问题文件
                    bad_files.append(file)
                    break
        except Exception as e:
            print(f"读取文件出错: {file}, 错误: {e}")
            bad_files.append(file)

    print("空值比例过高的文件有：")
    for bad_file in bad_files:
        print(bad_file)

if __name__ == "__main__":
    check_option_price_nulls("local_db/option_price_minute/SHFE_AG", threshold=0.7)