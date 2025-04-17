# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars",
# ]
# ///

import os
import sys

import polars as pl

# Usage: python csv-convert.py <parquet-data-dir> <csv-data-dir>


def convert(parquet_data_dir, csv_data_dir):
    os.makedirs(csv_data_dir, exist_ok=True)
    for parquet_file in os.listdir(parquet_data_dir):
        print(parquet_file)
        if parquet_file != "Globals_tick.parquet" and parquet_file.endswith(".parquet"):
            df = pl.read_parquet(os.path.join(parquet_data_dir, parquet_file))
            print(df)
            df.write_csv(os.path.join(csv_data_dir, parquet_file.replace(".parquet", ".csv")))


def main():
    if len(sys.argv) < 3:
        print("Usage: python csv-convert.py <parquet-data-dir> <csv-data-dir>")
        sys.exit(1)
    parquet_data_dir = sys.argv[1]
    csv_data_dir = sys.argv[2]
    convert(parquet_data_dir, csv_data_dir)


if __name__ == "__main__":
    main()
