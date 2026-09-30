"""
Bronze layer

Purpose: Copies the raw data into the datamart as-is, one file per source per month.

- Nothing is cleaned or changed here. All columns are read as text so messy
  values like "40_" or "!@9#%8" stay exactly how they came in. If something
  goes wrong in silver later, we can just rebuild it from here.
- Each month gets its own file, like a monthly batch job at a bank. This means
  we can re-run one month without redoing everything.
- Name and SSN are still in here since this is the raw copy. They get dropped
  in silver.
"""

import os
from pyspark.sql.functions import col


def process_bronze_table(snapshot_date_str, source_name, source_filepath, bronze_directory, spark):
    """
    Ingest one month of one raw source into the bronze layer.

    snapshot_date_str : "YYYY-MM-DD", the month being ingested
    source_name       : short name used for the folder / file, e.g. "financials"
    source_filepath   : path to the raw csv, e.g. "data/features_financials.csv"
    bronze_directory  : root bronze folder, e.g. "datamart/bronze/"
    """
    # 1. read the raw source (IRL: query the source system for this month only)
    df = spark.read.csv(source_filepath, header=True, inferSchema=False) \
              .filter(col("snapshot_date") == snapshot_date_str)

    # 2. write the partition, unchanged, in the same format as the source (csv)
    output_dir = os.path.join(bronze_directory, source_name)
    os.makedirs(output_dir, exist_ok=True)

    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(output_dir, partition_name)
    df.toPandas().to_csv(filepath, index=False)

    print(f"[bronze] {source_name:<12} {snapshot_date_str}  rows={df.count():>6}  -> {filepath}")
    return df
