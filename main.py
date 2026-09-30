"""
main.py - runs the full data pipeline: raw data -> bronze -> silver -> gold
"""

import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"   # last month that has application-time features

DATAMART = "datamart/"
BRONZE_DIR = DATAMART + "bronze/"
SILVER_DIR = DATAMART + "silver/"
GOLD_DIR = DATAMART + "gold/"

# raw sources: short name -> file path
SOURCES = {
    "lms":         "data/lms_loan_daily.csv",
    "attributes":  "data/features_attributes.csv",
    "financials":  "data/features_financials.csv",
    "clickstream": "data/feature_clickstream.csv",
}


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Return ['2023-01-01', '2023-02-01', ...] up to and including end date."""
    current = datetime.strptime(start_date_str, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=1)
    return dates


def main():
    spark = pyspark.sql.SparkSession.builder \
        .appName("cs611_assignment1") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    dates = generate_first_of_month_dates(START_DATE, END_DATE)
    print(f"Processing {len(dates)} months: {dates[0]} to {dates[-1]}")

    # ---------------- BRONZE Layer ----------------
    for source_name, source_path in SOURCES.items():
        for date_str in dates:
            bronze.process_bronze_table(date_str, source_name, source_path, BRONZE_DIR, spark)

    # ---------------- SILVER Layer ----------------
    silver_jobs = {
        "lms":         silver.process_silver_lms,
        "attributes":  silver.process_silver_attributes,
        "financials":  silver.process_silver_financials,
        "clickstream": silver.process_silver_clickstream,
    }
    for source_name, silver_job in silver_jobs.items():
        for date_str in dates:
            silver_job(date_str, BRONZE_DIR, SILVER_DIR, spark)

    # ---------------- GOLD Layer ------------------
    for date_str in dates:
        gold.process_gold_label_store(date_str, SILVER_DIR, GOLD_DIR, spark, dpd=30, mob=6)
    for date_str in dates:
        gold.process_gold_feature_store(date_str, SILVER_DIR, GOLD_DIR, spark)

    spark.stop()
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
