"""
Silver layer

Purpose:Cleans the bronze data and gives each column its proper data type. Each source
still gets its own table here, nothing is joined yet.

- Bad values are set to null instead of being filled in. Filling missing
  values is a modelling choice, so it happens during training (assignment 2)
  using only the training data.
- Cleaning uses fixed rules only, like "age must be between 0 and 100".
  No averages or percentiles from the whole dataset, because that would let
  future data leak into the past.
- Name and SSN are dropped here so they never reach gold.
- Saved as parquet instead of csv because it keeps the data types and is
  smaller and faster to read.
"""

import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ---------------------------------------------------------------
# Cleaning rules (kept in one place so they are easy to review / change)
# ---------------------------------------------------------------

# valid range for numeric columns; anything outside -> NULL
VALID_RANGES = {
    "Age":                    (0, 100),
    "Num_Bank_Accounts":      (0, 20),
    "Num_Credit_Card":        (0, 20),
    "Interest_Rate":          (0, 50),     # in %
    "Num_of_Loan":            (0, 10),
    "Num_of_Delayed_Payment": (0, 50),
    "Num_Credit_Inquiries":   (0, 40),
}

# allowed values for categorical columns; anything else -> NULL
VALID_CATEGORIES = {
    "Occupation": ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
                   "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
                   "Accountant", "Musician", "Mechanic", "Writer", "Architect"],
    "Credit_Mix": ["Bad", "Standard", "Good"],
    "Payment_of_Min_Amount": ["Yes", "No"],          # "NM" = not mentioned -> NULL
    "Payment_Behaviour": ["Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
                          "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
                          "High_spent_Medium_value_payments", "High_spent_Large_value_payments"],
}

# placeholder values the source system uses to mean "missing"
SENTINELS = {
    "Amount_invested_monthly": "__10000__",
    "Monthly_Balance":         "__-333333333333333333333333333__",
}


# ---------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------

def clean_numeric(df, column, dtype):
    """Strip stray underscores (e.g. '52312.68_'), cast, then apply the valid range."""
    df = df.withColumn(column, F.regexp_replace(col(column), "_", "").cast(dtype))
    if column in VALID_RANGES:
        low, high = VALID_RANGES[column]
        df = df.withColumn(column, F.when(col(column).between(low, high), col(column)))
    return df


def clean_category(df, column):
    """Keep only known category values; junk like '_' or '!@9#%8' becomes NULL."""
    return df.withColumn(column, F.when(col(column).isin(VALID_CATEGORIES[column]), col(column)))


def read_bronze(bronze_directory, source_name, snapshot_date_str, spark):
    partition = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, source_name, partition)
    return spark.read.csv(filepath, header=True, inferSchema=False)


def write_silver(df, silver_directory, source_name, snapshot_date_str):
    output_dir = os.path.join(silver_directory, source_name)
    os.makedirs(output_dir, exist_ok=True)
    partition = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(output_dir, partition)
    df.write.mode("overwrite").parquet(filepath)
    print(f"[silver] {source_name:<12} {snapshot_date_str}  rows={df.count():>6}  -> {filepath}")
    return df


# ---------------------------------------------------------------
# One function per source
# ---------------------------------------------------------------

def process_silver_lms(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Loan repayments: enforce types, add months-on-book (mob) and days-past-due (dpd). From Lab 2."""
    df = read_bronze(bronze_directory, "lms", snapshot_date_str, spark)

    column_type_map = {
        "loan_id": StringType(), "Customer_ID": StringType(), "loan_start_date": DateType(),
        "tenure": IntegerType(), "installment_num": IntegerType(), "loan_amt": FloatType(),
        "due_amt": FloatType(), "paid_amt": FloatType(), "overdue_amt": FloatType(),
        "balance": FloatType(), "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # months on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0,
                       F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0,
                       F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    return write_silver(df, silver_directory, "lms", snapshot_date_str)


def process_silver_attributes(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Customer attributes: drop PII, clean Age and Occupation."""
    df = read_bronze(bronze_directory, "attributes", snapshot_date_str, spark)

    df = df.drop("Name", "SSN")                      # PII never leaves bronze
    df = clean_numeric(df, "Age", IntegerType())
    df = clean_category(df, "Occupation")
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])

    return write_silver(df, silver_directory, "attributes", snapshot_date_str)


def process_silver_financials(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Customer financials: the dirtiest source."""
    df = read_bronze(bronze_directory, "financials", snapshot_date_str, spark)

    # 1. sentinels first - otherwise stripping '_' would turn '__10000__' into a real-looking 10000
    for column, sentinel in SENTINELS.items():
        df = df.withColumn(column, F.when(col(column) != sentinel, col(column)))

    # 2. numeric columns: strip '_', cast, apply valid range
    float_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit", "Outstanding_Debt",
                  "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]
    int_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]
    for c in float_cols:
        df = clean_numeric(df, c, FloatType())
    for c in int_cols:
        df = clean_numeric(df, c, IntegerType())

    # 3. cross-field sanity checks (row-level rules, no dataset statistics)
    #    annual income can't be more than 3x the yearly take-home salary -> corrupted
    df = df.withColumn("Annual_Income", F.when(
        col("Annual_Income") <= 3 * 12 * col("Monthly_Inhand_Salary"), col("Annual_Income")))
    #    monthly loan repayments (EMI) can't exceed monthly salary -> corrupted
    df = df.withColumn("Total_EMI_per_month", F.when(
        col("Total_EMI_per_month") <= col("Monthly_Inhand_Salary"), col("Total_EMI_per_month")))

    # 4. categorical columns: keep known values only
    for c in ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]:
        df = clean_category(df, c)

    # 5. "10 Years and 9 Months" -> 129 (months)
    years = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType())
    months = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType())
    df = df.withColumn("Credit_History_Months", years * 12 + months).drop("Credit_History_Age")

    # 6. "Auto Loan, Payday Loan, and Student Loan" -> ["Auto Loan", "Payday Loan", "Student Loan"]
    df = df.withColumn("Type_of_Loan", F.split(F.regexp_replace(col("Type_of_Loan"), "and ", ""), ",\\s*"))
    #    if Num_of_Loan was invalid, recover it from the number of loan types listed
    df = df.withColumn("Num_of_Loan", F.coalesce(
        col("Num_of_Loan"),
        F.when(col("Type_of_Loan").isNull(), F.lit(0)).otherwise(F.size(col("Type_of_Loan")))))

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])

    return write_silver(df, silver_directory, "financials", snapshot_date_str)


def process_silver_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Clickstream: already clean - just enforce types and remove duplicates."""
    df = read_bronze(bronze_directory, "clickstream", snapshot_date_str, spark)

    for c in [f"fe_{i}" for i in range(1, 21)]:
        df = df.withColumn(c, col(c).cast(IntegerType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])

    return write_silver(df, silver_directory, "clickstream", snapshot_date_str)
