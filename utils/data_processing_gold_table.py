"""
Gold layer

Builds the two tables that a model can be trained on:

1. Label store - did the loan default? One row per loan, taken at month 6
   of the loan.
2. Feature store - what we knew about the customer when they applied.
   One row per loan, taken at the month they applied.

Both tables have loan_id, so they can be joined with:
    features.join(labels, on="loan_id")

Main rule: only use data from the time of application.
The model has to predict default when the customer applies, so features for
a loan applied in a given month can only use data from that month or before.
Clickstream data keeps coming in after the loan starts. Using it would mean
using information we wouldn't have had at the time.

Other notes:
- No scaling or filling of missing values here. Those should be done during
  model training, based on the training data only.
- Categories are turned into numbers using fixed lists we wrote ourselves,
  not lists learned from the data.
- All feature columns start with fe_ plus where they came from (fe_attr_,
  fe_fin_, fe_click_). The model can pick them all up with
  startswith("fe_"), and it's easy to see which source each one is from.
"""
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

def read_silver(silver_directory, source_name, snapshot_date_str, spark):
    """Read one silver partition. Returns None if that month doesn't exist."""
    partition = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, source_name, partition)
    if not os.path.exists(filepath):
        return None
    return spark.read.parquet(filepath)


def write_gold(df, gold_directory, table_name, snapshot_date_str):
    output_dir = os.path.join(gold_directory, table_name)
    os.makedirs(output_dir, exist_ok=True)
    partition = f"gold_{table_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(output_dir, partition)
    df.write.mode("overwrite").parquet(filepath)
    print(f"[gold]   {table_name:<13} {snapshot_date_str}  rows={df.count():>6}  -> {filepath}")
    return df


def safe_divide(numerator, denominator):
    return F.when(denominator > 0, numerator / denominator)


# ---------------------------------------------------------------
# 1. LABEL STORE  (from Lab 2)
# ---------------------------------------------------------------

def process_gold_label_store(snapshot_date_str, silver_directory, gold_directory, spark, dpd=30, mob=6):
    """
    Label = 1 if the loan is `dpd`+ days past due when it is `mob` months old.
    Default: 30 days past due at month 6 ("30dpd_6mob").
    """
    df = read_silver(silver_directory, "lms", snapshot_date_str, spark)

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    return write_gold(df, gold_directory, "label_store", snapshot_date_str)


# ---------------------------------------------------------------
# 2. FEATURE STORE
# ---------------------------------------------------------------

# fixed category lists -> encodings don't depend on the data
CREDIT_MIX_ORDER = {"Bad": 0, "Standard": 1, "Good": 2}
MIN_PAYMENT_MAP = {"No": 0, "Yes": 1}
SPEND_LEVEL_MAP = {"Low": 0, "High": 1}
PAYMENT_VALUE_MAP = {"Small": 0, "Medium": 1, "Large": 2}
OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer",
               "Media_Manager", "Doctor", "Journalist", "Manager", "Accountant", "Musician",
               "Mechanic", "Writer", "Architect"]
LOAN_TYPES = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
              "Mortgage Loan", "Payday Loan", "Personal Loan", "Student Loan", "Not Specified"]


def map_values(column, mapping):
    """Encode a string column with a fixed dict, e.g. {'Bad':0, 'Standard':1, 'Good':2}."""
    expr = None
    for key, value in mapping.items():
        expr = F.when(column == key, value) if expr is None else expr.when(column == key, value)
    return expr.cast(IntegerType())


def snake(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def build_applications(snapshot_date_str, silver_directory, spark):
    """Loans applied for in this month = LMS rows at month-on-book 0."""
    lms = read_silver(silver_directory, "lms", snapshot_date_str, spark)
    return lms.filter(col("mob") == 0).select(
        "loan_id", "Customer_ID", "snapshot_date",
        (col("loan_amt") / col("tenure")).alias("installment_amt"))


def build_attribute_features(snapshot_date_str, silver_directory, spark):
    df = read_silver(silver_directory, "attributes", snapshot_date_str, spark)
    features = [col("Age").alias("fe_attr_age")]
    # one-hot encode occupation (missing occupation -> all zeros)
    for occ in OCCUPATIONS:
        features.append(F.when(col("Occupation") == occ, 1).otherwise(0).alias(f"fe_attr_occ_{snake(occ)}"))
    return df.select("Customer_ID", *features)


def build_financial_features(snapshot_date_str, silver_directory, spark):
    df = read_silver(silver_directory, "financials", snapshot_date_str, spark)

    # raw numeric features -> renamed with the fe_fin_ prefix
    numeric_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
                    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
                    "Credit_Utilization_Ratio", "Credit_History_Months", "Total_EMI_per_month",
                    "Amount_invested_monthly", "Monthly_Balance"]
    features = [col(c).alias(f"fe_fin_{snake(c)}") for c in numeric_cols]

    # categorical -> numeric (fixed mappings)
    features += [
        map_values(col("Credit_Mix"), CREDIT_MIX_ORDER).alias("fe_fin_credit_mix"),
        map_values(col("Payment_of_Min_Amount"), MIN_PAYMENT_MAP).alias("fe_fin_pays_min_amount_only"),
        # "High_spent_Small_value_payments" -> spend level (High) + payment value (Small)
        map_values(F.split(col("Payment_Behaviour"), "_").getItem(0), SPEND_LEVEL_MAP).alias("fe_fin_spend_level"),
        map_values(F.split(col("Payment_Behaviour"), "_").getItem(2), PAYMENT_VALUE_MAP).alias("fe_fin_payment_value"),
    ]

    # loan type list -> one flag per type
    for loan_type in LOAN_TYPES:
        features.append(
            F.when(F.array_contains(col("Type_of_Loan"), loan_type), 1).otherwise(0)
             .alias(f"fe_fin_has_{snake(loan_type)}"))

    # engineered ratios - affordability & indebtedness
    features += [
        safe_divide(col("Outstanding_Debt"), col("Annual_Income")).alias("fe_fin_debt_to_income"),
        safe_divide(col("Total_EMI_per_month"), col("Monthly_Inhand_Salary")).alias("fe_fin_emi_to_salary"),
        safe_divide(col("Amount_invested_monthly"), col("Monthly_Inhand_Salary")).alias("fe_fin_invest_to_salary"),
        safe_divide(col("Monthly_Balance"), col("Monthly_Inhand_Salary")).alias("fe_fin_balance_to_salary"),
    ]
    return df.select("Customer_ID", "Monthly_Inhand_Salary", *features)


def build_clickstream_features(snapshot_date_str, silver_directory, spark, window_months=3):
    """
    Point-in-time clickstream features for an application in month D:
      * fe_click_i          - value in month D itself
      * fe_click_i_avg3m    - average over months D-2 .. D (only months that exist)
    Months AFTER D are never read -> no temporal leakage.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    window_dates = [(snapshot_date - relativedelta(months=i)).strftime("%Y-%m-%d") for i in range(window_months)]

    frames = [read_silver(silver_directory, "clickstream", d, spark) for d in window_dates]
    frames = [f for f in frames if f is not None]
    history = frames[0]
    for f in frames[1:]:
        history = history.unionByName(f)

    click_cols = [f"fe_{i}" for i in range(1, 21)]

    current = history.filter(col("snapshot_date") == snapshot_date_str) \
                     .select("Customer_ID", *[col(c).alias(f"fe_click_{c[3:]}") for c in click_cols])

    rolling = history.groupBy("Customer_ID").agg(
        *[F.avg(c).alias(f"fe_click_{c[3:]}_avg{window_months}m") for c in click_cols],
        F.count("*").alias("fe_click_months_available"))

    return current.join(rolling, on="Customer_ID", how="left")


def process_gold_feature_store(snapshot_date_str, silver_directory, gold_directory, spark):
    """One row per loan applied for in this month, with everything known at application time."""
    apps = build_applications(snapshot_date_str, silver_directory, spark)
    attrs = build_attribute_features(snapshot_date_str, silver_directory, spark)
    fins = build_financial_features(snapshot_date_str, silver_directory, spark)
    clicks = build_clickstream_features(snapshot_date_str, silver_directory, spark)

    df = apps.join(attrs, on="Customer_ID", how="left") \
             .join(fins, on="Customer_ID", how="left") \
             .join(clicks, on="Customer_ID", how="left")

    # new loan's monthly instalment relative to salary (affordability of THIS loan)
    df = df.withColumn("fe_fin_installment_to_salary", safe_divide(col("installment_amt"), col("Monthly_Inhand_Salary")))
    # customers who applied after clickstream collection stopped have no clickstream at all
    df = df.withColumn("fe_click_available", F.when(col("fe_click_1").isNotNull(), 1).otherwise(0))
    df = df.withColumn("fe_click_months_available", F.coalesce(col("fe_click_months_available"), F.lit(0)))

    df = df.drop("installment_amt", "Monthly_Inhand_Salary") \
           .withColumnRenamed("snapshot_date", "feature_snapshot_date")

    # key columns first, then features
    key_cols = ["loan_id", "Customer_ID", "feature_snapshot_date"]
    feature_cols = sorted([c for c in df.columns if c.startswith("fe_")])
    df = df.select(*key_cols, *feature_cols)

    return write_gold(df, gold_directory, "feature_store", snapshot_date_str)
