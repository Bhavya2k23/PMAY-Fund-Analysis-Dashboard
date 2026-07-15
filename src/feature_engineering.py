import pandas as pd
import numpy as np
import os

# ==========================================================
# PMAY FUND ANALYSIS
# Phase 7 - Feature Engineering
# ==========================================================

INPUT_FILE = "data/cleaned/master_dataset.csv"
OUTPUT_FILE = "data/cleaned/final_dataset.csv"
LOG_FILE = "reports/feature_engineering_log.txt"

print("=" * 70)
print("PHASE 7 : FEATURE ENGINEERING")
print("=" * 70)

# ----------------------------------------------------------
# Check File
# ----------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("Master dataset not found.")
    print("Run merge_data.py first.")
    exit()

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset Loaded Successfully")
print(f"Shape : {df.shape}")

log = []

# ==========================================================
# 1. Fund Gap
# ==========================================================

if (
    "funds_released_crore" in df.columns and
    "funds_utilized_crore" in df.columns
):

    df["fund_gap_crore"] = (
        df["funds_released_crore"] -
        df["funds_utilized_crore"]
    ).round(2)

    log.append("Fund Gap Created")

# ==========================================================
# 2. Utilization Category
# ==========================================================

if "fund_utilization_percent" in df.columns:

    conditions = [

        df["fund_utilization_percent"] >= 100,

        (df["fund_utilization_percent"] >= 80) &
        (df["fund_utilization_percent"] < 100),

        (df["fund_utilization_percent"] >= 60) &
        (df["fund_utilization_percent"] < 80),

        df["fund_utilization_percent"] < 60

    ]

    choices = [

        "Excellent",

        "Good",

        "Average",

        "Poor"

    ]

    df["utilization_category"] = np.select(
        conditions,
        choices,
        default="Unknown"
    )

    log.append("Utilization Category Created")

# ==========================================================
# 3. Beneficiary Rank
# ==========================================================

if "total" in df.columns:

    df["beneficiary_rank"] = (
        df["total"]
        .rank(
            ascending=False,
            method="dense"
        )
        .astype(int)
    )

    log.append("Beneficiary Rank Created")

# ==========================================================
# 4. Funds Released Rank
# ==========================================================

if "funds_released_crore" in df.columns:

    df["fund_release_rank"] = (

        df["funds_released_crore"]

        .rank(

            ascending=False,

            method="dense"

        )

        .astype(int)

    )

    log.append("Fund Release Rank Created")

# ==========================================================
# 5. Funds Utilized Rank
# ==========================================================

if "funds_utilized_crore" in df.columns:

    df["fund_utilized_rank"] = (

        df["funds_utilized_crore"]

        .rank(

            ascending=False,

            method="dense"

        )

        .astype(int)

    )

    log.append("Fund Utilized Rank Created")

# ==========================================================
# 6. Beneficiary Density Index
# ==========================================================

if (
    "total" in df.columns and
    "panchayat_block_count" in df.columns
):

    df["beneficiary_density"] = (

        df["total"] /

        df["panchayat_block_count"]

    ).round(2)

    log.append("Beneficiary Density Created")

# ==========================================================
# 7. High Fund Release Flag
# ==========================================================

if "funds_released_crore" in df.columns:

    median = df["funds_released_crore"].median()

    df["high_fund_release"] = np.where(

        df["funds_released_crore"] >= median,

        "Yes",

        "No"

    )

    log.append("High Fund Release Flag Created")

# ==========================================================
# 8. High Beneficiary Flag
# ==========================================================

if "total" in df.columns:

    median = df["total"].median()

    df["high_beneficiary"] = np.where(

        df["total"] >= median,

        "Yes",

        "No"

    )

    log.append("High Beneficiary Flag Created")

# ==========================================================
# 9. Overall Performance
# ==========================================================

if (
    "utilization_category" in df.columns and
    "high_fund_release" in df.columns
):

    performance = []

    for _, row in df.iterrows():

        if (
            row["utilization_category"] == "Excellent"
            and row["high_fund_release"] == "Yes"
        ):

            performance.append("High Performer")

        elif (
            row["utilization_category"] == "Poor"
            and row["high_fund_release"] == "Yes"
        ):

            performance.append("Needs Improvement")

        else:

            performance.append("Average")

    df["performance_category"] = performance

    log.append("Performance Category Created")

# ==========================================================
# Save Dataset
# ==========================================================

os.makedirs("data/cleaned", exist_ok=True)

df.to_csv(

    OUTPUT_FILE,

    index=False

)

# ==========================================================
# Save Log
# ==========================================================

os.makedirs("reports", exist_ok=True)

with open(

    LOG_FILE,

    "w",

    encoding="utf-8"

) as f:

    f.write("FEATURE ENGINEERING LOG\n")

    f.write("=" * 50 + "\n\n")

    for item in log:

        f.write(item + "\n")

# ==========================================================
# Output
# ==========================================================

print("\nFeature Engineering Completed Successfully!")

print(f"\nNew Shape : {df.shape}")

print("\nNew Features Created:")

new_cols = [

    "fund_gap_crore",

    "utilization_category",

    "beneficiary_rank",

    "fund_release_rank",

    "fund_utilized_rank",

    "beneficiary_density",

    "high_fund_release",

    "high_beneficiary",

    "performance_category"

]

for col in new_cols:

    if col in df.columns:

        print(f"✔ {col}")

print(f"\nDataset Saved : {OUTPUT_FILE}")

print(f"Log Saved : {LOG_FILE}")

print("=" * 70)