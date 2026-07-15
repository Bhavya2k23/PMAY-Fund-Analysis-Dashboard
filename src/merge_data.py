import pandas as pd
import os
import sys
from utils import save_cleaning_log

# ==========================================================
# PMAY Fund Analysis
# Phase 4 : Merge Cleaned Datasets
# ==========================================================

OGD_FILE = "data/cleaned/ogd_cleaned.csv"
FUNDS_FILE = "data/cleaned/funds_cleaned.csv"

OUTPUT_FILE = "data/cleaned/master_dataset.csv"
LOG_FILE = "reports/merge_log.txt"

log_lines = []


# ==========================================================
# Load Files
# ==========================================================

def load_cleaned_files():

    for file in [OGD_FILE, FUNDS_FILE]:

        if not os.path.exists(file):
            print(f"❌ Missing file : {file}")
            sys.exit()

    ogd = pd.read_csv(OGD_FILE)

    funds = pd.read_csv(FUNDS_FILE)

    return ogd, funds


# ==========================================================
# Standardize State Names
# ==========================================================

def standardize_state_names(df):

    df["state_name"] = (
        df["state_name"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    replacements = {

        "Orissa": "Odisha",

        "Uttaranchal": "Uttarakhand",

        "Nct Of Delhi": "Delhi",

        "Andaman & Nicobar Islands": "Andaman And Nicobar",

        "Jammu & Kashmir": "Jammu And Kashmir",

        "Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",

        "Dadra And Nagar Haveli": "Dadra And Nagar Haveli And Daman And Diu"

    }

    df["state_name"] = df["state_name"].replace(replacements)

    return df


# ==========================================================
# Aggregate OGD
# ==========================================================

def aggregate_ogd(ogd):

    print("\nAggregating OGD dataset...")

    numeric_columns = [

        "sc",

        "st",

        "others",

        "total"

    ]

    agg = (

        ogd

        .groupby("state_name", as_index=False)

        .agg({

            "sc": "sum",

            "st": "sum",

            "others": "sum",

            "total": "sum"

        })

    )

    block_count = (

        ogd

        .groupby("state_name")

        .size()

        .reset_index(name="panchayat_block_count")

    )

    agg = agg.merge(block_count, on="state_name")

    print(f"State Level Shape : {agg.shape}")

    return agg


# ==========================================================
# Merge
# ==========================================================

def merge_data(ogd, funds):

    print("\nMerging datasets...")

    master = pd.merge(

        ogd,

        funds,

        on="state_name",

        how="outer",

        indicator=True

    )

    print(master["_merge"].value_counts())

    unmatched = master[master["_merge"] != "both"]

    if len(unmatched):

        print("\n⚠ Unmatched States")

        print(unmatched[["state_name", "_merge"]])

    master.drop(columns="_merge", inplace=True)

    return master


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)

    print("BUILDING MASTER DATASET")

    print("=" * 70)

    ogd, funds = load_cleaned_files()

    print("\nOriginal Shapes")

    print("OGD   :", ogd.shape)

    print("Funds :", funds.shape)

    # -----------------------------
    # Standardize Names
    # -----------------------------

    ogd = standardize_state_names(ogd)

    funds = standardize_state_names(funds)

    # -----------------------------
    # Aggregate
    # -----------------------------

    ogd = aggregate_ogd(ogd)

    # -----------------------------
    # Merge
    # -----------------------------

    master = merge_data(ogd, funds)

    # -----------------------------
    # Fill Missing Fund Values
    # -----------------------------

    fund_columns = [

        "funds_released_crore",

        "funds_utilized_crore",

        "fund_utilization_percent"

    ]

    for col in fund_columns:

        if col in master.columns:

            master[col] = master[col].fillna(0)

    # -----------------------------
    # Sort
    # -----------------------------

    master = master.sort_values("state_name")

    master.reset_index(drop=True, inplace=True)

    # -----------------------------
    # Save
    # -----------------------------

    os.makedirs("data/cleaned", exist_ok=True)

    master.to_csv(OUTPUT_FILE, index=False)

    log_lines.append(f"Final Shape : {master.shape}")

    save_cleaning_log(log_lines, LOG_FILE)

    print("\n")

    print("=" * 70)

    print("MASTER DATASET CREATED SUCCESSFULLY")

    print("=" * 70)

    print(f"Saved : {OUTPUT_FILE}")

    print(f"Shape : {master.shape}")

    print("=" * 70)


if __name__ == "__main__":

    main()