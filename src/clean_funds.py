import pandas as pd
import numpy as np
import os
import sys
from utils import (
    safe_read_csv,
    standardize_state_names,
    drop_summary_rows,
    log_missing_before_after,
    save_cleaning_log,
)

# ==========================================================
# PMAY Fund Analysis
# Dataset Cleaning - Fund Utilization Dataset (Enhanced)
# ==========================================================

INPUT_FILE = "data/raw/RS_Session_267_AU_3981_1.csv"
OUTPUT_FILE = "data/cleaned/funds_cleaned.csv"
LOG_FILE = "reports/funds_cleaning_log.txt"

log_lines = []


def main():
    print("=" * 70)
    print("CLEANING FUND UTILIZATION DATASET")
    print("=" * 70)

    try:
        df = safe_read_csv(INPUT_FILE)
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"\nOriginal Shape: {df.shape}")
    log_lines.append(f"Original Shape: {df.shape}")

    # --------------------------------------------------
    # Strip column names FIRST, then rename — fixes the bug
    # where trailing spaces in headers (e.g. "State/UT ")
    # would make the rename dict silently not match.
    # --------------------------------------------------
    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Sl. No.": "serial_no",
        "State/UT": "state_name",
        "Total Central Share Released": "funds_released_crore",
        "Total Utilization": "funds_utilized_crore",
    }, inplace=True)

    if "state_name" not in df.columns:
        print(f"❌ Expected 'state_name' column not found after rename. Actual columns: {df.columns.tolist()}")
        sys.exit(1)

    df["state_name"] = df["state_name"].astype(str).str.strip()

    # --------------------------------------------------
    # Drop "Total" / "All India" summary rows BEFORE any
    # numeric aggregation — otherwise every KPI downstream
    # (avg utilization, top/bottom states) will be wrong.
    # --------------------------------------------------
    df = drop_summary_rows(df, "state_name")

    # --------------------------------------------------
    # Standardize state names so this merges cleanly with
    # the OGD beneficiary dataset later.
    # --------------------------------------------------
    df = standardize_state_names(df, "state_name")

    # --------------------------------------------------
    # Convert data types
    # --------------------------------------------------
    if "serial_no" in df.columns:
        df["serial_no"] = pd.to_numeric(df["serial_no"], errors="coerce").fillna(0).astype(int)

    for col in ["funds_released_crore", "funds_utilized_crore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    log_lines.append(f"Duplicate rows removed: {duplicates}")
    df.drop_duplicates(inplace=True)

    # --------------------------------------------------
    # Missing values — log before filling
    # --------------------------------------------------
    log_missing_before_after(df, "Before fill", log_lines)

    df["funds_released_crore"] = df["funds_released_crore"].fillna(0)
    df["funds_utilized_crore"] = df["funds_utilized_crore"].fillna(0)

    log_missing_before_after(df, "After fill", log_lines)

    # --------------------------------------------------
    # Build fund utilization KPI
    # --------------------------------------------------
    df["fund_utilization_percent"] = np.where(
        df["funds_released_crore"] == 0,
        0,
        (df["funds_utilized_crore"] / df["funds_released_crore"]) * 100,
    )
    df["fund_utilization_percent"] = df["fund_utilization_percent"].round(2)

    # --------------------------------------------------
    # Flag (don't silently hide) impossible values — utilization
    # over 100% usually means a data entry error upstream, worth
    # knowing about rather than quietly accepting.
    # --------------------------------------------------
    over_100 = df[df["fund_utilization_percent"] > 100]
    if not over_100.empty:
        msg = f"⚠️  {len(over_100)} state(s) show utilization > 100%: {over_100['state_name'].tolist()}"
        print(msg)
        log_lines.append(msg)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    save_cleaning_log(log_lines, LOG_FILE)

    print("\nCleaning Completed Successfully!")
    print(f"Cleaned Shape: {df.shape}")
    print(f"Dataset Saved: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()