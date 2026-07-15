import pandas as pd
import os
import sys
from utils import safe_read_csv, standardize_state_names, log_missing_before_after, save_cleaning_log

# ==========================================================
# PMAY Fund Analysis
# Dataset Cleaning - OGD Beneficiary Dataset (Enhanced)
# ==========================================================

INPUT_FILE = "data/raw/OGDSECCAwaasPlusData_18092023.csv"
OUTPUT_FILE = "data/cleaned/ogd_cleaned.csv"
LOG_FILE = "reports/ogd_cleaning_log.txt"

# Columns known to be numeric identifiers/counts in this dataset.
# Kept as a constant at the top so it's easy to update if column
# names change.
CANDIDATE_NUMERIC_COLS = [
    "state_code", "district_code", "block_code", "panchayat_code",
    "others", "sc", "st", "total",
]

log_lines = []


def main():
    print("=" * 70)
    print("CLEANING OGD BENEFICIARY DATASET")
    print("=" * 70)

    try:
        df = safe_read_csv(INPUT_FILE)
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"\nOriginal Shape: {df.shape}")
    log_lines.append(f"Original Shape: {df.shape}")

    # --------------------------------------------------
    # Clean column names FIRST (lowercase + strip), before
    # trying to rename anything — this is the fix for the
    # rename-not-matching bug.
    # --------------------------------------------------
    df.columns = df.columns.str.strip().str.lower()

    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        print(f"Removed unnamed columns: {unnamed_cols}")
        log_lines.append(f"Removed unnamed columns: {unnamed_cols}")

    # Rename using lowercase keys since columns are now lowercase
    df.rename(columns={
        "district_code": "district_code",
        "block_code": "block_code",
        "panchayat_code": "panchayat_code",
        "minority": "minority",
        "others": "others",
        "sc": "sc",
        "st": "st",
        "total": "total",
    }, inplace=True)

    # --------------------------------------------------
    # Strip whitespace from all text columns
    # --------------------------------------------------
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    # --------------------------------------------------
    # Standardize state names BEFORE anything else touches
    # them, so later merges with other datasets work.
    # --------------------------------------------------
    state_col_candidates = [c for c in df.columns if "state" in c and "code" not in c]
    if state_col_candidates:
        state_col = state_col_candidates[0]
        df = standardize_state_names(df, state_col)
        print(f"Standardized state names in column: '{state_col}'")
        log_lines.append(f"Standardized state names in column: '{state_col}'")
    else:
        print("⚠️  No state name column detected (only state_code found) — skipping name standardization")

    # --------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    log_lines.append(f"Duplicate rows removed: {duplicates}")
    df.drop_duplicates(inplace=True)

    # --------------------------------------------------
    # Log missing values BEFORE filling, so you have a
    # record of what was actually missing vs. assumed.
    # --------------------------------------------------
    log_missing_before_after(df, "Before fill", log_lines)

    # --------------------------------------------------
    # Convert candidate numeric columns FIRST, before any
    # text-fill logic touches them (fixes the fill-then-convert
    # contradiction in the original script).
    # --------------------------------------------------
    print("Converting numeric columns...")
    for col in CANDIDATE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0).astype("Int64")

    # Minority column: only convert to numeric if it's genuinely
    # numeric everywhere; otherwise keep as text and say so.
    if "minority" in df.columns:
        numeric_attempt = pd.to_numeric(df["minority"], errors="coerce")
        if numeric_attempt.notnull().all():
            df["minority"] = numeric_attempt.astype("Int64")
        else:
            non_numeric_sample = df.loc[numeric_attempt.isnull(), "minority"].unique()[:5]
            print(f"⚠️  'minority' column has non-numeric values, e.g. {non_numeric_sample}. Keeping as text.")
            log_lines.append(f"'minority' kept as text — non-numeric values found: {non_numeric_sample}")

    # --------------------------------------------------
    # Now fill remaining missing values in whatever text
    # columns are left (after numeric conversion above)
    # --------------------------------------------------
    remaining_text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[remaining_text_cols] = df[remaining_text_cols].fillna("Unknown")

    numeric_cols_now = df.select_dtypes(include="number").columns
    df[numeric_cols_now] = df[numeric_cols_now].fillna(0)

    log_missing_before_after(df, "After fill", log_lines)

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    print("\nFinal Shape:", df.shape)
    print("\nFinal Data Types:\n", df.dtypes)
    log_lines.append(f"\nFinal Shape: {df.shape}")
    log_lines.append(f"\nFinal Data Types:\n{df.dtypes.to_string()}")

    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    save_cleaning_log(log_lines, LOG_FILE)

    print("\n" + "=" * 70)
    print("Cleaning Completed Successfully")
    print(f"Cleaned Dataset Saved To: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()