import pandas as pd
import os

# ==========================================================
# Shared utilities for PMAY Fund Analysis cleaning scripts
# ==========================================================

# Master state/UT name map — add to this as you spot new
# inconsistencies while cleaning more files.
STATE_NAME_MAP = {
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Pondicherry": "Puducherry",
    "NCT of Delhi": "Delhi",
    "Delhi (NCT)": "Delhi",
    "Andaman & Nicobar": "Andaman and Nicobar Islands",
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "J&K": "Jammu and Kashmir",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Dadra And Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
}

# Rows to drop before analysis — these are summary/total rows that
# commonly appear at the bottom of government report CSVs and will
# silently distort every state-level KPI if left in.
SUMMARY_ROW_KEYWORDS = ["total", "all india", "grand total", "all-india"]


def safe_read_csv(path):
    """Read a CSV trying multiple encodings, since government exports
    are frequently NOT plain UTF-8."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not read {path} with any known encoding")


def standardize_state_names(df, state_col):
    """Strip, title-case, and map known spelling variants to one
    canonical name so files can be merged/joined reliably later."""
    df[state_col] = df[state_col].astype(str).str.strip().str.title()
    df[state_col] = df[state_col].replace(STATE_NAME_MAP)
    return df


def drop_summary_rows(df, state_col):
    """Drop rows like 'Total' / 'All India' that are report footers,
    not actual states — these will wreck state-level aggregations
    if left in the dataset."""
    mask = df[state_col].astype(str).str.lower().str.strip().isin(SUMMARY_ROW_KEYWORDS)
    dropped = df[mask]
    if len(dropped) > 0:
        print(f"⚠️  Dropping {len(dropped)} summary/total row(s): {dropped[state_col].tolist()}")
    return df[~mask].copy()


def log_missing_before_after(df, stage_name, out_lines):
    """Record missing-value counts at a given cleaning stage into a
    log list so you can save a before/after report."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    out_lines.append(f"\n[{stage_name}] Missing values:")
    out_lines.append(missing.to_string() if not missing.empty else "  None")
    return out_lines


def save_cleaning_log(log_lines, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n📄 Cleaning log saved to: {output_path}")