import pandas as pd
import os
import sys
from datetime import datetime

# ==========================================================
# PMAY Fund Analysis - Dataset Inspection Script (Enhanced)
# ==========================================================

RAW_FOLDER = "data/raw"
REPORT_FOLDER = "reports"
REPORT_FILE = os.path.join(REPORT_FOLDER, "inspection_report.txt")

# Pandas display settings so large tables don't get truncated in terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.max_rows", 50)


def get_supported_files(folder):
    """Return only csv/xlsx/xls files, ignoring hidden/system files."""
    supported_ext = (".csv", ".xlsx", ".xls")
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(supported_ext) and not f.startswith(("~$", "."))
    ])


def read_csv_safely(path):
    """
    Try multiple encodings since government data files are often
    saved in encodings other than UTF-8 (common cause of UnicodeDecodeError).
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except pd.errors.EmptyDataError as e:
            raise e  # no point retrying encodings on an empty file
    raise last_error


def read_excel_safely(path):
    """
    Read Excel file. If it has multiple sheets, read all of them
    and return a dict of {sheet_name: dataframe} instead of silently
    reading only the first sheet (a common hidden bug).
    """
    sheets = pd.read_excel(path, sheet_name=None)  # dict of all sheets
    return sheets


def inspect_dataframe(df, label, out):
    """Print + write a full inspection summary for a single dataframe."""
    if df.empty:
        msg = f"⚠️  '{label}' is empty (0 rows). Skipping detailed inspection.\n"
        print(msg)
        out.write(msg)
        return

    lines = []
    lines.append(f"\n{'-'*70}\nSHEET/FILE: {label}\n{'-'*70}")
    lines.append(f"\n✅ Shape : {df.shape[0]} Rows x {df.shape[1]} Columns")

    lines.append("\n📌 Columns:")
    lines.append(str(df.columns.tolist()))

    lines.append("\n📌 Data Types:")
    lines.append(df.dtypes.to_string())

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_summary = missing_summary[missing_summary["missing_count"] > 0]
    lines.append("\n📌 Missing Values (columns with at least 1 missing value):")
    lines.append(missing_summary.to_string() if not missing_summary.empty else "None found ✅")

    lines.append(f"\n📌 Duplicate Rows: {df.duplicated().sum()}")

    lines.append(f"\n📌 Memory Usage: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")

    lines.append("\n📌 First 5 Rows:")
    lines.append(df.head().to_string())

    # Flag likely ID/text columns with inconsistent values (e.g. state name spelling issues)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        lines.append("\n📌 Unique value counts for text columns (checking for inconsistent naming):")
        for col in obj_cols:
            n_unique = df[col].nunique(dropna=True)
            lines.append(f"   - {col}: {n_unique} unique values")
            # If a column looks like it could be a state/region column with a
            # suspiciously high number of near-duplicate values, show them
            if n_unique <= 60:  # states/UTs in India are ~36, so this catches likely candidates
                sample_vals = sorted(df[col].dropna().unique().tolist())
                lines.append(f"     values: {sample_vals}")

    lines.append("\n📌 Summary Statistics (numeric columns):")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        lines.append(numeric_df.describe().to_string())
    else:
        lines.append("No numeric columns found.")

    text = "\n".join(lines) + "\n"
    print(text)
    out.write(text)


def main():
    if not os.path.exists(RAW_FOLDER):
        print(f"❌ Folder not found: {RAW_FOLDER}")
        print(f"   Create it with: mkdir -p {RAW_FOLDER}  (or put your files there)")
        sys.exit(1)

    files = get_supported_files(RAW_FOLDER)
    if not files:
        print(f"❌ No CSV/Excel files found in {RAW_FOLDER}")
        sys.exit(1)

    os.makedirs(REPORT_FOLDER, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as out:
        header = (
            "=" * 70 + "\n"
            "PMAY DATASET INSPECTION\n"
            f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 70 + "\n"
        )
        print(header)
        out.write(header)

        for file in files:
            path = os.path.join(RAW_FOLDER, file)
            file_header = f"\n{'='*70}\n📂 FILE: {file}\n{'='*70}"
            print(file_header)
            out.write(file_header + "\n")

            try:
                if file.lower().endswith(".csv"):
                    df = read_csv_safely(path)
                    inspect_dataframe(df, file, out)
                else:
                    sheets = read_excel_safely(path)
                    if len(sheets) == 1:
                        only_df = list(sheets.values())[0]
                        inspect_dataframe(only_df, file, out)
                    else:
                        note = f"ℹ️  This file has {len(sheets)} sheets: {list(sheets.keys())}"
                        print(note)
                        out.write(note + "\n")
                        for sheet_name, sheet_df in sheets.items():
                            inspect_dataframe(sheet_df, f"{file} -> sheet '{sheet_name}'", out)

            except pd.errors.EmptyDataError:
                msg = f"❌ '{file}' is empty or has no columns to parse."
                print(msg)
                out.write(msg + "\n")
            except FileNotFoundError:
                msg = f"❌ '{file}' could not be found (may have been moved/deleted)."
                print(msg)
                out.write(msg + "\n")
            except PermissionError:
                msg = f"❌ Permission denied reading '{file}'. Close it if it's open in Excel."
                print(msg)
                out.write(msg + "\n")
            except Exception as e:
                msg = f"❌ Unexpected error reading '{file}': {type(e).__name__}: {e}"
                print(msg)
                out.write(msg + "\n")

        footer = "\n" + "=" * 70 + "\n✅ Dataset Inspection Completed Successfully\n" + "=" * 70
        print(footer)
        out.write(footer + "\n")

    print(f"\n📄 Full report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()