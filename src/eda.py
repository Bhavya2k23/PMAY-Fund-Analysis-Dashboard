
"""
PMAY Fund Analysis
Phase 5 - Exploratory Data Analysis (EDA)
Compatible with current master_dataset.csv
"""

import os
import sys
import pandas as pd

MASTER_FILE = "data/cleaned/master_dataset.csv"
REPORT_FILE = "reports/eda_report.txt"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

def section(title):
    log("\n" + "="*70)
    log(title)
    log("="*70)

def load_master():
    if not os.path.exists(MASTER_FILE):
        print(f"ERROR: {MASTER_FILE} not found.")
        sys.exit(1)
    return pd.read_csv(MASTER_FILE)

def top_bottom_states(df,col,title,n=5):
    if col not in df.columns:
        log(f"Skipping {title} (missing column: {col})")
        return
    tmp=df[["state_name",col]].copy()
    tmp[col]=pd.to_numeric(tmp[col],errors="coerce")
    tmp=tmp.dropna().sort_values(col,ascending=False)
    if tmp.empty:
        log(f"No data for {title}")
        return
    log(f"\nTop {n} {title}")
    log(tmp.head(n).to_string(index=False))
    log(f"\nBottom {n} {title}")
    log(tmp.tail(n).to_string(index=False))

def basic(df):
    section("1. BASIC OVERVIEW")
    log(f"Rows : {len(df)}")
    log(f"Columns : {len(df.columns)}")
    log(df.columns.tolist())
    miss=df.isna().sum()
    miss=miss[miss>0]
    log("\nMissing Values:")
    log("None" if miss.empty else miss.to_string())

def stats(df):
    section("2. DESCRIPTIVE STATISTICS")
    num=df.select_dtypes(include="number")
    if not num.empty:
        log(num.describe().round(2).to_string())

def funds(df):
    section("3. FUND UTILIZATION")
    req=["funds_released_crore","funds_utilized_crore","fund_utilization_percent"]
    if not all(c in df.columns for c in req):
        log("Required columns missing.")
        return
    rel=df["funds_released_crore"].sum()
    uti=df["funds_utilized_crore"].sum()
    pct=(uti/rel*100) if rel else 0
    log(f"Funds Released : {rel:,.2f}")
    log(f"Funds Utilized : {uti:,.2f}")
    log(f"Overall Utilization : {pct:.2f}%")
    top_bottom_states(df,"fund_utilization_percent","Fund Utilization %")

def beneficiaries(df):
    section("4. BENEFICIARY DISTRIBUTION")
    if "total" not in df.columns:
        log("total column missing.")
        return
    tmp=df.copy()
    tmp["total"]=pd.to_numeric(tmp["total"],errors="coerce").fillna(0)
    cats=[c for c in ["sc","st","others","minority"] if c in tmp.columns]
    total=tmp["total"].sum()
    for c in cats:
        tmp[c]=pd.to_numeric(tmp[c],errors="coerce").fillna(0)
        tmp[c+"_share_pct"]=((tmp[c]/tmp["total"].replace(0,pd.NA))*100).fillna(0).round(2)
        pct=(tmp[c].sum()/total*100) if total else 0
        log(f"{c.upper()} : {tmp[c].sum():,.0f} ({pct:.2f}%)")
    top_bottom_states(tmp,"total","Total Beneficiaries")
    if "sc_share_pct" in tmp.columns:
        top_bottom_states(tmp,"sc_share_pct","SC Share %")
    if "st_share_pct" in tmp.columns:
        top_bottom_states(tmp,"st_share_pct","ST Share %")

def corr(df):
    section("5. CORRELATION")
    num=df.select_dtypes(include="number")
    if num.shape[1]<2:
        log("Not enough numeric columns.")
        return
    c=num.corr().round(2)
    log(c.to_string())

def insights(df):
    section("6. KEY INSIGHTS")
    if "fund_utilization_percent" in df.columns:
        b=df.loc[df["fund_utilization_percent"].idxmax()]
        w=df.loc[df["fund_utilization_percent"].idxmin()]
        log(f"Highest utilization : {b['state_name']} ({b['fund_utilization_percent']:.2f}%)")
        log(f"Lowest utilization : {w['state_name']} ({w['fund_utilization_percent']:.2f}%)")
    if "total" in df.columns:
        x=df.loc[df["total"].idxmax()]
        log(f"Highest beneficiaries : {x['state_name']} ({x['total']:,.0f})")

def main():
    log("="*70)
    log("PMAY FUND ANALYSIS - EXPLORATORY DATA ANALYSIS")
    log("="*70)
    df=load_master()
    basic(df)
    stats(df)
    funds(df)
    beneficiaries(df)
    corr(df)
    insights(df)
    os.makedirs("reports",exist_ok=True)
    with open(REPORT_FILE,"w",encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nReport saved to: {REPORT_FILE}")

if __name__=="__main__":
    main()
