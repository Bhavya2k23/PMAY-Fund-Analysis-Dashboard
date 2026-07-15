import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PMAY FUND ANALYSIS
# Phase 6 - Data Visualization
# ==========================================================

MASTER_FILE = "data/cleaned/master_dataset.csv"
OUTPUT_DIR = "images"

# ----------------------------------------------------------
# Check Dataset
# ----------------------------------------------------------

if not os.path.exists(MASTER_FILE):
    print("Master dataset not found!")
    print("Run merge_data.py first.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# Visualization Settings
# ----------------------------------------------------------

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv(MASTER_FILE)

print("=" * 70)
print("PMAY DATA VISUALIZATION")
print("=" * 70)
print(f"Dataset Shape : {df.shape}")

# ----------------------------------------------------------
# Helper Function
# ----------------------------------------------------------

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ==========================================================
# 1. Top 10 States by Beneficiaries
# ==========================================================

if "total" in df.columns:

    top = df.sort_values(
        "total",
        ascending=False
    ).head(10)

    plt.figure()

    sns.barplot(
        data=top,
        x="total",
        y="state_name",
        hue="state_name",
        palette="viridis",
        legend=False
    )

    plt.title("Top 10 States by Total Beneficiaries")
    plt.xlabel("Beneficiaries")
    plt.ylabel("State")

    save_plot("01_top_beneficiaries.png")

# ==========================================================
# 2. Funds Released
# ==========================================================

if "funds_released_crore" in df.columns:

    top = df.sort_values(
        "funds_released_crore",
        ascending=False
    ).head(10)

    plt.figure()

    sns.barplot(
        data=top,
        x="funds_released_crore",
        y="state_name",
        hue="state_name",
        palette="Blues_r",
        legend=False
    )

    plt.title("Top 10 States by Funds Released")
    plt.xlabel("Funds Released (Crore)")
    plt.ylabel("State")

    save_plot("02_funds_released.png")

# ==========================================================
# 3. Funds Utilized
# ==========================================================

if "funds_utilized_crore" in df.columns:

    top = df.sort_values(
        "funds_utilized_crore",
        ascending=False
    ).head(10)

    plt.figure()

    sns.barplot(
        data=top,
        x="funds_utilized_crore",
        y="state_name",
        hue="state_name",
        palette="Greens_r",
        legend=False
    )

    plt.title("Top 10 States by Funds Utilized")
    plt.xlabel("Funds Utilized (Crore)")
    plt.ylabel("State")

    save_plot("03_funds_utilized.png")

# ==========================================================
# 4. Fund Utilization %
# ==========================================================

if "fund_utilization_percent" in df.columns:

    top = df.sort_values(
        "fund_utilization_percent",
        ascending=False
    ).head(10)

    plt.figure()

    sns.barplot(
        data=top,
        x="fund_utilization_percent",
        y="state_name",
        hue="state_name",
        palette="magma",
        legend=False
    )

    plt.title("Top 10 States by Fund Utilization")
    plt.xlabel("Utilization (%)")
    plt.ylabel("State")

    save_plot("04_utilization_percent.png")

# ==========================================================
# 5. Social Category Distribution
# ==========================================================

category_cols = [
    col for col in ["sc", "st", "others"]
    if col in df.columns
]

if category_cols:

    totals = df[category_cols].sum()

    plt.figure(figsize=(8, 8))

    plt.pie(
        totals,
        labels=[c.upper() for c in category_cols],
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.03] * len(category_cols)
    )

    plt.title("Beneficiary Social Category Distribution")

    save_plot("05_social_distribution.png")

# ==========================================================
# 6. Correlation Heatmap
# ==========================================================

numeric = df.select_dtypes(include="number")

if not numeric.empty:

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=.5
    )

    plt.title("Correlation Heatmap")

    save_plot("06_heatmap.png")

# ==========================================================
# 7. Boxplot
# ==========================================================

if "fund_utilization_percent" in df.columns:

    plt.figure()

    sns.boxplot(
        x=df["fund_utilization_percent"],
        color="orange"
    )

    plt.title("Distribution of Fund Utilization")

    save_plot("07_boxplot.png")

# ==========================================================
# 8. Histogram
# ==========================================================

if "fund_utilization_percent" in df.columns:

    plt.figure()

    sns.histplot(
        df["fund_utilization_percent"],
        bins=12,
        kde=True,
        color="steelblue"
    )

    plt.title("Fund Utilization Distribution")
    plt.xlabel("Fund Utilization (%)")

    save_plot("08_histogram.png")

# ==========================================================
# 9. Scatter Plot
# ==========================================================

if (
    "funds_released_crore" in df.columns and
    "funds_utilized_crore" in df.columns
):

    plt.figure()

    sns.scatterplot(
        data=df,
        x="funds_released_crore",
        y="funds_utilized_crore",
        s=120,
        alpha=0.8
    )

    plt.title("Funds Released vs Funds Utilized")
    plt.xlabel("Funds Released (Crore)")
    plt.ylabel("Funds Utilized (Crore)")

    save_plot("09_scatter.png")

# ==========================================================
# 10. Beneficiaries vs Funds Released
# ==========================================================

if (
    "total" in df.columns and
    "funds_released_crore" in df.columns
):

    plt.figure()

    sns.scatterplot(
        data=df,
        x="total",
        y="funds_released_crore",
        s=120,
        alpha=0.8
    )

    plt.title("Beneficiaries vs Funds Released")
    plt.xlabel("Total Beneficiaries")
    plt.ylabel("Funds Released (Crore)")

    save_plot("10_beneficiaries_vs_funds.png")

# ----------------------------------------------------------
# Finish
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved in folder : {OUTPUT_DIR}")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(file)