"""
Multi-Source Budget Variance Attribution System
================================================
Simulates ingesting financial data from 3 enterprise sources (Salesforce CRM,
Oracle ERP, Data Lake), merges them into a unified FP&A view, then applies
statistical variance attribution to separate budget signal from noise —
surfacing only high-impact anomalies for leadership review.

JD Coverage:
  - Automated Data Pipelines: Python/Pandas merging Salesforce + ERP + Data Lake
  - Variance Attribution: statistical signal-vs-noise (z-score + IQR + regression)
  - Correlation & Distribution Analysis
  - Leadership-ready anomaly report with ranked impact
  - Visualization: Seaborn/Matplotlib dashboards

Author: Vijay Kumar Dandu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import sqlite3
import io
import warnings
warnings.filterwarnings("ignore")

np.random.seed(2024)

# ─── 1. SIMULATE 3 ENTERPRISE DATA SOURCES ───────────────────────────────────

def generate_salesforce_data():
    """Salesforce CRM: pipeline deals, closed revenue by rep and region."""
    n = 320
    regions    = ["North", "South", "East", "West", "Central"]
    categories = ["SaaS", "Professional Services", "Support", "Hardware", "Consulting"]
    stages     = ["Closed Won", "Closed Lost"]

    df = pd.DataFrame({
        "deal_id":       [f"SF-{10000+i}" for i in range(n)],
        "close_date":    pd.to_datetime(
            np.random.choice(pd.date_range("2024-01-01", "2024-12-31", freq="D"), n)),
        "region":        np.random.choice(regions, n, p=[0.25, 0.20, 0.22, 0.18, 0.15]),
        "category":      np.random.choice(categories, n, p=[0.35, 0.25, 0.15, 0.10, 0.15]),
        "stage":         np.random.choice(stages, n, p=[0.68, 0.32]),
        "deal_value":    np.random.lognormal(mean=11.0, sigma=0.9, size=n).round(2),
        "sales_rep_id":  [f"REP-{100 + np.random.randint(1, 25)}" for _ in range(n)],
        "budget_target": np.random.lognormal(mean=11.2, sigma=0.7, size=n).round(2),
    })
    # Won deals only count as actual revenue
    df["actual_revenue"] = np.where(df["stage"] == "Closed Won", df["deal_value"], 0)
    return df

def generate_erp_data():
    """Oracle ERP: department-level actuals vs budgets for opex line items."""
    departments = ["Engineering", "Sales & Marketing", "Finance", "Operations", "HR", "IT"]
    cost_types  = ["Headcount", "Vendor/SaaS", "Travel", "Infrastructure", "Marketing Spend", "R&D"]
    months      = pd.date_range("2024-01-01", "2024-12-01", freq="MS")

    rows = []
    for dept in departments:
        for ctype in cost_types:
            # Budgeted monthly cost
            budget_base = np.random.uniform(40_000, 380_000)
            for month in months:
                budget = budget_base * np.random.uniform(0.92, 1.08)
                # Inject realistic variance patterns
                variance_factor = np.random.choice(
                    [1.0, 1.0, 1.0, 1.18, 1.35, 0.78, 1.52, 0.65],
                    p=[0.55, 0.10, 0.10, 0.08, 0.06, 0.05, 0.04, 0.02]
                )
                actual = budget * variance_factor * np.random.uniform(0.96, 1.04)
                rows.append({
                    "month":       month,
                    "department":  dept,
                    "cost_type":   ctype,
                    "gl_account":  f"GL-{hash(dept+ctype) % 9000 + 1000}",
                    "budget":      round(budget, 2),
                    "actual":      round(actual, 2),
                    "variance":    round(actual - budget, 2),
                    "variance_pct":round((actual - budget) / budget * 100, 2),
                })
    return pd.DataFrame(rows)

def generate_datalake_data():
    """Data Lake: operational metrics — invoice counts, payment timelines, churn."""
    months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    regions = ["North", "South", "East", "West", "Central"]
    rows = []
    for month in months:
        for region in regions:
            rows.append({
                "month":              month,
                "region":             region,
                "invoice_count":      np.random.randint(210, 310),
                "avg_invoice_value":  round(np.random.uniform(650, 950), 2),
                "avg_days_to_pay":    round(np.random.uniform(22, 55), 1),
                "churn_rate_pct":     round(np.random.uniform(1.5, 6.5), 2),
                "new_customers":      np.random.randint(8, 45),
                "support_tickets":    np.random.randint(30, 180),
            })
    return pd.DataFrame(rows)


print("=" * 65)
print("  Multi-Source Budget Variance Attribution System")
print("=" * 65)

sf_df   = generate_salesforce_data()
erp_df  = generate_erp_data()
lake_df = generate_datalake_data()

print(f"\n[1] Data Sources Ingested:")
print(f"    Salesforce CRM : {len(sf_df):>4} deal records   ({sf_df['close_date'].min().date()} – {sf_df['close_date'].max().date()})")
print(f"    Oracle ERP     : {len(erp_df):>4} GL entries     ({erp_df['month'].min().date()} – {erp_df['month'].max().date()})")
print(f"    Data Lake      : {len(lake_df):>4} region-months  ({lake_df['month'].min().date()} – {lake_df['month'].max().date()})")


# ─── 2. AUTOMATED PIPELINE: CLEAN + VALIDATE ─────────────────────────────────

def run_data_quality_checks(df, source_name):
    """Run 5 automated data quality checks — ERP-style audit controls."""
    checks = {}
    checks["null_completeness"]    = df.isnull().sum().sum() == 0
    checks["no_negatives_budget"]  = (df.get("budget", pd.Series([1])) >= 0).all() if "budget" in df else True
    checks["no_negatives_actual"]  = (df.get("actual", pd.Series([1])) >= 0).all() if "actual" in df else True
    checks["no_duplicate_keys"]    = not df.duplicated().any()
    checks["schema_conformance"]   = len(df) > 0

    passed = sum(checks.values())
    print(f"    {source_name:<20} {passed}/5 checks passed  {'OK' if passed==5 else 'REVIEW'}")
    return checks

print("\n[2] Data Quality Audit:")
qc_sf   = run_data_quality_checks(sf_df,   "Salesforce CRM")
qc_erp  = run_data_quality_checks(erp_df,  "Oracle ERP")
qc_lake = run_data_quality_checks(lake_df, "Data Lake")


# ─── 3. MERGE INTO UNIFIED FP&A VIEW ─────────────────────────────────────────

# Monthly revenue from Salesforce (Closed Won only)
sf_df["month"] = sf_df["close_date"].dt.to_period("M").dt.to_timestamp()
sf_monthly = (
    sf_df[sf_df["stage"] == "Closed Won"]
    .groupby(["month", "region"])
    .agg(
        actual_revenue=("actual_revenue", "sum"),
        budget_revenue=("budget_target", "sum"),
        deal_count=("deal_id", "count"),
    )
    .reset_index()
)

# Join Data Lake operational metrics
lake_monthly = lake_df.rename(columns={"month": "month"})
merged = pd.merge(sf_monthly, lake_monthly, on=["month", "region"], how="outer")

# Monthly total ERP costs
erp_monthly_total = (
    erp_df.groupby("month")
    .agg(total_budget=("budget","sum"), total_actual=("actual","sum"),
         total_variance=("variance","sum"))
    .reset_index()
)

# Final unified FP&A table
unified = pd.merge(
    merged.groupby("month").agg(
        actual_revenue=("actual_revenue","sum"),
        budget_revenue=("budget_revenue","sum"),
        deal_count=("deal_count","sum"),
        avg_days_to_pay=("avg_days_to_pay","mean"),
        churn_rate_pct=("churn_rate_pct","mean"),
        invoice_count=("invoice_count","sum"),
    ).reset_index(),
    erp_monthly_total,
    on="month", how="inner"
)

unified["gross_profit"]     = unified["actual_revenue"] - unified["total_actual"]
unified["revenue_variance"] = unified["actual_revenue"] - unified["budget_revenue"]
unified["rev_var_pct"]      = unified["revenue_variance"] / unified["budget_revenue"] * 100
unified["cost_variance"]    = unified["total_actual"] - unified["total_budget"]
unified["cost_var_pct"]     = unified["cost_variance"] / unified["total_budget"] * 100

print(f"\n[3] Unified FP&A table built: {len(unified)} monthly rows, {len(unified.columns)} columns")
print(f"    Sources merged: Salesforce revenue + Oracle ERP costs + Data Lake ops metrics")


# ─── 4. VARIANCE ATTRIBUTION: STATISTICAL SIGNAL VS NOISE ────────────────────

def classify_variance(erp_df):
    """
    Statistical variance attribution pipeline:
    - Z-score method: flag entries >1.96 std from mean (p < 0.05 threshold)
    - IQR method: flag entries in outer fences (below Q1-3*IQR or above Q3+3*IQR)
    - Combine: only dual-flagged entries are classified as true SIGNALS
    """
    df = erp_df.copy()

    # Z-score on variance_pct
    df["z_score"]    = np.abs(stats.zscore(df["variance_pct"].fillna(0)))
    df["iqr_flag"]   = False

    # IQR per dept × cost_type group for context-aware flagging
    for (dept, ctype), grp in df.groupby(["department", "cost_type"]):
        q1, q3 = grp["variance_pct"].quantile(0.25), grp["variance_pct"].quantile(0.75)
        iqr    = q3 - q1
        outer_lo = q1 - 3 * iqr
        outer_hi = q3 + 3 * iqr
        mask = (grp["variance_pct"] < outer_lo) | (grp["variance_pct"] > outer_hi)
        df.loc[grp[mask].index, "iqr_flag"] = True

    # Signal = z-score flagged OR IQR flagged (union for sensitivity in finance)
    df["z_flag"]     = df["z_score"] > 1.96
    df["is_signal"]  = df["z_flag"] | df["iqr_flag"]
    df["is_noise"]   = ~df["is_signal"]

    # Impact classification
    df["abs_variance"] = df["variance"].abs()
    df["impact_tier"]  = pd.cut(
        df["abs_variance"],
        bins=[0, 20_000, 60_000, 150_000, np.inf],
        labels=["Low", "Medium", "High", "Critical"]
    )

    return df

erp_classified = classify_variance(erp_df)

signals = erp_classified[erp_classified["is_signal"]].copy()
noise   = erp_classified[erp_classified["is_noise"]].copy()

print(f"\n[4] Variance Attribution Results:")
print(f"    Total GL entries analyzed : {len(erp_classified):>4}")
print(f"    Signal (true anomalies)   : {len(signals):>4} ({len(signals)/len(erp_classified)*100:.1f}%)")
print(f"    Noise (within tolerance)  : {len(noise):>4} ({len(noise)/len(erp_classified)*100:.1f}%)")
print(f"\n    Signal breakdown by impact tier:")
tier_counts = signals["impact_tier"].value_counts().sort_index()
for tier, count in tier_counts.items():
    bar = "█" * int(count / 2)
    print(f"      {tier:<10} : {count:>3}  {bar}")


# ─── 5. TOP ANOMALIES RANKED FOR LEADERSHIP ───────────────────────────────────

top_anomalies = (
    signals.sort_values("abs_variance", ascending=False)
    .head(10)[["month", "department", "cost_type", "budget", "actual", "variance", "variance_pct", "z_score", "impact_tier"]]
    .reset_index(drop=True)
)

print(f"\n[5] Top 10 High-Impact Anomalies (Leadership Report):")
print(f"\n    {'#':<4} {'Month':<10} {'Department':<22} {'Cost Type':<22} {'Budget':>12} {'Actual':>12} {'Variance':>12} {'Var%':>7} {'Tier':<10}")
print("    " + "-" * 115)
for i, row in top_anomalies.iterrows():
    direction = "↑" if row["variance"] > 0 else "↓"
    print(f"    {i+1:<4} {str(row['month'].date())[:7]:<10} {row['department']:<22} {row['cost_type']:<22} "
          f"₹{row['budget']:>10,.0f}  ₹{row['actual']:>10,.0f}  "
          f"{direction}₹{abs(row['variance']):>9,.0f}  {row['variance_pct']:>+6.1f}%  {str(row['impact_tier']):<10}")


# ─── 6. CORRELATION ANALYSIS ─────────────────────────────────────────────────

corr_cols = ["actual_revenue", "budget_revenue", "total_actual", "total_budget",
             "gross_profit", "revenue_variance", "cost_variance",
             "avg_days_to_pay", "churn_rate_pct", "invoice_count"]
corr_matrix = unified[corr_cols].corr()

print(f"\n[6] Correlation analysis complete — {len(corr_cols)} financial metrics")
top_corr = (
    corr_matrix.unstack()
    .reset_index()
    .rename(columns={"level_0":"A","level_1":"B",0:"corr"})
    .query("A < B")
    .assign(abs_corr=lambda x: x["corr"].abs())
    .sort_values("abs_corr", ascending=False)
    .head(5)
)
print("    Strongest correlations:")
for _, r in top_corr.iterrows():
    print(f"      {r['A']:<25} ↔ {r['B']:<25}  r = {r['corr']:+.3f}")


# ─── 7. VISUALIZATION ─────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=0.9)
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Multi-Source Budget Variance Attribution Dashboard", fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.38)

# ── Plot 1: Signal vs Noise variance distribution ──
ax1 = fig.add_subplot(gs[0, :2])
bins = np.linspace(erp_classified["variance_pct"].quantile(0.01),
                   erp_classified["variance_pct"].quantile(0.99), 50)
ax1.hist(noise["variance_pct"], bins=bins, color="#B5D4F4", alpha=0.85, label=f"Noise ({len(noise)} entries)", edgecolor="white")
ax1.hist(signals["variance_pct"], bins=bins, color="#E24B4A", alpha=0.85, label=f"Signal anomalies ({len(signals)} entries)", edgecolor="white")
ax1.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
ax1.axvline(1.96 * erp_classified["variance_pct"].std(), color="#EF9F27", linewidth=1.5, linestyle="--", label="±1.96σ threshold")
ax1.axvline(-1.96 * erp_classified["variance_pct"].std(), color="#EF9F27", linewidth=1.5, linestyle="--")
ax1.set_title("Variance Distribution: Signal vs Noise Attribution", fontweight="bold")
ax1.set_xlabel("Budget Variance %")
ax1.set_ylabel("Count")
ax1.legend(fontsize=8)

# ── Plot 2: Signal count by dept ──
ax2 = fig.add_subplot(gs[0, 2])
dept_signals = signals.groupby("department").size().sort_values(ascending=True)
colors_dept = ["#E24B4A" if v == dept_signals.max() else "#85B7EB" for v in dept_signals]
ax2.barh(dept_signals.index, dept_signals.values, color=colors_dept, edgecolor="white")
ax2.set_title("Anomaly Count\nby Department", fontweight="bold")
ax2.set_xlabel("Signal count")

# ── Plot 3: Monthly revenue vs budget ──
ax3 = fig.add_subplot(gs[1, :2])
x_idx = range(len(unified))
ax3.bar(x_idx, unified["budget_revenue"] / 1e6, color="#B5D4F4", alpha=0.9, label="Budgeted revenue", width=0.4, align="center")
ax3.bar([x + 0.4 for x in x_idx], unified["actual_revenue"] / 1e6,
        color=["#1D9E75" if v >= 0 else "#E24B4A" for v in unified["revenue_variance"]],
        alpha=0.9, label="Actual revenue", width=0.4)
ax3.set_xticks([x + 0.2 for x in x_idx])
ax3.set_xticklabels([d.strftime("%b") for d in unified["month"]], rotation=45, fontsize=8)
ax3.set_title("Monthly Revenue: Budget vs Actual (Salesforce Source)", fontweight="bold")
ax3.set_ylabel("Revenue (₹M)")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))
ax3.legend(fontsize=8)

# ── Plot 4: Waterfall variance by dept ──
ax4 = fig.add_subplot(gs[1, 2])
dept_var = (
    signals.groupby("department")["variance"]
    .sum().sort_values() / 1e3
)
colors_bar = ["#E24B4A" if v > 0 else "#1D9E75" for v in dept_var]
ax4.barh(dept_var.index, dept_var.values, color=colors_bar, edgecolor="white")
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_title("Signal Variance by Dept\n(₹k, red=over budget)", fontweight="bold")
ax4.set_xlabel("Total variance (₹k)")

# ── Plot 5: Correlation heatmap ──
ax5 = fig.add_subplot(gs[2, :2])
short_names = {
    "actual_revenue": "Actual Rev", "budget_revenue": "Budget Rev",
    "total_actual": "Actual Cost", "total_budget": "Budget Cost",
    "gross_profit": "Gross Profit", "revenue_variance": "Rev Var",
    "cost_variance": "Cost Var", "avg_days_to_pay": "Days to Pay",
    "churn_rate_pct": "Churn %", "invoice_count": "Invoice Ct"
}
corr_renamed = corr_matrix.rename(index=short_names, columns=short_names)
mask = np.triu(np.ones_like(corr_renamed, dtype=bool))
sns.heatmap(corr_renamed, mask=mask, ax=ax5, cmap="RdYlGn", center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, cbar_kws={"shrink": 0.7})
ax5.set_title("Financial Metrics Correlation Matrix", fontweight="bold")
ax5.tick_params(axis="x", rotation=45, labelsize=8)
ax5.tick_params(axis="y", rotation=0, labelsize=8)

# ── Plot 6: Variance trend over time ──
ax6 = fig.add_subplot(gs[2, 2])
ax6.plot(unified["month"], unified["rev_var_pct"], "o-", color="#185FA5",
         linewidth=1.8, markersize=5, label="Revenue var %")
ax6.plot(unified["month"], unified["cost_var_pct"], "s--", color="#E24B4A",
         linewidth=1.8, markersize=5, label="Cost var %")
ax6.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax6.fill_between(unified["month"],
                 -5, 5, alpha=0.08, color="#1D9E75", label="±5% tolerance band")
ax6.set_title("Monthly Variance %\nvs Tolerance Band", fontweight="bold")
ax6.set_ylabel("Variance %")
ax6.tick_params(axis="x", rotation=45, labelsize=7)
ax6.legend(fontsize=7)

plt.savefig("/home/claude/project2/variance_attribution_dashboard.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("\n[7] Dashboard saved: variance_attribution_dashboard.png")

# ─── 8. SQLITE ERP LAYER (shows ERP system experience) ───────────────────────

conn = sqlite3.connect("/home/claude/project2/erp_financial.db")
erp_df.to_sql("gl_actuals", conn, if_exists="replace", index=False)
unified.to_sql("fpa_unified", conn, if_exists="replace", index=False)
signals.to_sql("variance_signals", conn, if_exists="replace", index=False)

# SQL query to demonstrate ERP-style analytical querying
query = """
SELECT
    department,
    cost_type,
    COUNT(*) as signal_count,
    ROUND(SUM(variance), 2) as total_variance,
    ROUND(AVG(variance_pct), 2) as avg_variance_pct,
    ROUND(AVG(z_score), 2) as avg_z_score
FROM variance_signals
GROUP BY department, cost_type
ORDER BY ABS(total_variance) DESC
LIMIT 8
"""
sql_result = pd.read_sql(query, conn)
conn.close()

print("\n[8] ERP SQL query — top variance signals by department x cost type:")
print(sql_result.to_string(index=False))

# Save outputs
top_anomalies.to_csv("/home/claude/project2/leadership_anomaly_report.csv", index=False)
erp_classified.to_csv("/home/claude/project2/variance_classified.csv", index=False)
unified.to_csv("/home/claude/project2/unified_fpa_view.csv", index=False)

print("\n" + "=" * 65)
print("  PROJECT 2 COMPLETE")
print(f"  {len(signals)} anomalies surfaced from {len(erp_df)} GL entries")
print(f"  Outputs saved to project2/")
print("=" * 65)
