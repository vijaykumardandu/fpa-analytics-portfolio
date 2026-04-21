# 📊 FP&A Analytics Portfolio — Vijay Kumar Dandu

> **Role Applied For:** FP&A Intern (Predictive Analytics & Python Automation)
> 
> **Contact:** danduvijaykumar841@gmail.com | 7981003902 | Hyderabad
> 
> **LinkedIn:** linkedin.com/in/vijaykumar841 | **GitHub:** github.com/vijaykumardandu

---

## 👋 About This Portfolio

This repository contains **2 end-to-end data analytics projects** built specifically to demonstrate skills required for an FP&A Intern role focused on predictive modeling and Python automation.

Both projects simulate **real-world financial scenarios** that an FP&A team deals with daily — forecasting next month's revenue and identifying where the budget is going off-track.

**No prior FP&A work experience? These projects show I can do the job.**

---

## 🗂️ Repository Structure

fpa-analytics-portfolio/
│
├── project1_forecasting_engine/         ← Revenue & Cost Forecasting
│   ├── fpa_forecasting_engine.py        (main Python script)
│   ├── forecast_dashboard.png           (output chart)
│   ├── cost_margin_analysis.png         (output chart)
│   ├── feature_importance.png           (output chart)
│   ├── forecast_output.csv              (4-week projection results)
│   └── cleaned_invoice_data.csv         (processed dataset)
│
└── project2_variance_attribution/       ← Budget Variance Attribution
    ├── variance_attribution_system.py   (main Python script)
    ├── variance_attribution_dashboard.png (output dashboard)
    ├── leadership_anomaly_report.csv    (top 10 anomalies for management)
    ├── unified_fpa_view.csv             (merged financial data)
    └── variance_classified.csv          (all entries with signal/noise labels)

---

# 📁 Project 1: FP&A Revenue & Cost Forecasting Engine

## What Problem Does This Solve?

Every FP&A team needs to answer: **"What will our revenue and costs look like next month?"**

This project automates that answer. It takes 2 years of weekly invoice data (104 weeks,
250+ invoices), cleans it automatically, builds a predictive model, and produces a 4-week
revenue and cost forecast — complete with confidence intervals so leadership knows the
best-case and worst-case range.

## What Did I Actually Build?

| Step | What Happens | Tools Used |
|------|-------------|------------|
| Data Ingestion | Loads 104 weeks of raw invoice records | Python, Pandas |
| Data Cleaning | Auto-detects and fixes missing values via interpolation | Pandas |
| Feature Engineering | Creates 12 predictive features: lag values, rolling averages, seasonality encoding | NumPy, Pandas |
| Predictive Model | Linear Regression trained on historical patterns | Scikit-learn |
| Forecasting | Projects next 4 weeks of revenue and costs | Scikit-learn, NumPy |
| Confidence Intervals | Adds 95% upper/lower bounds to every forecast | SciPy |
| Visualization | 3 dashboard charts with trend lines and CI bands | Matplotlib, Seaborn |
| Output | CSV with weekly projections ready for Excel/Power BI | Pandas |

## Results (Plain English)

- The model predicts next month's revenue within **1.9% error on average**
- Total next-month revenue projected: **₹12,89,338** (range: ₹12,37,045 – ₹13,41,632)
- Gross margin forecast: **37.8%**
- Every weekly forecast includes lower and upper 95% confidence bounds

## Skills Demonstrated

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | SciPy | Linear Regression | Time-Series Modeling | Feature Engineering | Confidence Intervals | Automated Pipeline

---

# 📁 Project 2: Multi-Source Budget Variance Attribution System

## What Problem Does This Solve?

FP&A teams receive hundreds of budget line items every month. Most variances are just
normal fluctuation — **noise**. But a few are real problems leadership needs to act on — **signals**.

This project automatically tells the difference. It pulls data from 3 enterprise sources
(Salesforce CRM, Oracle ERP, Data Lake), merges them into one unified financial view,
then uses statistical methods to classify every budget variance as signal or noise —
delivering a ranked report to leadership.

## What Did I Actually Build?

| Step | What Happens | Tools Used |
|------|-------------|------------|
| Source 1 — Salesforce CRM | 320 deal records: closed revenue, pipeline, targets by region | Python, Pandas |
| Source 2 — Oracle ERP | 432 GL entries: dept actuals vs budgets across 6 cost types | Python, Pandas, SQLite |
| Source 3 — Data Lake | 60 region-month records: invoice counts, churn, payment timelines | Python, Pandas |
| Data Quality Audit | 5 automated checks per source (nulls, negatives, duplicates, schema) | Pandas |
| Data Merge | Joins all 3 sources into one unified FP&A monthly table | Pandas |
| Variance Attribution | Z-score + IQR dual-method flags true anomalies vs noise | SciPy, NumPy |
| Impact Classification | Tiers each anomaly: Low / Medium / High / Critical | Pandas |
| Leadership Report | Top 10 anomalies ranked by financial impact | Pandas, CSV |

## Results (Plain English)

- Analyzed **432 budget GL entries** across 6 departments × 6 cost types × 12 months
- Automatically identified **63 true anomalies (14.6%)** — 369 confirmed as normal noise
- Found **3 Critical-tier anomalies** each exceeding ₹1,50,000 over budget
- Top finding: Sales & Marketing Travel **+57.1% over budget** in November 2024
- Leadership reviews **63 items instead of 432** — **85% reduction** in manual workload
- All 3 data sources passed **5/5 data quality checks**

## Sample Leadership Report (Top 5)

| # | Month | Department | Cost Type | Budget | Actual | Over/Under | Tier |
|---|-------|-----------|-----------|--------|--------|------------|------|
| 1 | Nov 2024 | Sales & Marketing | Travel | ₹3,01,863 | ₹4,74,143 | +57.1% | Critical |
| 2 | Jun 2024 | IT | Infrastructure | ₹3,20,895 | ₹5,02,170 | +56.5% | Critical |
| 3 | Jan 2024 | Sales & Marketing | R&D | ₹3,76,453 | ₹5,63,767 | +49.8% | Critical |
| 4 | Feb 2024 | Operations | Infrastructure | ₹3,81,133 | ₹5,26,730 | +38.2% | High |
| 5 | Feb 2024 | HR | Vendor/SaaS | ₹3,61,873 | ₹5,02,487 | +38.9% | High |

## Skills Demonstrated

Python | Pandas | NumPy | SQL | SQLite | SciPy | Matplotlib | Seaborn | Z-Score Analysis | IQR Anomaly Detection | Multi-Source ETL Pipeline | Correlation Analysis | Data Quality Controls | ERP Data Modeling

---

## 🛠️ How to Run

pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels

Project 1:  python fpa_forecasting_engine.py
Project 2:  python variance_attribution_system.py

Both scripts are fully self-contained — they generate data, run analysis, and save all outputs automatically.

---

## 💼 Skills Summary

| Category | Tools |
|----------|-------|
| Programming | Python, SQL |
| Data Manipulation | Pandas, NumPy |
| Statistical Modeling | Linear Regression, Z-Score, IQR, Correlation, Confidence Intervals |
| Visualization | Matplotlib, Seaborn, Power BI |
| Data Engineering | ETL Pipelines, Multi-Source Merging, Data Quality Controls |
| Finance Domain | FP&A, Budget Variance, Revenue Forecasting, Cost Modeling |
| Databases | SQLite (ERP-layer simulation) |

---

## 📬 Contact

Email: danduvijaykumar841@gmail.com
Phone: 7981003902
LinkedIn: linkedin.com/in/vijaykumar841
Location: Hyderabad, India
