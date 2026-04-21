# 📊 FP&A Analytics Portfolio — Vijay Kumar Dandu

## 👋 About This Portfolio

This repository contains **2 complete, independent data analytics projects** built to demonstrate the exact skills required for an FP&A Intern role — predictive modeling, automated data pipelines, statistical variance analysis, and financial dashboards.

Both projects solve **real problems FP&A teams face every day:**
- Project 1 answers: *"What will our revenue and costs be next month?"*
- Project 2 answers: *"Which budget variances are real problems vs normal noise?"*

> All code is fully self-contained — runs end-to-end and generates every output file and chart automatically.

---

## 🗂️ Repository Structure

```
fpa-analytics-portfolio/
│
├── README.md
│
├── project1_forecasting_engine/
│   ├── fpa_forecasting_engine.py       ← main Python script
│   ├── forecast_dashboard.png          ← output: main forecast chart
│   ├── cost_margin_analysis.png        ← output: cost & margin breakdown
│   ├── feature_importance.png          ← output: model feature chart
│   ├── forecast_output.csv             ← output: 4-week projections
│   └── cleaned_invoice_data.csv        ← output: processed dataset
│
└── project2_variance_attribution/
    ├── variance_attribution_system.py  ← main Python script
    ├── variance_attribution_dashboard.png ← output: 6-panel dashboard
    ├── leadership_anomaly_report.csv   ← output: top 10 anomalies
    ├── unified_fpa_view.csv            ← output: merged financial data
    └── variance_classified.csv         ← output: all GL entries classified
```

---
---

# 📁 Project 1 — FP&A Revenue & Cost Forecasting Engine

## 🔍 What Problem Does This Solve?

Finance teams manually build forecasts in Excel every month — it's slow, error-prone, and doesn't scale. This project **automates the entire forecasting pipeline** using Python:

- Ingests 2 years of weekly invoice data (104 weeks, 250+ weekly invoices)
- Cleans and prepares the data automatically
- Engineers 12 predictive features from raw numbers
- Builds a Linear Regression model to predict revenue and costs
- Outputs a 4-week forward forecast with **95% confidence intervals**
- Generates 3 ready-to-present dashboard charts

---

## ⚙️ How It Works — Step by Step

| Step | What Happens | Tool |
|------|-------------|------|
| 1 | Generates 104 weeks of invoice data with trend, seasonality & noise | Python, NumPy |
| 2 | Detects and fixes missing values automatically via interpolation | Pandas |
| 3 | Engineers lag features (1-week, 4-week), rolling MAs (4-week, 13-week), sin/cos seasonality encoding | Pandas, NumPy |
| 4 | Trains Linear Regression on 88 weeks, tests on last 4 weeks (time-series split — no data leakage) | Scikit-learn |
| 5 | Projects next 4 weeks of revenue and total cost | Scikit-learn |
| 6 | Calculates 95% confidence intervals using residual standard deviation | SciPy, NumPy |
| 7 | Saves forecast output to CSV (ready for Power BI / Excel) | Pandas |
| 8 | Generates 3 dashboard charts | Matplotlib, Seaborn |

---

## 📊 Dashboard Output — Charts

### Chart 1: Revenue Forecast Dashboard
*Actual weekly revenue → Linear Regression fit → 4-week forecast with 95% CI bands + 13-week moving average trend line*


<img width="2094" height="1668" alt="forecast_dashboard" src="https://github.com/user-attachments/assets/5109d97d-5f51-4ca8-8232-eb2e1711baec" />


---

### Chart 2: Cost & Margin Analysis
*Quarterly COGS vs OPEX breakdown (left) + Gross Margin trend with forecast (right)*

<img width="2080" height="738" alt="cost_margin_analysis" src="https://github.com/user-attachments/assets/7f87f63a-a2a6-4cbd-9349-f32ac559eb45" />


---

### Chart 3: Feature Importance
*Which engineered features contribute most to the revenue prediction model*

<img width="1328" height="879" alt="feature_importance" src="https://github.com/user-attachments/assets/6788edb7-e85e-48ea-8b53-2e9cec4f6abb" />


---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Revenue forecast error (MAPE) | **1.9%** — strong predictive accuracy |
| Next month revenue projection | **₹12,89,338** |
| 95% confidence range | ₹12,37,045 – ₹13,41,632 |
| Gross margin forecast | **37.8%** |
| Features engineered | 12 (lag-1, lag-4, MA-4, MA-13, sin/cos, quarter flags) |
| Forecast horizon | 4 weeks with per-week CI bounds |

---

## 🛠️ Skills Demonstrated

`Python` `Pandas` `NumPy` `Scikit-learn` `SciPy` `Matplotlib` `Seaborn`
`Linear Regression` `Time-Series Modeling` `Feature Engineering`
`Confidence Intervals` `Automated Data Pipeline` `Financial Forecasting`

---
---

# 📁 Project 2 — Multi-Source Budget Variance Attribution System

## 🔍 What Problem Does This Solve?

Every month, FP&A teams receive **hundreds of budget line items**. Manually reviewing all of them to find the real problems is time-consuming and unreliable.

This project builds an **automated variance attribution system** that:
- Pulls data from 3 enterprise sources (Salesforce CRM, Oracle ERP, Data Lake)
- Merges them into one unified FP&A view
- Runs statistical analysis to separate **true anomalies (signals)** from **normal fluctuation (noise)**
- Delivers a ranked leadership report of only the items that need attention

---

## ⚙️ How It Works — Step by Step

| Step | What Happens | Tool |
|------|-------------|------|
| 1 | Ingests Salesforce CRM — 320 deal records (revenue, pipeline, targets by region) | Python, Pandas |
| 2 | Ingests Oracle ERP — 432 GL entries (actuals vs budgets across 6 depts × 6 cost types × 12 months) | Python, Pandas, SQLite |
| 3 | Ingests Data Lake — 60 region-month records (invoice count, churn, days-to-pay) | Python, Pandas |
| 4 | Runs 5 automated data quality checks per source (nulls, negatives, duplicates, schema, completeness) | Pandas |
| 5 | Merges all 3 sources into one unified monthly FP&A table | Pandas |
| 6 | Applies Z-score flagging (>1.96σ threshold) on variance % | SciPy |
| 7 | Applies IQR outer-fence flagging per dept × cost type group | NumPy |
| 8 | Classifies each entry: Signal (anomaly) or Noise (normal) | Pandas |
| 9 | Tiers signals by impact: Low / Medium / High / Critical | Pandas |
| 10 | Runs SQL analytics on ERP layer to find top dept × cost type combinations | SQLite |
| 11 | Outputs ranked leadership report (top 10 anomalies) | Pandas, CSV |
| 12 | Generates 6-panel dashboard with correlation heatmap | Matplotlib, Seaborn |

---

## 📊 Dashboard Output — Charts

### 6-Panel Variance Attribution Dashboard
*Signal vs noise distribution | Anomaly count by department | Monthly budget vs actual | Department waterfall | Correlation heatmap | Monthly variance % trend*


<img width="2256" height="1982" alt="variance_attribution_dashboard" src="https://github.com/user-attachments/assets/fc4bdf29-1212-4289-afde-604c247220f9" />

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Total GL entries analyzed | **432** |
| True anomalies detected (signals) | **63 (14.6%)** |
| Normal entries confirmed (noise) | **369 (85.4%)** |
| Critical-tier anomalies (>₹1.5L over budget) | **3** |
| Top anomaly | Sales & Marketing Travel **+57.1% over budget** |
| Leadership review workload reduced by | **85%** (63 items vs 432) |
| Data quality pass rate | **5/5 checks on all 3 sources** |

---

## 🏆 Top 10 Anomalies — Leadership Report

| # | Month | Department | Cost Type | Budget | Actual | Variance | Tier |
|---|-------|-----------|-----------|--------|--------|----------|------|
| 1 | Nov 2024 | Sales & Marketing | Travel | ₹3,01,863 | ₹4,74,143 | +57.1% | 🔴 Critical |
| 2 | Jun 2024 | IT | Infrastructure | ₹3,20,895 | ₹5,02,170 | +56.5% | 🔴 Critical |
| 3 | Jan 2024 | Sales & Marketing | R&D | ₹3,76,453 | ₹5,63,767 | +49.8% | 🔴 Critical |
| 4 | Feb 2024 | Operations | Infrastructure | ₹3,81,133 | ₹5,26,730 | +38.2% | 🟠 High |
| 5 | Feb 2024 | HR | Vendor/SaaS | ₹3,61,873 | ₹5,02,487 | +38.9% | 🟠 High |
| 6 | Mar 2024 | Engineering | Travel | ₹2,57,702 | ₹3,96,004 | +53.7% | 🟠 High |
| 7 | Feb 2024 | Operations | Marketing Spend | ₹2,84,081 | ₹4,21,627 | +48.4% | 🟠 High |
| 8 | May 2024 | Engineering | Vendor/SaaS | ₹2,76,977 | ₹4,13,652 | +49.4% | 🟠 High |
| 9 | Sep 2024 | IT | Infrastructure | ₹3,50,124 | ₹2,28,298 | -34.8% | 🟠 High |
| 10 | Jun 2024 | Sales & Marketing | Travel | ₹3,01,235 | ₹4,17,928 | +38.7% | 🟠 High |

---

## 🛠️ Skills Demonstrated

`Python` `Pandas` `NumPy` `SQL` `SQLite` `SciPy` `Matplotlib` `Seaborn`
`Z-Score Analysis` `IQR Anomaly Detection` `Multi-Source ETL Pipeline`
`Correlation Analysis` `Data Quality Controls` `ERP Data Modeling`
`Budget Variance Attribution` `Statistical Signal vs Noise`

---
---

## 🚀 How to Run Both Projects

### Install dependencies (one time)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

### Run Project 1
```bash
cd project1_forecasting_engine
python fpa_forecasting_engine.py
```

### Run Project 2
```bash
cd project2_variance_attribution
python variance_attribution_system.py
```

> Both scripts are 100% self-contained — they generate synthetic data, run the full analysis pipeline, and save all charts and CSV outputs automatically. No external data files needed.

---

## 💼 Full Skills Summary

| Category | Tools & Technologies |
|----------|---------------------|
| Programming | Python, SQL |
| Data Manipulation | Pandas, NumPy |
| Statistical Modeling | Linear Regression, Z-Score, IQR, Correlation Analysis, Confidence Intervals |
| Visualization | Matplotlib, Seaborn, Power BI |
| Data Engineering | Automated ETL Pipelines, Multi-Source Merging, Data Quality Audits |
| Finance Domain | FP&A, Budget Variance Analysis, Revenue Forecasting, Cost Modeling, GL Accounting |
| Databases | SQLite — ERP-layer analytical queries |

---

## 📬 Get In Touch

I am actively looking for Data Analyst, FP&A Intern opportunities where I can apply Python automation and statistical modeling to improve financial forecasting and reporting accuracy.

| | |
|--|--|
| **Email** | danduvijaykumar841@gmail.com |
| **Phone** | 7981003902 |
| **LinkedIn** | linkedin.com/in/vijaykumar841 |
| **GitHub** | github.com/vijaykumardandu |
| **Location** | Hyderabad, India |
