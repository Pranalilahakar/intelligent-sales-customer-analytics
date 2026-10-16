# Intelligent Sales & Customer Analytics Platform

An end-to-end analytics project combining retail sales analysis, customer segmentation, inactivity analysis, customer prioritization, a REST API, and Power BI dashboards.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- PostgreSQL and SQL
- Power BI
- FastAPI and Uvicorn
- Pytest
- Git and GitHub

## Project Features

### Sales Analytics
Analyzes revenue, quantities, invoices, customers, countries, and product performance.

### Customer Segmentation
Uses Recency, Frequency, and Monetary (RFM) features to group customers and summarize their purchasing behavior.

### Customer Inactivity Analysis
Classifies customers using a 90-day inactivity threshold.

This is a retrospective inactivity-based classification, not a validated predictive churn model.

### Customer Priority Scoring
Calculates a weighted, rule-based customer priority score:

- Recency: 40%
- Frequency: 30%
- Monetary value: 30%

Customers are assigned Low, Medium, or High Priority categories. These scores are not validated predictions of lead conversion.

### FastAPI REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome information |
| GET | `/health` | Check API and data-file availability |
| GET | `/summary` | Sales and customer summary |
| GET | `/segments` | Customer segment summary |
| GET | `/segments/{customer_id}` | Details for one customer's segment |
| GET | `/priority` | Ranked customer priority list |
| GET | `/priority/{customer_id}` | One customer's priority details |

Interactive documentation is available at `/docs` when the API server is running.

## Project Structure

```text
intelligent-sales-analytics/
├── api/
│   └── main.py
├── data/
│   └── processed/
├── notebooks/
├── sql/
├── tests/
│   └── test_api.py
├── analyze_data.py
├── churn_analysis.py
├── churn_business_impact.py
├── clean_data.py
├── customer_priority_scoring.py
├── customer_segmentation.py
├── explore_data.py
├── load_data.py
├── requirements.txt
├── .gitignore
└── README.md




## Setup and Run

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Start the API

```powershell
python -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

