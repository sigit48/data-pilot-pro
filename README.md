# 🚀 Data Pilot Pro — Executive Analytics Engine  
**AI-Powered Enterprise Analytics Platform for Tactical Decision Intelligence**

---

## 📌 Project Snapshot (TL;DR)

**Data Pilot Pro** is an enterprise-grade analytics platform that combines:

- **Machine Learning**
- **Multi-Agent Generative AI**
- **Smart Data Cleaning**
- **Natural Language Analytics**
- **Anomaly Detection**
- **Forecasting Intelligence**

to transform raw operational data into **tactical executive recommendations** within seconds.

Designed for executives, analysts, and decision-makers, the platform turns fragmented datasets into **actionable business command orders** using AI-powered reasoning and automated KPI analysis.

---

## 🌍 Why This Project Matters

Organizations often struggle with:

- Fragmented and dirty datasets  
- Slow reporting cycles  
- Manual KPI calculations  
- Delayed executive decision-making  
- Limited tactical interpretation from dashboards  

**Data Pilot Pro addresses these challenges by automating the entire analytics workflow** — from data ingestion and cleaning to executive-level recommendations.

Instead of merely visualizing data, the system generates **decision intelligence**.

---

# ✨ Core Features

## 🔒 1. Embedded License Engine (HMAC-SHA256)

Enterprise-ready local licensing architecture.

### Capabilities
- Secure local authentication  
- No external licensing server required  
- HMAC-SHA256 cryptographic validation  
- Tier-based access control

### License Tiers

| Tier | Data Capacity |
|------|---------------|
| TRIAL | 500 rows |
| BASIC | 5,000 rows |
| PRO | 500,000 rows |
| ENTERPRISE | 999,999+ rows |

---

## 🧼 2. Smart Data Cleaning Engine

Built with **Polars** and **Pandas** for scalable preprocessing.

### Automated Cleaning
- Null synonym handling (`N/A`, `empty`, `?`, `-`, etc.)
- Automatic `snake_case` column standardization
- Currency symbol normalization
- Invisible character cleaning
- Whitespace optimization

### Intelligent Type Detection
- Numeric
- Date/Time
- Categorical
- Mixed Types

### Advanced Processing
- Outlier tagging via **IQR × 3.0**
- Missing value imputation:
  - Median (numeric)
  - `"Unknown"` (categorical)

---

## 📐 3. 40+ KPI & Formula Engine

Cross-functional KPI calculation across departments.

### 💰 Financial Analytics
- ROI
- EBITDA Margin
- Gross Profit Margin
- Current Ratio
- Debt-to-Equity Ratio

### 📈 Sales & Marketing
- Conversion Rate
- Average Order Value (AOV)
- Churn Rate
- ROAS
- CTR
- NPS

### 🔗 Supply Chain
- Inventory Turnover
- Days Inventory Outstanding (DIO)
- OTIF
- Fill Rate
- Cash Conversion Cycle (CCC)

### ⚙️ Operations
- Overall Equipment Effectiveness (OEE)
- Utilization Rate
- MTTR
- MTBF
- Defect Rate

### 👥 HR Analytics
- Employee Turnover Rate
- Revenue per Employee
- Absenteeism Rate

### Interactive Visualizations
Dynamic Plotly-based charts:
- Distribution Analysis
- Time-Series Trends
- Box Plots
- Correlation Analysis

---

## ⚔️ 4. Command Control War Room (AI Multi-Agent System)

A collaborative **multi-agent AI architecture** for tactical business analysis.

### AI Agents

#### 🚢 Logistics & Supply Chain Agent
Analyzes:
- Warehouse operations
- Distribution bottlenecks
- Supply chain inefficiencies

#### 💰 Financial & Risk Agent
Analyzes:
- Margin optimization
- Cost control
- Capital allocation
- Financial risks

#### ⭐ Chief Commander
Synthesizes findings into a tactical:

**Executive Command Order**

Including:
- Situation Assessment
- Tactical Actions
- PIC Assignment
- Deadlines
- Operational Status

### Advanced Capabilities
- **Mini-RAG document ingestion**
- PDF, Excel, CSV, TXT support
- Agent debate mode
- Cross-agent reasoning

---

## 🔍 5. Natural Language Query (NLQ)

Ask business questions in plain language.

Example:

> “Which products generate the highest operational risk?”

The system automatically:

1. Generates Pandas code  
2. Executes in sandbox environment  
3. Produces charts  
4. Returns business interpretation  

### AI Auto-Recovery
Automatic one-cycle retry if generated code fails.

---

## 👁️ 6. AI Vision OCR

Convert visual reports into structured datasets.

Supports:
- Dashboard screenshots
- Printed reports
- Table images

### Optimization Features
- Automatic image compression
- API token optimization
- MD5 encrypted client-side caching

---

## 🚨 7. Machine Learning Anomaly Detection

Smart anomaly detection using:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **Z-Score Analysis**

Designed for multidimensional business anomalies.

---

## 📈 8. Advanced Forecasting Engine

Business metric forecasting using:

- **Holt-Winters ETS**
- **Meta Prophet** *(optional)*

Includes:
- Confidence interval visualization
- Forecast uncertainty banding

---

# 🛠 Tech Stack

### Data Engineering
- Python
- Pandas
- Polars
- NumPy

### Machine Learning & Statistics
- Scikit-learn
- Statsmodels
- SciPy

### AI Integration
- Google Gemini API
- Multi-Agent Generative AI

### Visualization
- Plotly
- Streamlit

### File Processing
- OpenPyXL
- PyPDF2
- pdfplumber
- ReportLab

---

# ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/sigit48/data-pilot-pro.git
cd datapilot-pro
```

### 2. Create Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
streamlit run app.py
```

Application will run at:

```text
http://localhost:8501
```

---

## 🤖 AI Configuration (Gemini API)

To activate AI-powered features:

- Command War Room  
- NLQ Analytics  
- Vision OCR  
- Smart Narrative  

Create a free API key via **Google AI Studio** and insert it into the application sidebar.

---

## 📁 Project Structure

```text
datapilot-pro/
│
├── app.py
├── README.md
├── requirements.txt
├── .datapilot_license
│
├── exports/
│   ├── pdf/
│   ├── csv/
│   └── json/
│
└── assets/
```

---

## 🎯 Strategic Positioning

Data Pilot Pro transforms analytics from:

> **Passive Dashboard Consumption**

into:

> **AI-Assisted Tactical Decision Intelligence**

This project demonstrates capabilities in:

- Enterprise Analytics Engineering  
- AI Product Development  
- Machine Learning Integration  
- Multi-Agent Systems  
- Executive Decision Support  
- Business Intelligence Architecture  

---

## 👤 Author

**Sigit Dwiantoro**  
Business Intelligence | Financial & Risk Analytics | AI-Powered Decision Systems

---

## 📄 Disclaimer

This platform is developed for **educational, research, and portfolio demonstration purposes**.

---

> **Navigate Your Data. Command Your Strategy.**
