# Azure Sentiment Data Pipeline

End-to-end Data Engineering pipeline that ingests, cleans, and analyzes tweet sentiment using a hybrid AI architecture (Machine Learning + LLM fallback), orchestrated in a cloud environment using Azure services.

---

## 🧠 Project Overview

This project simulates a real-world production data pipeline for social media analytics.

It processes raw tweets, applies sentiment analysis using a pre-trained transformer model, and improves prediction reliability using a Large Language Model (LLM) fallback mechanism.

The final output is stored in a structured format ready for analytics and business intelligence.

---

## 🏗️ Architecture & Technologies

### 💻 Stack
- **Language:** Python
- **Big Data Processing:** PySpark (Azure Databricks)
- **Machine Learning:** HuggingFace Transformers
- **LLM Integration:** Google Gemini API

### ☁️ Azure Services
- Azure Data Factory (ADF) → Pipeline orchestration
- Azure Databricks → Scalable data processing with Spark
- Azure Data Lake Storage → Raw & processed data storage
- Azure SQL Database → Analytical data warehouse
- Azure Key Vault → Secure secrets management

---

## 🔄 Pipeline Workflow

### 1. Data Ingestion
- Raw tweet datasets are uploaded to Azure Data Lake Storage (`raw-data` container).

### 2. Event-Based Orchestration
- Azure Data Factory triggers the pipeline when new data is detected.

### 3. Data Processing (Databricks / Python Local Prototype)
- Data cleaning using Pandas and Regex
- Standardization of text (lowercase, removal of noise)
- Structured logging and configuration-driven file paths

### 4. Sentiment Analysis (Hybrid AI System)

The system uses a **two-layer AI architecture**:

#### 🧠 Layer 1 — Local ML Model
- Pre-trained HuggingFace model:
  - `distilbert-base-uncased-finetuned-sst-2-english`
- Provides fast, low-cost sentiment prediction

#### 🤖 Layer 2 — LLM Fallback (Gemini)
- Activated when:
  - Model confidence is low
  - Text contains ambiguous or conflictive language
- Used for semantic reasoning and edge-case interpretation

---

### 5. Data Storage
- Processed results are stored in:
  - Parquet format (Data Lake optimized)
  - Azure SQL Database (analytics-ready layer)

---

### 6. Analytics Layer
- Data is structured for:
  - Brand sentiment tracking
  - Trend analysis over time
  - Business intelligence dashboards (Power BI compatible)

---


## 🚀 How to local run

```bash
# Clone and set up
git clone https://github.com/LisaLorita/azure-sentiment-data-pipeline.git
cd azure-sentiment-data-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install streamlit python-dotenv   # extra for the dashboard

# Create .env (do NOT commit) as in .env.example

# Run the steps
python src/02_clean_data.py           # create data/clean_train.csv
python src/03_ia_model.py            # hybrid sentiment on a few tweets
python -m streamlit run src/05_dashboard.py   # open the dashboard in a browser
```

## 📊 Dashboard overview

- **Metrics**: total tweets processed.
- **Bar chart**: distribution of Positive / Negative / Neutral.
- **Filters**: select a brand or search keywords.
- **Live inference**: type any tweet; the app decides whether to use the local model or call Gemini and shows the final sentiment.

## 🔐 Security (Best Practices)

No hardcoded passwords or keys exist in the source code. All sensitive information (Storage Account access keys, SQL Server passwords) are stored securely in a `.env` file for local development and **Azure Key Vault** for the cloud. Databricks accesses these credentials at runtime using a configured _Secret Scope_.

## 👩‍💻 Stay in touch

- Author - Alba Martin
- Github - https://github.com/LisaLorita
