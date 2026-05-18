# Azure Sentiment Data Pipeline

Automated Data Engineering pipeline that ingests, cleans, and analyzes the sentiment of tweets using a HuggingFace AI model, orchestrated in the Azure Cloud ecosystem.

## 🏗️ Architecture & Technologies

This project simulates a data pipeline, moving from raw data ingestion to analytical storage.

- **Language:** Python & PySpark
- **AI Model:** HuggingFace `transformers`
- **Cloud Provider:** Microsoft Azure
  - **Azure Databricks**
  - **Azure Data Factory (ADF):**
  - **Azure SQL Database:**
  - **Azure Key Vault:**

## 🔄 Pipeline Workflow

1.  **Event Trigger:** A new raw CSV file containing tweets is dropped into the `raw-data` container in Azure Data Lake.
2.  **Orchestration:** Azure Data Factory detects the _Storage Event_ and automatically triggers the pipeline.
3.  **Processing (Databricks):**
    - Reads the raw CSV from the Data Lake.
    - Cleans the text (removes URLs, mentions, special characters) using regular expressions.
    - Applies a pre-trained NLP model to classify the sentiment of each tweet (Positive/Negative).
    - Groups and aggregates the data to generate a daily summary.
4.  **Storage:** The processed data is saved back to the Data Lake in highly efficient **Parquet** format.
5.  **Analytics:** The final summarized data is written to an **Azure SQL Database** via JDBC, ready to be consumed by Data Analysts.

## 🔐 Security (Best Practices)

No hardcoded passwords or keys exist in the source code. All sensitive information (Storage Account access keys, SQL Server passwords) are stored securely in **Azure Key Vault**. Databricks accesses these credentials at runtime using a configured _Secret Scope_.

## 🚀 How to Run (Local Exploration)

If you want to run the initial exploration scripts locally:

1. Clone the repository.
2. Activate the virtual environment: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the local cleaning script: `python src/02_clean_data.py`

## Stay in touch

- Author - Alba Martin
- Github - https://github.com/LisaLorita
