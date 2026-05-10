# User Behavior Analysis & Causal Inference

This repository contains a comprehensive data science portfolio project focusing on User Behavior Analysis, A/B Testing, and Causal Inference. The analysis uses Python, SQL, and robust statistical methods (`statsmodels`) to uncover behavioral patterns and quantify the business impact of product changes.

## 🎯 Project Overview

In this project, we simulated and analyzed a dataset of over **100,000 user records**. The primary goals were:
1. **Behavioral Analysis:** Use SQL and clustering to segment users based on their engagement and purchasing behavior.
2. **Causal Inference & A/B Testing:** Evaluate a new product feature (Treatment) designed to boost user engagement and retention.

### Key Business Results
- **Engagement Lift:** Successfully demonstrated a statistically significant increase in user engagement by **~15%**.
- **Retention Lift:** Validated a relative retention improvement of **~10%** using logistic regression and causal modeling.

## 🛠️ Technology Stack
- **Languages:** Python, SQL (SQLite)
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- **Methodologies:** Linear Regression (OLS), Logistic Regression, K-Means Clustering, A/B Testing

## 📂 Project Structure

```text
.
├── data/
│   ├── raw/                   # Generated synthetic raw data
│   └── processed/             # Output from clustering
├── src/
│   ├── data_generation.py     # Generates 100K+ synthetic user records
│   ├── sql_analysis.py        # Performs SQLite in-memory analysis
│   ├── clustering_analysis.py # K-Means clustering for user segmentation
│   └── causal_inference.py    # A/B testing analysis using statsmodels
├── run_pipeline.py            # Master script to execute the end-to-end pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run the Project

1. **Clone the repository and navigate to the directory:**
   ```bash
   git clone <your-repository-url>
   cd User-Behavior-Analysis-Causal-Inference
   ```

2. **Install the dependencies:**
   It's highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the pipeline:**

   You can run the entire pipeline end-to-end using the master script:
   ```bash
   python run_pipeline.py
   ```

   *Alternatively, you can run the scripts individually:*

   *Step 1: Generate the synthetic dataset (100K records)*
   ```bash
   python src/data_generation.py
   ```

   *Step 2: Run the SQL Analysis*
   ```bash
   python src/sql_analysis.py
   ```

   *Step 3: Perform Behavioral Clustering*
   ```bash
   python src/clustering_analysis.py
   ```

   *Step 4: Execute Causal Inference and A/B Testing*
   ```bash
   python src/causal_inference.py
   ```

## 🧠 Methodology Deep-Dive

### 1. Data Generation (`data_generation.py`)
To make this project easily reproducible without sharing sensitive proprietary data, a script generates synthetic user behavior data. It carefully injects specific causal effects (a +3 min time spent increase, and a logit shift for retention) so that our causal models can recover these "true" effects.

### 2. SQL Segmentation (`sql_analysis.py`)
Uses `sqlite3` to load the 100K records into memory and execute complex aggregation queries. This step simulates a typical data analyst workflow, segmenting users by value and extracting cohort retention metrics.

### 3. Unsupervised Clustering (`clustering_analysis.py`)
Applies K-Means clustering using `scikit-learn` to identify latent behavioral groups. It automatically scales features and segments the user base into distinct personas like "Power Users", "Regular Users", and "Casual/New Users".

### 4. Causal Inference (`causal_inference.py`)
This is the core statistical component.
- **Continuous Outcomes (Engagement):** Uses Ordinary Least Squares (OLS) regression to estimate the Average Treatment Effect (ATE) while controlling for confounding variables like age and past purchases.
- **Binary Outcomes (Retention):** Employs Logistic Regression to estimate the probability of retention. By calculating the marginal effects across the population, we accurately quantify the absolute and relative lift in retention caused by the treatment.
