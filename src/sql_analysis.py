import pandas as pd
import sqlite3
import os

def run_sql_analysis(data_path='data/raw/user_data.csv'):
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run data_generation.py first.")
        return

    print("Loading data into in-memory SQLite database...")
    df = pd.read_csv(data_path)
    
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    df.to_sql('users', conn, index=False)
    
    print("\n--- Executing SQL Analysis ---")
    
    # 1. High-value user segmentation
    query_1 = """
    SELECT 
        CASE 
            WHEN past_purchases >= 5 THEN 'High Value'
            WHEN past_purchases >= 2 THEN 'Medium Value'
            ELSE 'Low Value'
        END AS user_segment,
        COUNT(user_id) as user_count,
        AVG(time_spent_mins) as avg_time_spent,
        AVG(is_retained) * 100 as retention_rate_pct
    FROM users
    GROUP BY user_segment
    ORDER BY avg_time_spent DESC;
    """
    print("\n1. User Segmentation by Past Purchases:")
    print(pd.read_sql_query(query_1, conn))
    
    # 2. A/B Test preliminary results (SQL approach)
    query_2 = """
    SELECT 
        is_treatment as group_type,
        COUNT(user_id) as total_users,
        AVG(time_spent_mins) as avg_engagement_mins,
        SUM(is_retained)*1.0 / COUNT(user_id) * 100 as retention_rate_pct
    FROM users
    GROUP BY is_treatment;
    """
    print("\n2. Preliminary A/B Test Results (Control=0, Treatment=1):")
    print(pd.read_sql_query(query_2, conn))
    
    # 3. Cohort Analysis based on signup vintage
    query_3 = """
    SELECT 
        CASE 
            WHEN days_since_signup <= 30 THEN 'New (<30 days)'
            WHEN days_since_signup <= 180 THEN 'Established (30-180 days)'
            ELSE 'Veteran (>180 days)'
        END as tenure_cohort,
        COUNT(user_id) as user_count,
        AVG(is_retained) * 100 as retention_rate_pct
    FROM users
    GROUP BY tenure_cohort
    ORDER BY retention_rate_pct DESC;
    """
    print("\n3. Retention by Tenure Cohort:")
    print(pd.read_sql_query(query_3, conn))

if __name__ == "__main__":
    run_sql_analysis()
