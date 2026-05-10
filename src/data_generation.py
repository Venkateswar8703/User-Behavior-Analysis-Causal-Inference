import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_records=100_000, output_path='data/raw/user_data.csv'):
    np.random.seed(42)
    print(f"Generating {num_records} synthetic user records...")

    # Basic demographics and features
    user_ids = np.arange(1, num_records + 1)
    ages = np.random.normal(loc=35, scale=10, size=num_records).clip(18, 80).astype(int)
    days_since_signup = np.random.randint(1, 365, size=num_records)
    past_purchases = np.random.poisson(lam=2, size=num_records)
    
    # Randomly assign 50% to treatment group (A/B Test)
    # Treatment is a new UI feature aimed at improving engagement and retention
    is_treatment = np.random.choice([0, 1], size=num_records)
    
    # Base engagement (e.g., time spent in minutes per week)
    # Adding some noise and correlations with other features
    base_time_spent = 15 + (ages - 35) * -0.1 + past_purchases * 2 + np.random.normal(0, 5, num_records)
    
    # Injecting causal effect:
    # We want a ~15% increase in engagement (time_spent) due to treatment
    # Average base time spent is roughly 15-20. 15% of 20 is ~3 minutes.
    treatment_effect_time = 3.0
    time_spent_mins = base_time_spent + (is_treatment * treatment_effect_time)
    time_spent_mins = np.maximum(time_spent_mins, 1.0) # Ensure positive
    
    # Base retention (1 if retained next month, 0 otherwise)
    # Using a logistic model logic
    logit_base = -0.5 + (past_purchases * 0.2) + (days_since_signup * 0.001)
    
    # Injecting causal effect:
    # We want a ~10% relative increase in retention. 
    # If base retention is ~50%, a 10% relative increase is +5% absolute (55%).
    # In logit terms, an odds ratio corresponding to roughly +5% absolute probability near 0.5.
    treatment_effect_retention = 0.2 
    
    logit_retention = logit_base + (is_treatment * treatment_effect_retention)
    prob_retention = 1 / (1 + np.exp(-logit_retention))
    
    is_retained = np.random.binomial(n=1, p=prob_retention)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'days_since_signup': days_since_signup,
        'past_purchases': past_purchases,
        'is_treatment': is_treatment,
        'time_spent_mins': time_spent_mins,
        'is_retained': is_retained
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Print quick stats to verify the injected effects roughly hold
    print("\nQuick Sanity Check:")
    print("Average Time Spent (Control):", df[df['is_treatment'] == 0]['time_spent_mins'].mean())
    print("Average Time Spent (Treatment):", df[df['is_treatment'] == 1]['time_spent_mins'].mean())
    print("Retention Rate (Control):", df[df['is_treatment'] == 0]['is_retained'].mean())
    print("Retention Rate (Treatment):", df[df['is_treatment'] == 1]['is_retained'].mean())

if __name__ == "__main__":
    generate_synthetic_data()
