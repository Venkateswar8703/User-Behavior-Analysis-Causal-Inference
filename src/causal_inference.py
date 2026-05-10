import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

def perform_causal_inference(data_path='data/raw/user_data.csv'):
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run data_generation.py first.")
        return

    print("Loading data for Causal Inference & A/B Testing Analysis...")
    df = pd.read_csv(data_path)
    
    print("\n" + "="*50)
    print("PART 1: A/B Testing on Engagement (Linear Regression / OLS)")
    print("="*50)
    # We want to estimate the causal effect of 'is_treatment' on 'time_spent_mins'
    # We control for age, days_since_signup, and past_purchases to reduce variance
    # and adjust for any slight imbalances (though RCT should balance them on average).
    
    ols_model = smf.ols('time_spent_mins ~ is_treatment + age + days_since_signup + past_purchases', data=df)
    ols_results = ols_model.fit()
    
    print(ols_results.summary().tables[1])
    
    treatment_effect_time = ols_results.params['is_treatment']
    control_mean_time = df[df['is_treatment'] == 0]['time_spent_mins'].mean()
    relative_lift_time = (treatment_effect_time / control_mean_time) * 100
    
    print(f"\nInsight: The treatment caused an absolute increase of ~{treatment_effect_time:.2f} minutes in engagement.")
    print(f"Given the control group's average of {control_mean_time:.2f} mins, this is a ~{relative_lift_time:.2f}% relative increase in engagement.")

    print("\n" + "="*50)
    print("PART 2: Causal Inference on Retention (Logistic Regression)")
    print("="*50)
    
    # We want to estimate the causal effect of 'is_treatment' on 'is_retained'
    # Using Logistic Regression because retention is binary.
    
    logit_model = smf.logit('is_retained ~ is_treatment + age + days_since_signup + past_purchases', data=df)
    logit_results = logit_model.fit(disp=0) # disp=0 hides optimization output
    
    print(logit_results.summary().tables[1])
    
    # Calculate marginal effects to get absolute probability change
    marginal_effects = logit_results.get_margeff()
    # The first element in summary is typically the intercept, wait, get_margeff ignores intercept.
    # We can just look at the average predicted probabilities.
    
    # Predict for all users AS IF they were in control
    df_control_sim = df.copy()
    df_control_sim['is_treatment'] = 0
    prob_control = logit_results.predict(df_control_sim).mean()
    
    # Predict for all users AS IF they were in treatment
    df_treatment_sim = df.copy()
    df_treatment_sim['is_treatment'] = 1
    prob_treatment = logit_results.predict(df_treatment_sim).mean()
    
    absolute_lift_retention = prob_treatment - prob_control
    relative_lift_retention = (absolute_lift_retention / prob_control) * 100
    
    print(f"\nInsight: The expected retention rate without treatment is ~{prob_control*100:.2f}%.")
    print(f"The expected retention rate with treatment is ~{prob_treatment*100:.2f}%.")
    print(f"This represents an absolute lift of ~{absolute_lift_retention*100:.2f}% and a relative retention improvement of ~{relative_lift_retention:.2f}%.")

if __name__ == "__main__":
    perform_causal_inference()
