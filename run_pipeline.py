import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"🚀 Running {script_path}...")
    print(f"{'='*50}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_path}. Pipeline stopped.")
        sys.exit(1)

def main():
    print("Starting User Behavior Analysis & Causal Inference Pipeline...")
    
    scripts = [
        "src/data_generation.py",
        "src/sql_analysis.py",
        "src/clustering_analysis.py",
        "src/causal_inference.py"
    ]
    
    # Ensure we are in the correct directory (project root)
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the project root directory.")
        sys.exit(1)
        
    for script in scripts:
        run_script(script)
        
    print(f"\n{'='*50}")
    print("✅ Pipeline execution completed successfully!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
