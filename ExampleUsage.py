import pandas as pd
from datapilot import DataQualityChecker

# Load the sales data
file_path = 'bitcoin.csv'
df = pd.read_csv(file_path)

print(df.shape)

# Initialize the checker with custom parameters
checker = DataQualityChecker(df, npartitions=2, threshold=1.5, llm_api_key='ollama')

# Define column ranges based on the data description
column_ranges = {
    'Open': (4.39, 39301.0),  
    'Close': (4.4, 50638.48)  
}

# Run all checks
print("\nRunning all checks...")
results = checker.run_all_checks(column_ranges=column_ranges, llm_model='qwen2:7b')
results_file_path = 'results.json'
print(f"\nSaving results to {results_file_path}...")
checker.save_results(results, results_file_path)

print("\nData quality checks completed and results saved.")
