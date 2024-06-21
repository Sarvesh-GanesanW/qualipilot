import pandas as pd
from datapilot import DataQualityChecker

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward', 'Frank', 'Grace'],
    'Age': [25, 32, 37, 29, 41, None, 30],
    'Salary': [50000, 54000, 58000, 62000, 66000, 70000, 74000],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'IT', 'Finance', 'HR']
}
df = pd.DataFrame(data)

# Initialize the checker
checker = DataQualityChecker(df)

# Run all checks
column_ranges = {
    'Age': (18, 65),
    'Salary': (30000, 80000)
}
check_results = checker.run_all_checks(column_ranges)

# Display results
for check, result in check_results.items():
    print(f"--- {check} ---")
    print(result)
    print("\n")
