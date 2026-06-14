# scripts/safe_series_summary.py
import sys
import csv
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Error: Missing file path argument.", file=sys.stderr)
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Compute the average of the second column, assumed to be sales.
            sales = [float(row['sales']) for row in reader if row.get('sales')]
            if not sales:
                print("No active elements found.", file=sys.stderr)
                sys.exit(1)
            avg_sales = sum(sales) / len(sales)
            print(f"Metrics: Mean={avg_sales:.2f}, Max={max(sales)}, Min={min(sales)}")
    except Exception as e:
        print(f"Execution Error: {str(e)}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
