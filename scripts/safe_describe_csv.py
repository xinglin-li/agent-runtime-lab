# scripts/safe_describe_csv.py
import sys
import csv
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Error: Missing file path argument.", file=sys.stderr)
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            row_count = sum(1 for _ in reader)
            print(f"Headers: {headers} | Total Rows: {row_count}")
    except Exception as e:
        print(f"Execution Error: {str(e)}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()