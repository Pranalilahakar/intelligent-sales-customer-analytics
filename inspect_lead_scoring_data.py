
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

FILES = [
    "sales_analysis.csv",
    "customer_segments.csv",
    "customer_churn_analysis.csv",
]


def main():
    for filename in FILES:
        path = DATA_DIR / filename

        print("\n" + "=" * 65)
        print(f"FILE: {filename}")
        print("=" * 65)

        if not path.exists():
            print("STATUS: File not found")
            continue

        df = pd.read_csv(path, nrows=1000, low_memory=False)

        print(f"Rows inspected: {len(df):,}")
        print("Columns:")
        for column in df.columns:
            print(f"  - {column} ({df[column].dtype})")

        print("\nSample records:")
        print(df.head(3).to_string(index=False))

        possible_labels = [
            column
            for column in df.columns
            if any(
                keyword in column.lower()
                for keyword in [
                    "converted",
                    "conversion",
                    "lead_status",
                    "target",
                    "outcome",
                    "won",
                    "response",
                ]
            )
        ]

        print("\nPotential outcome/label columns:")
        if possible_labels:
            for column in possible_labels:
                print(f"  - {column}")
        else:
            print("  No obvious conversion-label column found.")

    print("\nInspection complete.")
    print(
        "Note: Column names alone cannot establish "
        "whether a valid prediction target exists."
    )


if __name__ == "__main__":
    main()
