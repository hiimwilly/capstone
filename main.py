"""
main.py
=======
Minimal demonstration of the preprocessing pipeline for the capstone dataset.
"""

import os
import sys

# Ensure src/ is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_data, preprocess_dataset  # noqa: E402


def main() -> None:
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    result = preprocess_dataset(df)

    print("Preprocessing complete.")
    print(f"  X_train shape : {result['X_train'].shape}")
    print(f"  X_test  shape : {result['X_test'].shape}")
    print(f"  y_train sample: {result['y_train'][:5]}")
    print(f"  y_test  sample: {result['y_test'][:5]}")


if __name__ == "__main__":
    main()
