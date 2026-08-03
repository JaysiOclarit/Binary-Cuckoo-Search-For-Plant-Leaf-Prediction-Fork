import os
import re
import csv
from pathlib import Path

def clean_class_name(raw_class: str) -> str:
    """
    Cleans target Class labels.
    Example: "17Senna alata(SA)" -> "Senna alata"
             "37Leucaena leucocephala(LL)" -> "Leucaena leucocephala"
             "13Momordica charantia (MC)" -> "Momordica charantia"
    """
    if not raw_class:
        return ""
    # Remove leading digits (e.g. "17Senna..." -> "Senna...")
    s = re.sub(r'^\d+', '', raw_class)
    # Remove trailing parenthetical acronyms (e.g. "(SA)" -> "")
    s = re.sub(r'\s*\([^)]+\)\s*$', '', s)
    return s.strip()

def clean_csv_dataset(file_path: Path):
    if not file_path.exists():
        print(f"[SKIP] File not found: {file_path}")
        return

    print(f"[PROCESSING] Cleaning {file_path.name}...")
    
    # Read CSV lines
    with open(file_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print(f"[WARN] Empty CSV file: {file_path}")
            return
        
        rows = list(reader)

    # Find Class column index (usually column 0)
    class_idx = 0
    if 'Class' in header:
        class_idx = header.index('Class')

    cleaned_rows = []
    unique_classes = set()

    for row in rows:
        if not row:
            continue
        raw_label = row[class_idx]
        clean_label = clean_class_name(raw_label)
        row[class_idx] = clean_label
        unique_classes.add(clean_label)
        cleaned_rows.append(row)

    # Overwrite CSV with cleaned rows
    with open(file_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned_rows)

    print(f"[SUCCESS] Cleaned {file_path.name} ({len(cleaned_rows)} rows, {len(unique_classes)} unique classes)")

def main():
    print("=================================================================")
    print("      PYTHON PLANT LEAF DATASET CLASS LABEL CLEANER               ")
    print("=================================================================\n")

    base_dir = Path(__file__).parent
    
    # List of candidate paths to clean
    target_files = [
        # Philippine Datasets
        base_dir / "backend" / "Entire Data Folder" / "Original Dataset" / "Philippine Leaf data.csv",
        base_dir / "backend" / "Entire Data Folder" / "Philippine After FS" / "Philippine After FS.csv",
        base_dir / "backend" / "Entire Data Folder" / "Philippine After GBCS-FS" / "Philippine After GBCS-FS.csv",
        base_dir / "Entire Data Folder" / "Original Dataset" / "Philippine Leaf data.csv",
        base_dir / "Entire Data Folder" / "Philippine After FS" / "Philippine After FS.csv",
        base_dir / "Entire Data Folder" / "Philippine After GBCS-FS" / "Philippine After GBCS-FS.csv",
    ]

    cleaned_count = 0
    for target in target_files:
        if target.exists():
            clean_csv_dataset(target)
            cleaned_count += 1

    print(f"\n=================================================================")
    print(f"Dataset Cleaning Finished! Processed {cleaned_count} CSV files.")
    print("=================================================================")

if __name__ == "__main__":
    main()
