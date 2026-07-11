import os
import re
import csv
import fitz  # PyMuPDF


CORPUS_DIR = "."  # <-- point this to the folder containing your category-wise PDF folders
OUTPUT_CSV = "dataset_raw.csv"


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file using PyMuPDF (fitz)."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"  [WARNING] Failed to read {pdf_path}: {e}")
    return text


def build_dataset(corpus_dir):
    rows = []
    categories = [d for d in os.listdir(corpus_dir)
                  if os.path.isdir(os.path.join(corpus_dir, d))]

    print(f"Found categories: {categories}")

    for category in categories:
        folder_path = os.path.join(corpus_dir, category)
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        print(f"  {category}: {len(pdf_files)} PDF file(s)")

        for file_name in pdf_files:
            pdf_path = os.path.join(folder_path, file_name)
            raw_text = extract_text_from_pdf(pdf_path)
            raw_text = raw_text.strip()

            if len(raw_text) < 10:
                print(f"  [SKIPPED] {file_name} - little/no extractable text (likely scanned image)")
                continue

            rows.append({
                "file_name": file_name,
                "category": category,
                "raw_text": raw_text
            })

    return rows


if __name__ == "__main__":
    rows = build_dataset(CORPUS_DIR)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "category", "raw_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} documents to {OUTPUT_CSV}")
