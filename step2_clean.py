import re
import csv
import sys
import nltk
from nltk.corpus import stopwords
csv.field_size_limit(sys.maxsize)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

INPUT_CSV = "dataset_raw.csv"
OUTPUT_CSV = "dataset_cleaned.csv"

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """Full cleaning pipeline using regular expressions."""
    text = text.lower()                                       # lowercase
    text = re.sub(r"http\S+|www\.\S+", " ", text)              # remove URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)                  # remove emails
    text = re.sub(r"\d+", " ", text)                           # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)                       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                   # normalize spaces

    tokens = text.split()                                      # tokenization
    tokens = [t for t in tokens if t not in STOP_WORDS]         # stop-word removal
    tokens = [t for t in tokens if len(t) > 2]                  # remove very short tokens

    return tokens


if __name__ == "__main__":
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = clean_text(row["raw_text"])
            cleaned_text = " ".join(tokens)
            row["cleaned_text"] = cleaned_text
            rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_name", "category", "raw_text", "cleaned_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cleaned {len(rows)} documents -> saved to {OUTPUT_CSV}")
    print("\nExample cleaned text (first doc):")
    print(rows[0]["cleaned_text"][:200])
