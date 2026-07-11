import csv
import sys
from collections import Counter
import matplotlib.pyplot as plt
csv.field_size_limit(sys.maxsize)

INPUT_CSV = "dataset_cleaned.csv"


def get_ngrams(tokens, n):
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def plot_top_freq(counter, title, filename, top_n=15):
    common = counter.most_common(top_n)
    labels = [c[0] for c in common]
    values = [c[1] for c in common]

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1], color="steelblue")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved graph: {filename}")


if __name__ == "__main__":
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    all_tokens = []
    unigram_docs, bigram_docs, trigram_docs = [], [], []

    for row in rows:
        tokens = row["cleaned_text"].split()
        all_tokens.extend(tokens)

        unigrams = get_ngrams(tokens, 1)
        bigrams = get_ngrams(tokens, 2)
        trigrams = get_ngrams(tokens, 3)

        unigram_docs.append(" ".join(unigrams))
        bigram_docs.append(" ".join(bigrams))
        trigram_docs.append(" ".join(trigrams))

    # Save n-gram versions of the corpus
    with open("dataset_ngrams.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_name", "category", "unigrams", "bigrams", "trigrams"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, u, b, t in zip(rows, unigram_docs, bigram_docs, trigram_docs):
            writer.writerow({
                "file_name": row["file_name"],
                "category": row["category"],
                "unigrams": u,
                "bigrams": b,
                "trigrams": t
            })
    print("Saved dataset_ngrams.csv")

    # Frequency counters
    unigram_counter = Counter(all_tokens)
    bigram_counter = Counter(" ".join(bigram_docs).split())
    trigram_counter = Counter(" ".join(trigram_docs).split())

    # Required graphs
    plot_top_freq(unigram_counter, "Top-Word Frequency (After Preprocessing)", "graph_top_words.png")
    plot_top_freq(unigram_counter, "Unigram Frequency", "graph_unigram_freq.png")
    plot_top_freq(bigram_counter, "Bigram Frequency", "graph_bigram_freq.png")
    plot_top_freq(trigram_counter, "Trigram Frequency", "graph_trigram_freq.png")

    print("\nTop 10 unigrams:", unigram_counter.most_common(10))
