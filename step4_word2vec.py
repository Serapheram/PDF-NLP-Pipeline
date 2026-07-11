import csv
import sys
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

csv.field_size_limit(sys.maxsize)

INPUT_CSV = "dataset_cleaned.csv"


def train_models(sentences, label):
    """Train CBOW (sg=0) and Skip-gram (sg=1) on a list of tokenized sentences."""
    cbow = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=0, epochs=100)
    skipgram = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=1, epochs=100)
    print(f"Trained CBOW and Skip-gram on {label} corpus "
          f"(vocab size: {len(cbow.wv)})")
    return cbow, skipgram


def safe_most_similar(model, word, topn=5):
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    return []


def compare_and_plot(cbow, skipgram, test_words, ngram_label):
    """Print comparison table and plot similarity scores for one test word."""
    print(f"\n--- CBOW vs Skip-gram comparison ({ngram_label}) ---")
    for word in test_words:
        cbow_sim = safe_most_similar(cbow, word)
        sg_sim = safe_most_similar(skipgram, word)
        print(f"\nInput word: '{word}'")
        print(f"  CBOW predictions     : {cbow_sim}")
        print(f"  Skip-gram predictions: {sg_sim}")

    # Plot for the first valid test word
    for word in test_words:
        cbow_sim = safe_most_similar(cbow, word)
        sg_sim = safe_most_similar(skipgram, word)
        if cbow_sim and sg_sim:
            words_c = [w for w, s in cbow_sim]
            scores_c = [s for w, s in cbow_sim]
            words_s = [w for w, s in sg_sim]
            scores_s = [s for w, s in sg_sim]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].barh(words_c[::-1], scores_c[::-1], color="darkorange")
            axes[0].set_title(f"CBOW - similar to '{word}' ({ngram_label})")
            axes[1].barh(words_s[::-1], scores_s[::-1], color="seagreen")
            axes[1].set_title(f"Skip-gram - similar to '{word}' ({ngram_label})")
            plt.tight_layout()
            fname = f"graph_cbow_vs_skipgram_{ngram_label}.png"
            plt.savefig(fname, dpi=120)
            plt.close()
            print(f"Saved graph: {fname}")
            break


if __name__ == "__main__":
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    unigram_sentences = [row["cleaned_text"].split() for row in rows]

    def make_ngram_sentences(n):
        sentences = []
        for row in rows:
            tokens = row["cleaned_text"].split()
            grams = ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            if grams:
                sentences.append(grams)
        return sentences

    bigram_sentences = make_ngram_sentences(2)
    trigram_sentences = make_ngram_sentences(3)

    # Pick a few frequent words from the corpus itself as test examples and change these based on your actual corpus vocabulary
    all_words = [w for sent in unigram_sentences for w in sent]
    from collections import Counter
    common_words = [w for w, c in Counter(all_words).most_common(10)]
    test_words = common_words[:3]
    print("Using test words from corpus:", test_words)

    # ---- Unigram models ----
    cbow_u, sg_u = train_models(unigram_sentences, "unigram")
    compare_and_plot(cbow_u, sg_u, test_words, "unigram")

    # ---- Bigram models ----
    cbow_b, sg_b = train_models(bigram_sentences, "bigram")
    bigram_test = [s.split()[0] for s in [bigram_sentences[0][0]] if bigram_sentences]
    compare_and_plot(cbow_b, sg_b, bigram_test, "bigram")

    # ---- Trigram models ----
    cbow_t, sg_t = train_models(trigram_sentences, "trigram")
    trigram_test = [trigram_sentences[0][0]] if trigram_sentences else []
    compare_and_plot(cbow_t, sg_t, trigram_test, "trigram")

    print("\nDone. Models trained and comparison graphs saved.")
