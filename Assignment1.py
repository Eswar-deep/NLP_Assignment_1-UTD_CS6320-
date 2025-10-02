import math
import argparse

def preprocess_file(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.lower().strip().split()
            tokens = ['<s>'] + tokens + ['</s>']
            lines.append(tokens)
    return lines

def build_vocabulary(lines, unk_threshold):
    word_counts = {}
    total_words = 0
    for tokens in lines:
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
            total_words += 1

    vocabulary = set()
    for word, count in word_counts.items():
        if count > unk_threshold:
            vocabulary.add(word)

    # Always include specials
    vocabulary.update({'<s>', '</s>', '<UNK>'})
    return vocabulary, total_words  # second value not used downstream

def replace_with_unk(lines, vocab):
    unk_lines = []
    for toks in lines:
        mapped = [w if w in vocab else '<UNK>' for w in toks]
        unk_lines.append(mapped)
    return unk_lines

class Unigram:
    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)
        self.vocabulary_size = len(self.vocabulary)
        self.unigram_counts = {}
        self.total_words = 0

    def fit(self, lines):
        # Count tokens; assume lines are already UNK-mapped
        for sent in lines:
            for tok in sent:
                self.unigram_counts[tok] = self.unigram_counts.get(tok, 0) + 1
                self.total_words += 1

    # MLE p(w) = c(w) / N
    def prob(self, token):
        c = self.unigram_counts.get(token, 0)
        if self.total_words == 0:
            return 0.0
        return c / self.total_words

    # Add-k smoothing: (c(w)+k) / (N + k*V)
    def prob_addk(self, token, k=1.0):
        c = self.unigram_counts.get(token, 0)
        N = self.total_words
        V = self.vocabulary_size
        return (c + k) / (N + k * V) 

    def log_probability(self, sent_tokens, smoothing=None, k=1.0):
        # sent_tokens already contain SOS/EOS if you included them
        logp = 0.0
        for w in sent_tokens:
            if smoothing is None:
                p = self.prob(w)
            elif smoothing == "addk":
                p = self.prob_addk(w, k=k)
            else:
                raise ValueError("Unknown smoothing")
            logp += math.log(p)
        return logp
    
class Bigram:
    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)
        self.V = len(self.vocabulary)
        # bigram_counts[(w_{i-1}, w_i)] = c
        self.bigram_counts = {}
        # context_counts[w_{i-1}] = sum over w_i c(w_{i-1}, w_i)
        self.context_counts = {}
        # Optional: unigram counts of tokens (helpful for checks)
        self.unigram_counts = {}

    def fit(self, lines):
        """
        Count bigrams on UNK-mapped data.
        Assumes each sentence starts with '<s>' and ends with '</s>'.
        """
        for sent in lines:
            # Count unigrams (for reference / sanity)
            for tok in sent:
                self.unigram_counts[tok] = self.unigram_counts.get(tok, 0) + 1

            # Count bigrams
            for prev, cur in zip(sent[:-1], sent[1:]):
                self.bigram_counts[(prev, cur)] = self.bigram_counts.get((prev, cur), 0) + 1
                self.context_counts[prev] = self.context_counts.get(prev, 0) + 1

    # MLE: P(w_i | w_{i-1}) = c(w_{i-1}, w_i) / c(w_{i-1})
    def prob(self, prev, cur):
        c_prev = self.context_counts.get(prev, 0)
        c_bigram = self.bigram_counts.get((prev, cur), 0)
        return c_bigram / c_prev

    # Add-k smoothing: (c(prev,cur)+k) / (c(prev)+k*V)
    def prob_addk(self, prev, cur, k=1.0):
        c_prev = self.context_counts.get(prev, 0)
        c_bigram = self.bigram_counts.get((prev, cur), 0)
        return (c_bigram + k) / (c_prev + k * self.V) 

    def log_probability(self, sent_tokens, smoothing=None, k=1.0, backoff_unigram=None, alpha=0.4):
        logp = 0.0
        for prev_w, w in zip(sent_tokens[:-1], sent_tokens[1:]):
            # sanity checks (remove if you want max speed)
            # assert prev_w in self.vocabulary, f"OOV prev token: {prev_w}"
            # assert w in self.vocabulary,      f"OOV token: {w}"

            if smoothing is None:
                p = self.prob(prev_w, w)
                if p == 0.0:
                    if backoff_unigram is not None:
                        pu = backoff_unigram.prob(w)
                        p = alpha * pu if pu > 0.0 else (1.0 / max(1, backoff_unigram.vocabulary_size))
                    else:
                        # return float('-inf')  # strict MLE
                        p = 1.0 / max(1, self.V)
            elif smoothing == "addk":
                p = self.prob_addk(prev_w, w, k=k)
            else:
                raise ValueError("Unknown smoothing")

            if p <= 0.0:
                return float('-inf')
            logp += math.log(p)

        return logp

def corpus_perplexity_unigram(model, lines, smoothing=None, k=1.0):
    total_logp = 0.0
    token_count = 0
    for sent in lines:
        total_logp += model.log_probability(sent, smoothing=smoothing, k=k)
        token_count += len(sent)
    if token_count == 0:
        return float('inf')
    avg_neg_logp = - total_logp / token_count
    return math.exp(avg_neg_logp)

def corpus_perplexity_bigram(model, lines, smoothing=None, k=1.0, backoff_unigram=None):
    total_logp = 0.0
    trans_count = 0
    for sent in lines:
        total_logp += model.log_probability(sent, smoothing=smoothing, k=k,backoff_unigram=backoff_unigram)
        # number of bigram transitions in this sentence
        trans = max(1, len(sent) - 1)
        trans_count += trans
    if trans_count == 0:
        return float('inf')
    avg_neg_logp = - total_logp / trans_count
    return math.exp(avg_neg_logp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.txt")
    ap.add_argument("--valid", required=True, help="Path to validation.txt")
    ap.add_argument("--unk_threshold", type=int, default=1,
                    help="Freq <= threshold becomes <UNK> in train")
    ap.add_argument("--smoothing", type=str, default=None,
                    choices=[None, "addk"], help="Smoothing method (None or addk)")
    ap.add_argument("--k", type=float, default=1.0,
                    help="k value for add-k smoothing")
    ap.add_argument("--backoff", action="store_true",
                    help="Use unigram backoff for bigram MLE")
    args = ap.parse_args()

    # 1) Load files
    train_lines = preprocess_file(args.train)
    val_lines = preprocess_file(args.valid)

    # 2) Build vocab + UNK replacement
    vocab, _ = build_vocabulary(train_lines, unk_threshold=args.unk_threshold)
    train_unk = replace_with_unk(train_lines, vocab)
    valid_unk = replace_with_unk(val_lines, vocab)

    # 3) Train models
    uni = Unigram(vocab)
    uni.fit(train_unk)

    bi = Bigram(vocab)
    bi.fit(train_unk)

    # 4) Perplexities
    print("=== Settings ===")
    print(f"UNK threshold (<=): {args.unk_threshold}")
    print(f"Vocab size (incl. specials): {len(vocab)}")
    print(f"Smoothing: {args.smoothing}, k={args.k}")
    print(f"Backoff enabled: {args.backoff}\n")

    pp_uni = corpus_perplexity_unigram(uni, valid_unk, smoothing=args.smoothing, k=args.k)
    pp_bi = corpus_perplexity_bigram(bi, valid_unk, smoothing=args.smoothing, k=args.k,backoff_unigram=uni if args.backoff else None)

    print("=== Validation Perplexities ===")
    print(f"Unigram ({args.smoothing or 'MLE'}): {pp_uni:.3f}")
    print(f"Bigram ({args.smoothing or 'MLE'}" f"{' + backoff' if args.backoff else ''}): {pp_bi:.3f}")

if name == "__main__":
    main()