from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from datasets import load_dataset
import numpy as np
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import pandas as pd
import re


def str_tokenize_words(s: str, stopwords = set()) -> list:
    words = re.findall("(\.?\w[\w'\.&-]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []

####################################################################
gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
print(gpt2.vocab_size)       # 50257
print(gpt2.eos_token)        # 
print(gpt2.eos_token_id)     # 50256
####################################################################


chunk_df = pd.read_parquet("datasets/eli5/pair/train-00000-of-00001.parquet", columns=["question", "answer"])

dataset = []

for idx, row in chunk_df.iterrows():
    question = row["question"]
    answer = row["answer"]
    if (idx + 1) % 1000 == 0:
        print(f"...items={idx}")
    dataset.append(question + " " + answer)


def train_tokenizer():
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=50_000, min_frequency=2)
    tokenizer.train_from_iterator(iter(dataset), trainer=trainer)

    vocab = tokenizer.get_vocab()
    id_to_token = sorted(vocab.items(), key=lambda x: x[1])
    for token, idx in id_to_token[:20]:
        print(f"{idx:5d} | {repr(token)}")

    tokens = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    #print(tokens[:20])
    return tokenizer


tokenizer = train_tokenizer()


def evaluate_tokenizer(tokenizer, texts):
    n_chars = 0
    n_tokens = 0
    oov = 0
    n_words = 0
    for text in texts:
        enc = tokenizer.encode(text)
        n_chars += len(text)
        n_tokens += len(enc.tokens)
        n_words += len(str_tokenize_words(text))
        # пример оценки OOV: если токен начинается с "##" или "�"
        oov += sum(1 for t in enc.tokens if "�" in t)
    avg_len = n_tokens / len(texts)
    ch_compression = n_chars / n_tokens
    w_compression = n_chars / len(texts) / avg_len  # TODO: check this formula
    return {"avg_tokens_per_text": avg_len,
            "char_compression_ratio": ch_compression,
            "word_compression_ratio": w_compression,
            "oov_rate": oov / n_tokens}


metrics = evaluate_tokenizer(tokenizer, dataset[:1000])
print(metrics)

lengths = [len(t) for t in tokenizer.get_vocab().keys()]
plt.hist(lengths, bins="auto")
plt.title("Distribution of token lengths")
plt.xlabel("Token Length")
plt.show()

##############################################################################

texts = [
    "Mixture-of-Experts with Expert Choice Routing",
    "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.",
    "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.",
    "Do Transformer Modifications Transfer Across Implementations and Applications?",
    "Learning, Factored Representations in a Deep Mixture of Experts.",
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.",
    "ST-MoE: Designing Stable and Transferable Sparse Expert Models.",
]

for t in texts:
    print(t)
    print(" ".join(str_tokenize_words(t)))
    print(tokenizer.encode(t).tokens)
