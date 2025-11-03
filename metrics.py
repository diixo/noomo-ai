from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import pandas as pd
import json
from utils import str_tokenize_words, clean_text, read_vocabulary


VOCAB_SZ = 12_032

stopwords = set(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                 "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                 "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                 ".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "[", "]", "{",
                 "}", "-", "_", "+", "=", "*", "&", "^", "%", "$", "#", "@", "~", "`",])


####################################################################
gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
print(gpt2.vocab_size)       # 50257
print(gpt2.eos_token)        # 
print(gpt2.eos_token_id)     # 50256
####################################################################

def read_eli5():
    dataset = []

    chunk_df = pd.read_parquet("datasets/eli5/pair/train-00000-of-00001.parquet", columns=["question", "answer"])

    for idx, row in chunk_df.iterrows():
        question = row["question"]
        answer = row["answer"]
        if (idx + 1) % 1000 == 0:
            print(f"...items={idx}")
        dataset.append(clean_text(question + " " + answer))
    return dataset


def train_tokenizer(dataset: list):
    if True:
        tokenizer = Tokenizer(WordPiece())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            vocab_size=VOCAB_SZ,
            min_frequency=1,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            )
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(vocab_size=VOCAB_SZ, min_frequency=2)

    tokenizer.train_from_iterator(iter(dataset), trainer=trainer)

    vocab = tokenizer.get_vocab()

    latin_tokens = []
    for idx, token in enumerate(list(vocab.keys())):
        latin_tokens.append({"id": idx, "token": token})

    with open("latin-tokens.json", "w", encoding="utf-8") as f:
        json.dump(latin_tokens, f, ensure_ascii=False, indent=2)

    return tokenizer

################################################################

def evaluate_tokenizer(tokenizer, texts):
    n_tokens = 0
    n_words = 0

    for text in texts:
        words = str_tokenize_words(text, stopwords)
        n_words += len(words)

        text = " ".join(words)

        enc = tokenizer.encode(text)
        n_tokens += len(enc.tokens)
        # enc = tokenizer(text, add_special_tokens=False)
        # n_tokens += len(enc["input_ids"])

    w_compression = n_tokens / n_words
    return { "word_compression_ratio": w_compression }

################################################################

dataset = list(read_vocabulary("data/db-full-58816.txt"))
dataset += read_eli5()

print(32 * "#")

tokenizer = train_tokenizer(dataset)

#metrics = evaluate_tokenizer(tokenizer, dataset[:1000])
metrics = evaluate_tokenizer(tokenizer, dataset)
print(metrics)

lengths = [len(t.lstrip("#")) for t in tokenizer.get_vocab().keys()]
plt.hist(lengths, bins="auto")
plt.title("Distribution of token lengths")
plt.xlabel("Token Length")
plt.ylabel("Count")
plt.show()

################################################################

texts = [
    "Mixture-of-Experts with Expert Choice Routing",
    "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.",
    "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.",
    "Do Transformer Modifications Transfer Across Implementations and Applications?",
    "Learning, Factored Representations in a Deep Mixture of Experts.",
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.",
    "ST-MoE: Designing Stable and Transferable Sparse Expert Models.",
    "was were has have had do does did go gone goes went see saw sees seen seeing run runs ran",
]

for t in texts:
    txt = " ".join(str_tokenize_words(t))
    print(tokenizer.encode(txt).tokens)
