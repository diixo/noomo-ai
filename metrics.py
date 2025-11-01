from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from datasets import load_dataset
import numpy as np
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt


gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
print(gpt2.vocab_size)       # 50257
print(gpt2.eos_token)        # 
print(gpt2.eos_token_id)     # 50256
####################################################################

# Загружаем текстовый корпус
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Обучаем токенизатор
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=50000, min_frequency=2)
tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

# Проверяем на корпусе
def evaluate_tokenizer(tokenizer, texts):
    n_chars = 0
    n_tokens = 0
    oov = 0
    for text in texts:
        enc = tokenizer.encode(text)
        n_chars += len(text)
        n_tokens += len(enc.tokens)
        # пример оценки OOV: если токен начинается с "##" или "�"
        oov += sum(1 for t in enc.tokens if "�" in t)
    avg_len = n_tokens / len(texts)
    compression = n_chars / n_tokens
    return {"avg_tokens_per_text": avg_len,
            "compression_ratio": compression,
            "oov_rate": oov / n_tokens}


metrics = evaluate_tokenizer(tokenizer, dataset["text"][:1000])
print(metrics)

lengths = [len(t) for t in tokenizer.get_vocab().keys()]
plt.hist(lengths, bins=50)
plt.title("Distribution of token lengths")
plt.show()
