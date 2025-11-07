from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from utils import read_eli5, read_jsonl, read_vocabulary
from pathlib import Path
from main import tokens_to_file


save_path = "my_bpe_tokenizer"


tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=11200, # 10_848 + 256 + 3 + 29,  # 11136 idx=140169 on 58880
    initial_alphabet=ByteLevel.alphabet(),
    min_frequency=1,
    special_tokens=["<s>", "</s>", "<unk>"]
    )

# Test training corpus:
# texts = [
#     "playing played plays play",
#     "run runner running runs",
#     "talking talked talks",
#     "cat cat cat dog dogs",
#     "are are are are from from from",
#     "the the the do doing go going so so so the now"
#     "the with with with",
# ]

word_list = []
with Path("data/db-full-58880.txt").open("r", encoding="utf-8") as f:
    word_list = [line.strip() for line in f if line.strip()]

dataset = 8 * word_list
# dataset += read_eli5()
# print(f"1.) sz={len(dataset)}")

# dataset += read_jsonl("datasets/arxiv-corpus/arxiv_cs_2015_2020.jsonl")
# dataset += read_jsonl("datasets/arxiv-corpus/arxiv_cs_2021_2024.jsonl")
# print(f"2.) sz={len(dataset)}")


tokenizer.train_from_iterator(dataset, trainer)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>"
)

fast_tokenizer.save_pretrained(save_path)


# print("\n✅ Токенизатор сохранён в:", save_path)

# --- 3️⃣ Загружаем обратно ---
tokenizer = PreTrainedTokenizerFast.from_pretrained(save_path)


# 3️⃣ Проверим словарь — должны появиться токены с '##'
vocab = tokenizer.get_vocab()
# for token in sorted(vocab.keys()):
#     if token.startswith("##"):
#         print(token)

# 4️⃣ Проверим, как токенизируется слово
# print("\nTokenize word: 'playing':")
print(tokenizer.tokenize("I'll playing so-so now with belingcat"))

# print("\nTokenize phrase: 'the cats are running':")
# print(tokenizer.tokenize("the cats are running ing"))


def calc_tokens(tokenizer, words: list):
    ids_count = 0
    vocab = set()
    for w in words:
        input_ids = tokenizer(w, add_special_tokens=False, padding=False, return_tensors="np")
        input_ids = input_ids["input_ids"]
        input_ids = input_ids[0]
        ids_count += len(input_ids)
        vocab.update(tokenizer.convert_ids_to_tokens(input_ids))
        #tokens = tokenizer.convert_ids_to_tokens(input_ids)
        #vocab.update([t for t in tokens if len(t.lstrip('Ġ')) > 1])

    vocab = sorted(vocab)
    print(f"word_compression_ratio: {ids_count/len(words):6f} (idx={ids_count}, words={len(words)}), vocab.sz={len(vocab)}, alphabet={len(ByteLevel.alphabet())}")


word_set = sorted(read_vocabulary("data/db-full-58880.txt", count=1))
calc_tokens(tokenizer, word_set)


# outpath = "data/output-cased.txt"
# SLICE = 26079
# tokens_to_file(tokenizer, word_set[SLICE:], outpath)
