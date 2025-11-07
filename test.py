from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast
from utils import read_vocabulary
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


save_path = "my_bpe_tokenizer"

tokenizer = Tokenizer(BPE())
#tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=10_914,  # idx=263749 on 58880
    initial_alphabet=ByteLevel.alphabet(),
    min_frequency=1,
    special_tokens=["<s>", "</s>", "<unk>"]
    )

# Обучающий корпус
texts = [
    "playing played plays play",
    "run runner running runs",
    "talking talked talks",
    "cat cat cat dog dogs",
    "are are are are from from from",
    "the the the do doing go going so so so the now"
    "the with with with",
]

tokenizer.train_from_iterator(texts, trainer)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>"
)

#fast_tokenizer.save_pretrained(my_bpe_tokenizer)


# print("\n✅ Токенизатор сохранён в:", save_path)

# --- 3️⃣ Загружаем обратно ---
tokenizer = PreTrainedTokenizerFast.from_pretrained(save_path)


# 3️⃣ Проверим словарь — должны появиться токены с '##'
vocab = tokenizer.get_vocab()
# for token in sorted(vocab.keys()):
#     if token.startswith("##"):
#         print(token)

# 4️⃣ Проверим, как токенизируется слово
print("\nTokenize word: 'playing':")
print(fast_tokenizer.tokenize("I'll playing so-so now with belingcat"))

print("\nTokenize phrase: 'the cats are running':")
print(fast_tokenizer.tokenize("the cats are running ing"))


# def tokens_to_file(tokenizer, words: list, outpath: str):
#     with open(outpath, "w", encoding="utf-8") as f_out:
#         for w in words:
#             if w.find("-") < 0:
#                 input_ids = tokenizer(w, add_special_tokens=False, padding=False, return_tensors="np")
#                 input_ids = input_ids["input_ids"]
#                 input_ids = input_ids[0]
#                 f_out.write(f"{w}: {str(tokenizer.convert_ids_to_tokens(input_ids))}\n")


# tokens_to_file(
#     tokenizer,
#     sorted(read_vocabulary("data/db-full.txt", count=1)),
#     save_path + "/output.txt")

