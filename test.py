from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast


save_path = "my_wordpiece_tokenizer"

# 1️⃣ Создаём токенизатор с WordPiece моделью
# tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
# tokenizer.normalizer = normalizers.NFKC()
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# # 2️⃣ Обучаем на примерах
# trainer = WordPieceTrainer(
#     vocab_size=50,
#     min_frequency=1,
#     special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
# )

# Обучающий корпус
# texts = [
#     "playing played plays play",
#     "run runner running runs",
#     "talking talked talks",
#     "cats cat cat cat dog dogs",
#     "are are are are from from from",
#     "do doing go going"
# ]

# tokenizer.train_from_iterator(texts, trainer)

# fast_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     cls_token="[CLS]",
#     sep_token="[SEP]"
# )
# fast_tokenizer.save_pretrained(save_path)

# print("\n✅ Токенизатор сохранён в:", save_path)

# --- 3️⃣ Загружаем обратно ---
tokenizer = PreTrainedTokenizerFast.from_pretrained(save_path)



# 3️⃣ Проверим словарь — должны появиться токены с '##'
vocab = tokenizer.get_vocab()
for token in sorted(vocab.keys()):
    if token.startswith("##"):
        print(token)

# 4️⃣ Проверим, как токенизируется слово
print("\nTokenize word: 'playing':")
print(tokenizer.tokenize("I'm playing now"))

print("\nTokenize phrase: 'the cats are running':")
print(tokenizer.tokenize("the cats are running"))
