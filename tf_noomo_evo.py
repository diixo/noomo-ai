
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast


filepath = "data/temp.txt"


with open("data/db-full.txt", "r", encoding="utf-8") as f:
    word_set = set([line.strip() for line in f if line.strip()])

count = 0
with open(filepath, "w", encoding="utf-8") as f_out:
    for w in word_set:
        if w.find("-") < 0:
            f_out.write(f"{w} {w}\n")
            count += 1

##########################################################################################
tokenizer_path  = "noomo"

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(), min_frequency=1,
                    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
                    )

tokenizer.train([filepath], trainer)

#tokenizer.add_tokens(list(word_set))

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    pad_token = "<pad>",
    mask_token = "<mask>"
)

fast_tokenizer.save_pretrained(tokenizer_path)

##########################################################################################

tokenizer_gpt = GPT2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)

def statistic(tokenizer_gpt: GPT2TokenizerFast):
    print(f"tokenizer_gpt.config: vocab.sz={len(tokenizer_gpt.get_vocab())},",
        f"pad_token_id={tokenizer_gpt.pad_token_id},",
        f"bos_token_id={tokenizer_gpt.bos_token_id},",
        f"eos_token_id={tokenizer_gpt.eos_token_id}",
        )

##########################################################################################

def print_tokenization(prompt: str):
    input_ids = tokenizer_gpt(prompt, add_special_tokens=False, padding=False, return_tensors="np")
    input_ids = input_ids["input_ids"]
    input_ids = input_ids[0]

    #print(input_ids)
    print(tokenizer_gpt.convert_ids_to_tokens(input_ids))
    print(tokenizer_gpt.decode(input_ids, skip_special_tokens=False))


if __name__ == '__main__':

    statistic(tokenizer_gpt)

    print(f"{len(word_set)}: {count}")

    print_tokenization("Learning learns learned hears is hearing")

    # print_tokenization("Do doing does teach teacher teaching")

    # print_tokenization("Wear wears wearing his this are wanting wanted wants")

    # print_tokenization("All is as was running were will")

    # print_tokenization("Here where there no nope not therefore anywhere still fill ill")

    # print_tokenization(
    #     "be being or so the that this its an should would could may say might fix post pre pro put ation ession too also but and end")

    # print_tokenization("postfix prefix international putting forever somewhere never profession professional")

    # print_tokenization("come become commit comes common cannot can't sooner")

    # print_tokenization("gather gathering gathered together more she because didn't")

    # print_tokenization("Ask ask task mask tasking masking")

    # print_tokenization("you young your yours we were mostly")

    # print_tokenization("us use used uses using usual usually known knows whenever everyday illness seemingly")

    # print_tokenization("densemind")


