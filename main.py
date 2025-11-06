
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from diixo import diixo
from utils import gpt_evaluate_to_file


outpath = "data/output-cased.txt"


with open("data/db-full-58880.txt", "r", encoding="utf-8") as f:
    word_set = set([line.strip() for line in f if line.strip()])

word_set = sorted(word_set)
count = len(word_set)

##########################################################################################
tokenizer_path  = "noomo"

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=50000,
    initial_alphabet=ByteLevel.alphabet(),
    min_frequency=1,
    special_tokens=["<s>", "</s>", "<unk>"]
    )

tokenizer.train([], trainer)

tokenizer.add_tokens(diixo)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>"
)

fast_tokenizer.save_pretrained(tokenizer_path)

##########################################################################################

tokenizer_gpt = GPT2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)

def statistic(tokenizer_gpt: GPT2TokenizerFast):
    print(f"tokenizer_gpt.config: vocab.sz={len(tokenizer_gpt.get_vocab())},",
        f"pad_token_id={tokenizer_gpt.pad_token_id},",
        f"bos_token_id={tokenizer_gpt.bos_token_id},",
        f"eos_token_id={tokenizer_gpt.eos_token_id}",)


def tokens_to_file(tokenizer, words: list, outpath: str):
    with open(outpath, "w", encoding="utf-8") as f_out:
        for w in words:
            if w.find("-") < 0:
                input_ids = tokenizer(w, add_special_tokens=False, padding=False, return_tensors="np")
                input_ids = input_ids["input_ids"]
                input_ids = input_ids[0]

                f_out.write(f"{w}: {str(tokenizer.convert_ids_to_tokens(input_ids))}\n")


##########################################################################################

def print_tokenization(prompt: str):

    input_ids = tokenizer_gpt(prompt, add_special_tokens=False, padding=False, return_tensors="np")
    input_ids = input_ids["input_ids"]
    input_ids = input_ids[0]

    print(tokenizer_gpt.convert_ids_to_tokens(input_ids))
    print(tokenizer_gpt.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True)
    )


if __name__ == '__main__':

    statistic(tokenizer_gpt)

    print(f"{len(word_set)}: {count}")

    # print_tokenization("Learning learning learns learned hears is hearing")

    # print_tokenization("Do doing does teach teacher teaching")

    # #print_tokenization("Wear wear wears wearing his this are wanting wanted wants")

    # print_tokenization("Here where there no nope not therefore anywhere still fill ill")

    # print_tokenization("postfix prefix international putting forever somewhere never profession professional")

    # print_tokenization("come become commit comes common cannot can't sooner")

    # print_tokenization("Ask ask task mask tasking masking")

    # print_tokenization("you you young your yours we were mostly")

    # print_tokenization("us us use used uses using usual usually known knows whenever everyday illness seemingly")

    # # print_tokenization("densemind")

    # print_tokenization("gather gathering gathered together more she because didn't")

    # print_tokenization("All is as was running were will")

    # print_tokenization(
    #     "be being or so the that this its an should would could may say might fix post pre pro put ation ession too also but and end extension recode")


    tokens_to_file(tokenizer_gpt, word_set, outpath)
    #gpt_evaluate_to_file(word_set, outpath)

