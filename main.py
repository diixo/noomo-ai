
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from diixo import diixo
import re


outpath = "data/output.txt"


with open("data/db-full.txt", "r", encoding="utf-8") as f:
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


def tokens_to_file(words: list, outpath: str = outpath):
    with open(outpath, "w", encoding="utf-8") as f_out:
        for w in words:
            if w.find("-") < 0:
                input_ids = tokenizer_gpt(w, add_special_tokens=False, padding=False, return_tensors="np")
                input_ids = input_ids["input_ids"]
                input_ids = input_ids[0]

                f_out.write(f"{w}: {str(tokenizer_gpt.convert_ids_to_tokens(input_ids))}\n")


def evaluate_to_file(words: list, outpath: str = outpath):
    import tiktoken
    import json

    model_name = "gpt-4o-mini"  # o200k_base
    model_name = "gpt-4o"       # o200k_base
    model_name = "gpt-4"        # cl100k_base

    # tiktoken.get_encoding("o200k_base")     # GPT-4o / GPT-4o-mini
    # tiktoken.get_encoding("cl100k_base")    # GPT-3.5 / GPT-4
    # tiktoken.get_encoding("p50k_base")      # Codex / old GPT-3
    # tiktoken.get_encoding("r50k_base")      # Old GPT-3

    enc = tiktoken.encoding_for_model(model_name)

    with open(outpath, "w", encoding="utf-8") as f_out:
        for w in words:
            if "-" not in w:  # skip words with hyphen 
                token_ids = enc.encode(w)
                # decode ID into string tokens (string subwords)
                tokens = [enc.decode([tid]) for tid in token_ids]

                f_out.write(f"{w}: {tokens}\n")


    # Get the entire dictionary: {token_id: byte_string}
    vocab = enc._mergeable_ranks

    latin_tokens = []

    # Regular: latin letters with space at the beginning (the same for GPT-tokens, like ' hello')
    pattern = re.compile(r"^[A-Za-z' ]+$")

    for token_bytes, token_id in vocab.items():
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue  # skip byte fallback-tokens

        if pattern.match(token_str):
            #latin_tokens.append((token_id, token_str))
            latin_tokens.append({
                "id": token_id,
                "token": token_str
            })

    with open("latin_tokens.json", "w", encoding="utf-8") as f:
        json.dump(latin_tokens, f, ensure_ascii=False, indent=2)


    print(f"Total latin tokens: {len(latin_tokens)}")

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

    tokens_to_file(word_set)
    #evaluate_to_file(word_set)

