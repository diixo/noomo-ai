from transformers import AutoTokenizer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from diixo import diixo, diixo_2
from utils import gpt_evaluate_to_file, tokens_to_file


outpath = "data/output-cased.txt"

outpath_gpt2 = "data/output-gpt2.txt"

SLICE = 26086

##########################################################################################
with open("data/db-full-58900.txt", "r", encoding="utf-8") as f:
    word_set = set([line.strip() for line in f if line.strip()])

word_set = sorted(word_set)

##########################################################################################

gpt2 = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

vocab = tokens_to_file(gpt2, word_set[SLICE:], outpath_gpt2)    # idx=77638

vocab = sorted(vocab)

##########################################################################################
tokenizer_path  = "noomo"

tokenizer = Tokenizer(BPE())
#tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=50000,
    initial_alphabet=ByteLevel.alphabet(),
    min_frequency=1,
    special_tokens=["<s>", "</s>", "<unk>"]
    )

tokenizer.train([], trainer)

tokenizer.add_tokens(vocab)

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


##########################################################################################

if __name__ == '__main__':

    statistic(tokenizer_gpt)

    tokens_to_file(tokenizer_gpt, word_set[SLICE:], outpath)    # idx=77884
    #gpt_evaluate_to_file(word_set, outpath)

