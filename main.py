from transformers import AutoTokenizer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from utils import tokens_to_file


outpath = "data/output-cased.txt"

outpath_gpt2 = "data/output-qwen3.txt"


##########################################################################################
def read_vocab(add_prefix_space=False, count=59968):

    prefix = " " if add_prefix_space==True else ""
    with open(f"data/db-full-{count}.txt", "r", encoding="utf-8") as f:
        word_set = set([prefix + line.strip() for line in f if line.strip()])
    return word_set


word_set = sorted(read_vocab(True)) + sorted(read_vocab(False))

##########################################################################################

tokenizer_path  = "noomo"

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=50_257,
    initial_alphabet=ByteLevel.alphabet(),
    min_frequency=1,
    special_tokens=["</s>",]
    )

tokenizer.train_from_iterator(word_set, trainer)


fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    eos_token = "</s>",
    bos_token = None,
    unk_token = None
)

fast_tokenizer.save_pretrained(tokenizer_path)

##########################################################################################

def statistic(tokenizer: GPT2TokenizerFast):
    print(f":: tokenizer.config: vocab.sz={len(tokenizer.get_vocab())},",
        f"eos_token_id={tokenizer.eos_token_id}",
        f"pad_token_id={tokenizer.pad_token_id},",
        f"bos_token_id={tokenizer.bos_token_id},")
    print(80*"#")

##########################################################################################

if __name__ == '__main__':

    neo = AutoTokenizer.from_pretrained("data/gpt-neo-125m", use_fast=True)
    pthia = AutoTokenizer.from_pretrained("data/pythia-31m", use_fast=True)
    qwen3 = AutoTokenizer.from_pretrained("data/Qwen3-1.7B", use_fast=True)


    my_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)

    statistic(my_tokenizer)

    tokens_to_file(pthia, word_set, None, "pythia-small")
    tokens_to_file(neo, word_set, None, "gpt-neo")
    tokens_to_file(qwen3, word_set, outpath_gpt2, "qwen3")
    tokens_to_file(my_tokenizer, word_set, outpath, "noomo")
