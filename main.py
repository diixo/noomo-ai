from transformers import AutoTokenizer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from utils import gpt_evaluate_to_file, tokens_to_file


outpath = "data/output-cased.txt"

outpath_gpt2 = "data/output-gpt2.txt"


expansion = ['gpt', 'GPT', 'fies', 'fied', 'fic', 'tion', 'tive', 'nce', 'nced', 'nces', 'ncy', 'sor', 'gic', 'dent', 'n\'t',
    'bility', 'nch', 'nal', 'shing', 'erce', 'tly', 'rk', 'LLa', 'lla', 'LM', 'LSTM', 'nge', 'dic', 'ely', '3D',]

##########################################################################################
with open("data/db-full-58900.txt", "r", encoding="utf-8") as f:
    word_set = set([line.strip() for line in f if line.strip()])

word_set = sorted(word_set)

##########################################################################################

gpt2 = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

counters = tokens_to_file(gpt2, word_set, outpath_gpt2) # idx=77638

vocab = sorted(counters.keys())

##########################################################################################
tokenizer_path  = "noomo"

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=50000,
    initial_alphabet=ByteLevel.alphabet(),
    min_frequency=1,
    special_tokens=["</s>",]
    )

tokenizer.train([], trainer)

tokenizer.add_tokens(vocab + expansion)

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

##########################################################################################

if __name__ == '__main__':

    my_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)

    statistic(my_tokenizer)

    tokens_to_file(my_tokenizer, word_set, outpath)    # idx=77728
    #gpt_evaluate_to_file(word_set, outpath)

