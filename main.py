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
def read_vocab(add_prefix_space=False):
    prefix = " " if add_prefix_space==True else ""
    with open("data/db-full-59776.txt", "r", encoding="utf-8") as f:
        word_set = set([prefix + line.strip() for line in f if line.strip()])
    return word_set


#word_set = sorted(read_vocab(False))
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

#tokenizer.add_tokens(vocab + expansion)

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
    #gpt_evaluate_to_file(word_set, outpath)


def filtering():

    eli5_freq = tokens_to_file(gpt2, read_vocabulary("eli5-dictionary.txt", 1), "eli-gpt2.txt")

    tokens_to_file(my_tokenizer, read_vocabulary("eli5-dictionary.txt", 1), "eli-noomo.txt")

    ####################################

    print(">> eli5.sz=", len(eli5_freq))
    for k in my_freq:
        eli5_freq.pop(k, None)

    for k in list(eli5_freq.keys()):
        if k.isdigit():
            eli5_freq.pop(k)
    eli5_sorted = eli5_freq.most_common()
    print("<< eli5.sz=", len(eli5_freq))


    vocab = my_tokenizer.get_vocab()
    append_sz = 12000 - (len(vocab) - 256 - 1)

    append_keys = [k for k, v in eli5_sorted[:append_sz]]
    print(append_sz, f": {len(append_keys)}, vocab.sz={len(vocab)}")

