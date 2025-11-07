
import tiktoken
import json
import re
from pathlib import Path
import pandas as pd


def gpt_evaluate_to_file(words: list, outpath: str):

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

################################################################

def str_tokenize_words(s: str, stopwords = set()) -> list:
    words = re.findall("(\.?\w[\w'\.&-]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []


def clean_text(text):
    """
    Clears the text leaving only:
    - latin (a-z, A-Z)
    - digits (0-9)
    - spaces and line breaks
    - base punctuation: . , ! ? ; : ' " ( ) -
    - additional symbols: [] {} <> @ # & + = * / % ^
    """
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\[\]\{\}<>\-@#&+=*/%^]", "", text)
    return text


def read_vocabulary(file_path: str, count: int = 5) -> set:
    word_set = set()

    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        word_list = [line.strip() for line in f if line.strip()]

        for word in word_list:
            if word not in word_set:
                word_set.add(" ".join([word for _ in range(count)]))
            else:
                print("###:", word)

    print(f"db-full.sz={len(word_set)}")
    return word_set


def read_jsonl(file_path: str) -> list:
    text = []
    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:
            item = json.loads(line)
            title = item["title"]
            description = item["description"]
            text.append(title + "\n" + description)

    return text


def read_eli5():
    dataset = []

    chunk_df = pd.read_parquet("datasets/eli5/pair/train-00000-of-00001.parquet", columns=["question", "answer"])

    for idx, row in chunk_df.iterrows():
        question = row["question"]
        answer = row["answer"]
        if (idx + 1) % 1000 == 0:
            print(f"...items={idx}")
        dataset.append(clean_text(question + " " + answer))
    return dataset


def tokens_to_file(tokenizer, words: list, outpath: str):
    ids_count = 0
    word_count = 0
    with open(outpath, "w", encoding="utf-8") as f_out:
        for w in words:
            if w.find("-") < 0:
                input_ids = tokenizer(w, add_special_tokens=False, padding=False, return_tensors="np")
                input_ids = input_ids["input_ids"]
                input_ids = input_ids[0]
                ids_count += len(input_ids)
                word_count += 1

                f_out.write(f"{w}: {str(tokenizer.convert_ids_to_tokens(input_ids))}\n")
##############################
    ids_count = 0
    vocab = set()
    for w in words:
        input_ids = tokenizer(w, add_special_tokens=False, padding=False, return_tensors="np")
        input_ids = input_ids["input_ids"]
        input_ids = input_ids[0]
        ids_count += len(input_ids)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        vocab.update([t for t in tokens if len(t.lstrip('Ġ')) > 1])

    print(f"word_compression_ratio: {ids_count/len(words):6f} (idx={ids_count}, words={len(words)}), tokens_vocab.sz={len(vocab)}")
