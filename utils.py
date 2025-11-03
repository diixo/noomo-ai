
import tiktoken
import json
import re


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
