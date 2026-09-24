"""Fetch local training data as GPT-2-tokenized .bin shards.

Shard format (same as modded-nanogpt / llm.c):
  header: 256 x int32 -> [20240520 magic, 1 version, n_tokens, 0, ...]
  body:   n_tokens x uint16 GPT-2 token ids

  uv run fetch_data.py shakespeare        # ~300K tokens, seconds
  uv run fetch_data.py fineweb            # val + 1 train shard, 100M tokens each (~400MB)
  uv run fetch_data.py fineweb --train-shards 9
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

DATA = Path(__file__).parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[:3] = [20240520, 1, len(tokens)]
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def shakespeare() -> None:
    out = DATA / "shakespeare"
    out.mkdir(parents=True, exist_ok=True)
    text = urllib.request.urlopen(SHAKESPEARE_URL).read().decode()
    tokens = np.array(tiktoken.get_encoding("gpt2").encode_ordinary(text))
    split = int(0.9 * len(tokens))
    write_shard(out / "shakespeare_train_000000.bin", tokens[:split])
    write_shard(out / "shakespeare_val_000000.bin", tokens[split:])
    print(f"shakespeare: {split:,} train / {len(tokens) - split:,} val tokens -> {out}")


def fineweb(train_shards: int) -> None:
    out = DATA / "fineweb10B"
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, train_shards + 1)]
    for name in names:
        hf_hub_download("kjj0/fineweb10B-gpt2", name, repo_type="dataset", local_dir=out)
        print(f"fineweb: {out / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["shakespeare", "fineweb"])
    parser.add_argument("--train-shards", type=int, default=1)
    args = parser.parse_args()
    shakespeare() if args.dataset == "shakespeare" else fineweb(args.train_shards)
