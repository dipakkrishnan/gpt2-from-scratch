"""Look at a token shard: what the model actually sees.

  uv run scripts/explore_data.py                     # FineWeb val shard
  uv run scripts/explore_data.py data/shakespeare/shakespeare_train_000000.bin

Prints a decoded sample with token boundaries and the most common tokens,
then saves a plot next to the shard (token frequency + document lengths).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tiktoken

EOT = 50256

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fineweb10B/fineweb_val_000000.bin")
tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4)
enc = tiktoken.get_encoding("gpt2")

print(f"{path.name}: {len(tokens):,} tokens\n")
print("sample, one [token] per id:")
print("".join(f"[{enc.decode([t])}]" for t in tokens[:60].tolist()), "\n")

counts = np.bincount(tokens, minlength=enc.n_vocab)
print(f"distinct tokens used: {(counts > 0).sum():,} / {enc.n_vocab:,}")
print("top 20:")
for t in np.argsort(counts)[::-1][:20]:
    print(f"  {counts[t] / len(tokens):6.2%}  {t:>5}  {enc.decode([t])!r}")

doc_starts = np.flatnonzero(tokens == EOT)
fig, axes = plt.subplots(1, 2 if len(doc_starts) > 1 else 1, figsize=(12, 4), squeeze=False)
ranked = np.sort(counts[counts > 0])[::-1]
axes[0, 0].loglog(np.arange(1, len(ranked) + 1), ranked)
axes[0, 0].set(title="token frequency by rank (Zipf)", xlabel="rank", ylabel="count")
if len(doc_starts) > 1:
    doc_lens = np.diff(doc_starts)
    print(f"\ndocuments: {len(doc_lens):,}, median {int(np.median(doc_lens))} tokens, p99 {int(np.percentile(doc_lens, 99))}")
    axes[0, 1].hist(doc_lens, bins=np.logspace(0, np.log10(doc_lens.max()), 60))
    axes[0, 1].set(xscale="log", title="document length (tokens between <|endoftext|>)", xlabel="tokens")
out = path.with_suffix(".png")
fig.tight_layout()
fig.savefig(out)
print(f"\nplot: {out}")
