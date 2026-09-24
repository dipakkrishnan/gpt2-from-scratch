# gpt2-from-scratch

Hand-written GPT-2 (124M), trained toward the modded-nanogpt target: **3.28 val loss on FineWeb**.

## Setup

```bash
uv sync
uv run fetch_data.py shakespeare
uv run fetch_data.py fineweb          # val + 1 train shard (~400MB)
```

## Plan

1. `model.py` — GPT-2.
2. Logit-match harness — load HF `gpt2` weights into your model, assert logits match `transformers.GPT2LMHeadModel`.
   Gotcha: HF uses `Conv1D`, so `c_attn`/`c_proj`/`c_fc` weights are transposed vs `nn.Linear`.
3. Overfit one batch on MPS → loss ≈ 0.
4. Train on Shakespeare until samples look like Shakespeare.
5. Rent 1×H100, train on FineWeb to 3.28. Add bf16, `torch.compile`, flash attention one at a time; measure tok/s each.
6. Port modded-nanogpt tricks one at a time; keep what beats your clock.

## Sanity numbers

- Initial loss ≈ ln(50257) ≈ 10.8. Much higher → init is broken.
- Training FLOPs ≈ 6 × params × tokens.

## Data format

`data/*/*.bin`: 256×int32 header `[20240520, 1, n_tokens, ...]`, then `n_tokens`×uint16 GPT-2 token ids.
Read with `np.fromfile(path, dtype=np.uint16, offset=256*4)`.
