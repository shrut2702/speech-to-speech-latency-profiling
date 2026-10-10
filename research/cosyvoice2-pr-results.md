# CosyVoice2 decoder process: PR branch results

The opt-in decoder process (flow and hift in their own process, optionally on
their own GPU) measured through the library's public API with
`modal_cosyvoice2_pr.py`. Runs are from commit `d1302aa` of
`shrut2702/CosyVoice`, branch `decoder-process`.

All runs: warm, streaming, cross-lingual off the repo's `zero_shot_prompt.wav`,
fp16 off, no JIT or TRT, on A10G. Each run is a separate `modal run`, and each
row is the mean of its runs.

- **First audio**: synthesis start to the first chunk out.
- **Wall clock per second of audio**: the whole run's time divided by the audio
  it produced. Lower is better. The LM samples, so audio length moves run to
  run, and this is what makes totals comparable. It is a whole-run real-time
  factor; it is not called RTF because the library logs a per-chunk `rtf` of its
  own, which is a different number.
- **ms per token**: the LM's first token to its last, divided by the tokens it
  emitted. Unloaded, the LM runs at 15.6.

## Short text

> The capital of France is Paris. It has been the seat of government since the
> tenth century. Today it is home to just over two million people.

| config | runs | first audio | finished | audio | wall clock per second of audio | ms per token | first audio speedup | total speedup |
|---|---|---|---|---|---|---|---|---|
| baseline (flag off) | 3 | 1648 ms | 6607 ms | 10.45 s | 0.632 | 21.7 | | |
| decoder process, one GPU | 2 | 1243 ms | 5937 ms | 10.32 s | 0.575 | 19.3 | 1.33x | 1.10x |
| decoder process, two GPUs | 2 | 1177 ms | 5045 ms | 10.32 s | 0.489 | 15.9 | 1.40x | 1.29x |

## Long text

> Photosynthesis converts light energy into chemical energy. Plants absorb
> sunlight through chlorophyll in their leaves, then use that energy to combine
> carbon dioxide from the air with water drawn up from the roots. The result is
> glucose, which the plant uses for growth, and oxygen, which is released back
> into the atmosphere. Almost every food chain on the planet starts with this
> reaction, which is why a change in plant cover affects far more than the
> plants themselves.

The library splits this into two pieces and runs them in turn, so there are two
LM rates, one per piece.

| config | runs | first audio | finished | audio | wall clock per second of audio | ms per token | first audio speedup | total speedup |
|---|---|---|---|---|---|---|---|---|
| baseline (flag off) | 2 | 1788 ms | 19879 ms | 29.78 s | 0.667 | 22.5 then 21.1 | | |
| decoder process, one GPU | 2 | 1311 ms | 18712 ms | 30.00 s | 0.624 | 22.1 then 19.2 | 1.36x | 1.07x |
| decoder process, two GPUs | 2 | 1220 ms | 14828 ms | 30.00 s | 0.494 | 16.0 then 15.8 | 1.47x | 1.35x |

The baseline's two runs were 9 and 10 chunks: piece one gets an extra decoder
pass when the LM emits 491 tokens or more. Every decoder-process run was 9.
Against the 9-chunk baseline run alone, the total speedup is 1.00x on one GPU
and 1.26x on two.

## Takeaway

On one GPU the decoder process gets audio started about a third sooner and
leaves the total roughly where it was. On two GPUs it is about 1.4x to first
audio and 1.3x on the total, with the LM back at its unloaded rate.
