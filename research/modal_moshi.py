"""Moshi, full-duplex. A spike, not part of the harness.

    modal run research/modal_moshi.py --warmup
    modal run research/modal_moshi.py --warmup --clip gsm8k_0000

The cascade tells you when it has something to say: nothing comes out of the
TTS until there is text to speak. Moshi does not. Mimi runs at 12.5Hz and the
model emits a frame every 80ms for as long as you feed it, silent or not, so
"the first chunk arrived" is true 80ms into every trial and means nothing.

This spike exists to work out what to measure instead. Two candidate signals,
recorded per frame:

  text token   Moshi's inner monologue, column 0 of the LM output. It emits a
               pad while silent and a word token when it decides to speak, so
               it marks intent with no threshold to tune. It leads the
               acoustics, by design.
  frame energy RMS of the decoded waveform. This is what a listener actually
               hears, so it is the number comparable to the cascade's time to
               first audio. Mimi's silence is codec noise rather than zeros, so
               the floor has to be calibrated rather than assumed.

Warmup feeds silence and calibrates both from that run rather than from a
constant: an energy floor, and the set of text tokens that appear while nothing
is being said. Fed silence, Moshi still talks unprompted, so neither can be
taken at face value. The floor is a high percentile rather than the maximum,
and a token has to cover a real share of the frames before it counts as a pad.
Both are printed alongside the raw distribution so the rule can be checked
rather than trusted.

Audio is fed at wall clock pace, as the harness feeds it. Handing Moshi the
whole array at once would let it answer before the speaker finished.
"""

import modal

REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_SR = 24000
MIMI_FRAME = 1920          # 80ms at 24kHz, one Mimi step
CODEBOOKS = 8
# Silence sits under 0.003 and speech over 0.025, with nothing between
# them, so the exact value does not matter as long as it is in the gap.
RMS_SPEECH = 0.01

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "moshi>=0.2", "soundfile>=0.12", "soxr>=0.5", "numpy<2",
        # Only for the rms histogram, written to the volume as a png.
        "matplotlib>=3.8",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    # Mounted, not baked: editing a clip should not rebuild the image.
    .add_local_dir("data/clips", "/root/clips")
    .add_local_file("data/manifest.jsonl", "/root/manifest.jsonl")
)

app = modal.App("moshi-spike", image=image)
cache = modal.Volume.from_name("s2s-models", create_if_missing=True)
out = modal.Volume.from_name("s2s-results", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/cache": cache, "/out": out}, timeout=3600)
def run(clip_id: str, warmup: bool, listen_s: float, device: str = "cuda") -> list[str]:
    import json
    import time

    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from moshi.models import LMGen, loaders

    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in open("/root/manifest.jsonl", encoding="utf-8")
        if line.strip()
    }
    row = rows[clip_id]
    audio, sr = sf.read(f"/root/clips/{row['file']}", dtype="float32")
    endpoint_s = row["endpoint_sample"] / sr
    speech = soxr.resample(audio[: row["endpoint_sample"]], sr, MIMI_SR)
    print(f"{clip_id}: {endpoint_s:.2f}s of speech, then silence", flush=True)
    print(f"  {row['transcript'][:100]}", flush=True)

    ckpt = loaders.CheckpointInfo.from_hf_repo(REPO)
    mimi = ckpt.get_mimi(device=device)
    mimi.set_num_codebooks(CODEBOOKS)
    lm = ckpt.get_moshi(device=device)
    try:
        text_tok = ckpt.get_text_tokenizer()
    except Exception:
        text_tok = None

    # Streaming opens once and stays open. The state lives on lm and mimi
    # rather than on LMGen, so building a second LMGen does not get you a clean
    # model: its _init_streaming_state calls lm.streaming() and asserts that
    # something is already streaming. reset_streaming is the way back.
    gen = LMGen(lm, temp=0.0, temp_text=0.0)
    gen.streaming_forever(1)
    mimi.streaming_forever(1)

    def new_turn():
        """Clears the KV cache and the codec state, keeping the stream open."""
        gen.reset_streaming()
        mimi.reset_streaming()

    def piece_of(tok: int) -> str:
        if text_tok is None:
            return ""
        try:
            return text_tok.id_to_piece(tok)
        except Exception:
            return ""

    def is_text(tok: int) -> bool:
        """A word, rather than one of the stream's structural tokens.

        <pad> fills the frames where nothing is being said. <unk> turns up
        interleaved through real speech as well, so it marks the stream rather
        than a word. Both are read off the tokenizer when there is one, and
        fall back to whatever the silence run showed.
        """
        piece = piece_of(tok)
        if piece:
            return not (piece.startswith("<") and piece.endswith(">"))
        return tok not in quiet_tokens

    def hist(values, bins: int = 24, width: int = 50) -> None:
        """Log-spaced histogram, printed.

        The values span three decades, from codec silence to Moshi talking, so
        linear bins would put everything in the first one and tell you nothing
        about where the two populations separate.
        """
        v = np.asarray(values, dtype=float)
        lo = max(float(v.min()), 1e-6)
        hi = max(float(v.max()), lo * 10)
        edges = np.geomspace(lo, hi, bins + 1)
        counts, _ = np.histogram(v, bins=edges)
        peak = max(int(counts.max()), 1)
        total = max(len(v), 1)
        for c, a in zip(counts, edges[:-1]):
            bar = "#" * int(round(width * c / peak))
            print(f"  {a:10.5f} {c:5d} {100 * c / total:5.1f}%  {bar}", flush=True)

    def plot_rms(values, floor_v: float) -> str:
        """The same distribution as a png on the volume.

        Log x axis for the same reason the printed bins are log spaced. The
        point of the picture is whether the silent frames and the spoken ones
        are two separate humps or one smear, since that decides whether energy
        alone can be used as a speech test.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        v = np.asarray(values, dtype=float)
        edges = np.geomspace(max(v.min(), 1e-6), max(v.max(), 1e-5), 40)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(v, bins=edges, weights=np.ones(len(v)) / len(v), color="#4c78a8")
        ax.axvline(floor_v, color="#e45756", ls="--",
                   label=f"floor {floor_v:.5f}")
        ax.set_xscale("log")
        ax.set_xlabel("frame rms")
        ax.set_ylabel("share of frames")
        ax.set_title(f"Moshi on silence, {len(v)} frames of 80ms")
        ax.legend()
        fig.tight_layout()
        path = f"/out/moshi_silence_rms_{clip_id}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    def step(block: np.ndarray):
        """One 80ms frame in, (text token, audio, gpu ms) out."""
        t0 = time.monotonic()
        with torch.no_grad():
            x = torch.from_numpy(block).to(device).unsqueeze(0).unsqueeze(0)
            tokens = gen.step(mimi.encode(x))
            if tokens is None:
                return None, None, (time.monotonic() - t0) * 1000
            # Column 0 is the inner monologue, the audio codebooks follow.
            text = int(tokens[0, 0, 0].item())
            pcm = mimi.decode(tokens[:, 1:]).cpu().numpy().reshape(-1)
        return text, pcm, (time.monotonic() - t0) * 1000

    # ---- warmup: what does this model look like when it is silent ---------

    floor = None
    plot_path = None
    quiet_tokens: set[int] = set()
    if warmup:
        from collections import Counter

        new_turn()
        silence = np.zeros(MIMI_FRAME, dtype=np.float32)
        rms, busy, seen = [], [], Counter()
        # The first frames pay for kernel selection, so they are fed and then
        # discarded rather than averaged into the floor.
        for i in range(int(12 / 0.08)):
            text, pcm, ms = step(silence)
            if pcm is None or i < 12:
                continue
            rms.append(float(np.sqrt(np.mean(pcm ** 2))))
            busy.append(ms)
            seen[text] += 1

        # Fed nothing but silence, Moshi still says things unprompted, so the
        # loudest frame in here is speech rather than a noise floor. A high
        # percentile is the compromise: above the codec's own hiss, below
        # anything the model actually chose to say.
        q = np.percentile(rms, [50, 90, 99]) if rms else [0.0, 0.0, 0.0]
        floor = float(q[1])
        # A token emitted once is a word it decided on. A pad appears
        # constantly, so require a real share of the frames.
        quiet_tokens = {tok for tok, n in seen.items() if n >= 0.05 * len(rms)}

        print(f"\nsilence over {len(rms)} frames", flush=True)
        print(f"  rms p50 {q[0]:.5f}  p90 {q[1]:.5f}  p99 {q[2]:.5f}  "
              f"max {max(rms):.5f}", flush=True)
        print(f"  gpu {np.median(busy):.1f} ms/frame over an 80ms budget, "
              f"rtf {np.median(busy) / 80:.2f}", flush=True)
        print(f"  text tokens: {seen.most_common()}", flush=True)
        print(f"  treating {sorted(quiet_tokens)} as quiet, floor {floor:.5f}",
              flush=True)
        # Where the two populations sit, and whether anything lands between
        # them. A gap means a flat threshold will do and the percentile is not
        # needed; an overlap means energy alone cannot separate speech from
        # silence and the text token has to carry it.
        print("\n  rms distribution, log bins (lower edge, count, share)",
              flush=True)
        hist(rms)
        plot_path = plot_rms(rms, floor)

    # ---- the turn ---------------------------------------------------------

    new_turn()
    frames = int((endpoint_s + listen_s) / 0.08)
    events = []
    produced = []
    busy = []

    t0 = time.monotonic()
    for i in range(frames):
        # Wall clock pace. A live model gets the next 80ms when 80ms has
        # passed, not when it has finished thinking about the last one.
        due = t0 + (i + 1) * 0.08
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        start = i * MIMI_FRAME
        block = speech[start : start + MIMI_FRAME]
        if len(block) < MIMI_FRAME:
            block = np.pad(block, (0, MIMI_FRAME - len(block)))

        text, pcm, ms = step(block)
        busy.append(ms)
        if pcm is None:
            continue
        produced.append(pcm)
        events.append({
            "t_ms": (time.monotonic() - t0) * 1000,
            "text": text,
            "rms": float(np.sqrt(np.mean(pcm ** 2))),
        })

    # ---- what happened ----------------------------------------------------

    zero = endpoint_s * 1000
    print(f"\n--- per frame, ms from endpoint (endpoint at {zero:.0f}ms into "
          f"the run) ---", flush=True)
    print(f"{'ms':>8s} {'rms':>9s} {'text':>6s}  piece", flush=True)
    for e in events:
        piece = ""
        if text_tok is not None:
            try:
                piece = text_tok.id_to_piece(e["text"])
            except Exception:
                piece = ""
        quiet = e["text"] in quiet_tokens and (floor is None or e["rms"] <= floor)
        if quiet:
            continue
        print(f"{e['t_ms'] - zero:8.0f} {e['rms']:9.5f} {e['text']:6d}  {piece}",
              flush=True)

    louder = [e for e in events if floor is not None and e["rms"] > floor]
    spoke = [e for e in events if is_text(e["text"])]
    # Time to first audio: the first frame that is past the endpoint, loud
    # enough to be speech, and carrying a word rather than a pad. The endpoint
    # condition is what separates a reply from Moshi greeting the user mid
    # question, which it does unprompted.
    after = [e for e in events if e["t_ms"] >= zero]
    loud_after = [e for e in after if e["rms"] >= RMS_SPEECH]
    answer = [e for e in loud_after if is_text(e["text"])]

    print("\n--- onsets, ms from endpoint ---", flush=True)
    if spoke:
        print(f"  text     {spoke[0]['t_ms'] - zero:8.0f}  token {spoke[0]['text']}",
              flush=True)
    if louder:
        print(f"  energy   {louder[0]['t_ms'] - zero:8.0f}  rms "
              f"{louder[0]['rms']:.5f} against a floor of {floor:.5f}", flush=True)
    if answer:
        a = answer[0]
        print(f"  ttfa     {a['t_ms'] - zero:8.0f}  rms {a['rms']:.5f} "
              f"token {a['text']} {piece_of(a['text'])}", flush=True)
    else:
        print("  ttfa          none  nothing past the endpoint cleared "
              f"rms {RMS_SPEECH} on a word token", flush=True)

    # The end of the response is energy alone. The text stream goes quiet
    # between words and pads out the ends of them, so requiring a word token
    # here would cut the reply off mid syllable. The +80ms is the frame's own
    # length: the audio runs to the end of that frame, not to its start.
    if loud_after:
        last = loud_after[-1]
        end_ms = last["t_ms"] - zero + 80
        print(f"  end      {end_ms:8.0f}  rms {last['rms']:.5f}", flush=True)
        if answer:
            print(f"  spoke    {end_ms - (answer[0]['t_ms'] - zero):8.0f}  ms of "
                  "response, first frame to last", flush=True)
        if last is events[-1]:
            print("  WARNING: still speaking when the run ended, raise listen_s",
                  flush=True)
    print(f"  gpu      {np.median(busy):.1f} ms/frame, rtf "
          f"{np.median(busy) / 80:.2f}", flush=True)

    written = [plot_path] if plot_path else []
    if produced:
        path = f"/out/moshi_{clip_id}.wav"
        sf.write(path, np.concatenate(produced), MIMI_SR)
        written.append(path)
    # One commit for both. Inside the `if` it would skip the png on a run that
    # produced no audio, which is the run you would most want to look at.
    if written:
        out.commit()
    return written


@app.local_entrypoint()
def main(
    clip: str = "gsm8k_0000",
    # Seconds to keep feeding silence past the endpoint. Moshi only generates
    # while frames arrive, so cutting the input short would cut it off mid
    # sentence.
    listen_s: float = 15.0,
    warmup: bool = False,
):
    print("\n".join(run.remote(clip, warmup, listen_s)))
