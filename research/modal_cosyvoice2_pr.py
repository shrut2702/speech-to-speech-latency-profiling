"""CosyVoice2 from the fork's PR branch on Modal. Same flow as modal_cosyvoice2.py.

    modal run research/modal_cosyvoice2_pr.py --stream --warmup
    modal run research/modal_cosyvoice2_pr.py --stream --warmup --decoder-process
    modal run research/modal_cosyvoice2_pr.py --stream --warmup --decoder-process --decoder-device 1
    modal run research/modal_cosyvoice2_pr.py --stream --warmup --sample long

Point of this file: run the opt-in decoder process (upstream issue #1951) through
the library's public API, and check it reproduces what modal_cosyvoice2_split.py
measured by driving internals. Same texts, same prompt, same timeline as
modal_cosyvoice2.py, so the three files can be read side by side.

The image clones the fork on the PR branch. The function pulls before running,
so a push made after the image was built is still what runs, and the commit it
ran is printed with the timeline.
"""

import modal

MODEL = "iic/CosyVoice2-0.5B"
REPO = "/opt/CosyVoice"
FORK = "https://github.com/shrut2702/CosyVoice"
BRANCH = "decoder-process"

# The two texts modal_cosyvoice2.py was run with. Three sentences for short, so
# streaming has something to chunk; the long one is split by the library into
# two pieces, which is the case where the gap between pieces shows.
SAMPLES = {
    "short": (
        "The capital of France is Paris. "
        "It has been the seat of government since the tenth century. "
        "Today it is home to just over two million people."
    ),
    "long": (
        "Photosynthesis converts light energy into chemical energy. "
        "Plants absorb sunlight through chlorophyll in their leaves, "
        "then use that energy to combine carbon dioxide from the air with water drawn up from the roots. "
        "The result is glucose, which the plant uses for growth, and oxygen, "
        "which is released back into the atmosphere. "
        "Almost every food chain on the planet starts with this reaction, "
        "which is why a change in plant cover affects far more than the plants themselves."
    ),
}

image = (
    modal.Image.micromamba(python_version="3.10")
    # pynini via conda-forge. pip builds of it fail on glibc and cost an hour.
    .micromamba_install("pynini=2.1.5", channels=["conda-forge"])
    .apt_install("git", "ffmpeg", "sox", "libsox-dev", "build-essential")
    .run_commands(
        f"git clone --recursive --branch {BRANCH} {FORK} {REPO}",
        # setuptools 81 dropped pkg_resources and one of CosyVoice's
        # requirements still imports it from its setup.py. PIP_CONSTRAINT is the
        # pin that reaches inside pip's isolated build environment.
        "echo 'setuptools<81' > /tmp/constraint.txt",
        "pip install 'setuptools<81' wheel",
        f"PIP_CONSTRAINT=/tmp/constraint.txt pip install -r {REPO}/requirements.txt",
        # DeepSpeed is a training dependency that compiles CUDA ops on import,
        # which needs nvcc this image does not carry. Inference never touches it.
        "pip uninstall -y deepspeed",
    )
    .pip_install("modelscope")
    .env({"PYTHONPATH": f"{REPO}:{REPO}/third_party/Matcha-TTS"})
)

app = modal.App("cosyvoice2-pr", image=image)
cache = modal.Volume.from_name("s2s-models", create_if_missing=True)
out = modal.Volume.from_name("s2s-results", create_if_missing=True)


def checkout() -> str:
    """Brings the image's clone up to the branch's latest push.

    The clone is baked into the image when it builds, and Modal reuses that
    layer for as long as the command reads the same, so without this every run
    after the first would test the commit the image happened to be built on.
    """
    import subprocess

    subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=REPO, check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          check=True, capture_output=True, text=True).stdout.strip()
    print(f"running {FORK} {BRANCH} @ {head}", flush=True)
    return head


def hops(cosyvoice, when: str) -> None:
    """The chunk schedule, printed where it is read.

    token_hop_len doubles in place during streaming and is not reset
    between utterances, so what it holds depends on what ran before.
    """
    m = cosyvoice.model
    print("[%s] token_hop_len=%s pre_lookahead_len=%s token_max_hop_len=%s"
          % (when, m.token_hop_len, m.flow.pre_lookahead_len, m.token_max_hop_len),
          flush=True)


def instrument(cosyvoice, events: list) -> None:
    """Times the LM, and the decoder where it can be seen.

    The LM is wrapped the same way in both paths. The decoder is not: in the
    shipped path token2wav runs in this process and is timed directly, but with
    the decoder process it runs in another interpreter, so the closest thing
    visible from here is the moment each chunk is handed over. The chunk-out
    events close the loop in both cases.
    """
    import time

    model = cosyvoice.model
    original_llm = model.llm.inference

    def timed_llm(*args, **kwargs):
        n = 0
        for token in original_llm(*args, **kwargs):
            n += 1
            if n == 1:
                events.append(("llm first token", time.monotonic()))
            yield token
        events.append((f"llm last token ({n})", time.monotonic()))

    model.llm.inference = timed_llm

    if model.decoder_process is not None:
        dp = model.decoder_process
        original_send = dp.send_chunk

        def timed_send(uuid, token, token_offset, finalize):
            events.append((f"chunk sent ({token.shape[1]} tokens, offset {token_offset}"
                           f"{', final' if finalize else ''})", time.monotonic()))
            return original_send(uuid, token, token_offset, finalize)

        dp.send_chunk = timed_send
    else:
        original_token2wav = model.token2wav

        def timed_token2wav(*args, **kwargs):
            events.append(("decoder start", time.monotonic()))
            try:
                return original_token2wav(*args, **kwargs)
            finally:
                events.append(("decoder end", time.monotonic()))

        model.token2wav = timed_token2wav


# Two cards so --decoder-device 1 can be measured. Without it the second one
# idles, which keeps every arm of the comparison on one image.
@app.function(gpu="A10G:2", volumes={"/cache": cache, "/out": out}, timeout=3600)
def synth(
    sample: str, stream: bool, prompt_wav: str, prompt_text: str,
    warmup: bool, decoder_process: bool, decoder_device: int | None,
) -> list[str]:
    import time

    import torchaudio
    from modelscope import snapshot_download

    head = checkout()
    # Imported after the pull, so the code that runs is the code just checked out.
    from cosyvoice.cli.cosyvoice import CosyVoice2

    text = SAMPLES[sample]
    model_dir = snapshot_download(MODEL, local_dir=f"/cache/{MODEL}")
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False,
                           decoder_process=decoder_process, decoder_device=decoder_device)
    print(f"decoder_process={decoder_process} decoder_device={decoder_device}", flush=True)

    hops(cosyvoice, "loaded")

    if warmup:
        # The first inference pays for CUDA kernel selection, lazy module init
        # and the text frontend building its FSTs, in both processes when the
        # decoder has its own.
        for _ in cosyvoice.inference_cross_lingual("Warming up.", prompt_wav, stream=True):
            pass
        print("warmed", flush=True)
        hops(cosyvoice, "after warmup")

    events: list = []
    instrument(cosyvoice, events)

    t0 = time.monotonic()
    paths = []
    gen = (
        cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=stream)
        if prompt_text
        else cosyvoice.inference_cross_lingual(text, prompt_wav, stream=stream)
    )
    tag = "dp" if decoder_process else "base"
    for i, chunk in enumerate(gen):
        audio = chunk["tts_speech"]
        dur = audio.shape[-1] / cosyvoice.sample_rate
        events.append((f"chunk {i} out ({dur:.2f}s audio)", time.monotonic()))
        path = f"/out/cosyvoice2_pr_{tag}_{i:03d}.wav"
        torchaudio.save(path, audio, cosyvoice.sample_rate)
        paths.append(path)

    total = (time.monotonic() - t0) * 1000
    # Printed at the end, not as they happen. The LM runs on another thread, so
    # inline prints would interleave the two and misrepresent the order.
    print(f"\n--- timeline, ms from synthesis start ({BRANCH} @ {head}) ---", flush=True)
    previous = t0
    for label, when in sorted(events, key=lambda e: e[1]):
        print(f"{(when - t0) * 1000:8.0f}  (+{(when - previous) * 1000:6.0f})  {label}")
        previous = when
    print(f"total {total:.0f} ms, {len(paths)} chunks", flush=True)

    out.commit()
    return paths


@app.local_entrypoint()
def main(
    sample: str = "short",
    stream: bool = False,
    # The repo's own asset, as in modal_cosyvoice2.py. A Chinese speaker, so
    # English text goes through the cross-lingual path.
    prompt_wav: str = f"{REPO}/asset/zero_shot_prompt.wav",
    prompt_text: str = "",
    warmup: bool = False,
    # Flow and hift in their own process. Only stream inference uses it.
    decoder_process: bool = False,
    # Which card that process gets. Leave it and it shares cuda:0 with the LM.
    decoder_device: int = -1,
):
    if sample not in SAMPLES:
        raise SystemExit(f"--sample must be one of {sorted(SAMPLES)}")
    print("\n".join(synth.remote(
        sample, stream, prompt_wav, prompt_text, warmup,
        decoder_process, None if decoder_device < 0 else decoder_device,
    )))
