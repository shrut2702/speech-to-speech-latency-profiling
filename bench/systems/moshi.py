"""Moshi: full-duplex speech-to-speech, behind the same interface as the cascade.

Structurally different in a way that matters for the comparison. Moshi listens
and speaks at once, so it has no endpointing step and no notion of "the user has
finished". It decides for itself when to talk, which is why the whole stream is
fed rather than stopping at the endpoint.

That is exactly why t=0 comes from the manifest's annotated end-of-speech rather
than from a VAD. It is a property of the audio file, so both systems are
measured from the identical instant and neither is handed an advantage.

Moshi is one model on one card, so it has none of the stage contention the
cascade suffers. It still runs in a worker process, as the cascade's stages do,
so that in both systems the harness process is doing nothing but feeding and
tracing. Sharing an interpreter with the model would mean the feeder waiting on
the lock to release a frame, and that delay would land in Moshi's own numbers.

Two rate mismatches the harness bridges. The feeder streams 16kHz because that
is what Whisper wants, and Mimi is a 24kHz codec. And Mimi runs at 12.5Hz, one
frame per 80ms, so the feeder's 20ms frames are accumulated before a step. That
80ms is a floor on how finely Moshi can react, and it is a property of the
architecture rather than the host.

Finding the response is the hard part, and research/moshi-profile.md is the
working. Moshi emits a frame every 80ms whether or not it is speaking, so the
first frame to arrive is not the first audio. A frame counts as speech when its
energy clears a threshold sitting in the empty gap between Mimi's silence and
Mimi's speech, and the response starts at the first such frame past the
endpoint that also carries a word rather than a `<pad>` or `<unk>`.

Audio before the endpoint is recorded and not counted. Fed a question, Moshi
often greets the user partway through it, which is real behaviour worth
reporting and is not a reply.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import numpy as np

from ..feeder import AudioFeeder
from ..trace import Trace, OUTPUT_FIRST_AUDIO, OUTPUT_CHUNK, OUTPUT_END
from ..workers import StageWorker
from .base import AudioChunk, S2SSystem

MIMI_SR = 24000
MIMI_FRAME = 1920  # 80ms at 24kHz
# Frames kept past the last one above the speech threshold. The decay at
# the end of a word drops under the threshold while it is still audible.
TAIL_FRAMES = 3


class MoshiSystem(S2SSystem):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", "moshi")
        model = cfg.get("model", {})
        self.devices = cfg.get("devices", {})
        self.worker = StageWorker("moshi", model, self.devices.get("moshi"))
        # Silence reads under 0.003 and speech over 0.02, with two empty
        # decades between, so the threshold only has to land in the gap.
        # research/moshi-profile.md has the distribution it came from.
        self.rms_speech = float(model.get("rms_speech", 0.01))
        # The feeder pads a long tail of silence because a microphone does not
        # switch off while the user waits. Feeding all of it every trial would
        # cost those seconds for nothing, so the turn ends when the model stops
        # talking. Both bounds are measured from the endpoint.
        self.silence_gap_s = float(model.get("silence_gap_s", 1.5))
        self.max_response_s = float(model.get("max_response_s", 20.0))
        self.out_sr = MIMI_SR
        self.rtf: float | None = None

    # ---- model ----------------------------------------------------------

    async def load(self) -> None:
        await self.worker.start()
        print(f"  moshi up on gpu {self.devices.get('moshi')}", flush=True)

    async def unload(self) -> None:
        await self.worker.stop()

    async def warmup(self, feeder_factory) -> None:
        """Refuses to run if Moshi cannot keep pace with the audio.

        Above a real-time factor of 1.0 every number afterwards describes the
        GPU rather than the architecture.
        """
        self.rtf = await self.worker.call("warmup")
        print(f"  moshi warm, rtf {self.rtf:.2f} "
              f"({self.rtf * 80:.0f} ms per 80ms frame)", flush=True)
        if self.rtf >= 1.0:
            raise RuntimeError(
                f"moshi real-time factor is {self.rtf:.2f}, at or above 1.0; "
                "move to a larger GPU before trusting any measurement"
            )

    @staticmethod
    def _is_word(piece: str) -> bool:
        """A word, rather than one of the stream's structural tokens.

        `<pad>` fills the frames where nothing is said. `<unk>` turns up
        interleaved through real speech as well, so it marks the stream rather
        than naming a sound. Neither starts a response.
        """
        if not piece:
            return False
        return not (piece.startswith("<") and piece.endswith(">"))

    # ---- the trial ------------------------------------------------------

    async def run(
        self, feeder: AudioFeeder, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        trace.env.setdefault("moshi_rtf", self.rtf)
        trace.env.setdefault("devices", self.devices)
        await self.worker.call("moshi_reset")

        buffer = np.zeros(0, dtype=np.float32)
        speaking = False
        last_speech: float | None = None
        early: list[float] = []
        early_said: list[str] = []
        said: list[str] = []
        pending: list[tuple[float, AudioChunk]] = []
        # Every decoded frame, regardless of whether it is early, quiet, or
        # part of the response.  Saved as session.wav so the full session
        # can be audited without re-running the trial.
        session_audio: list[np.ndarray] = []

        # Past the endpoint, not stopping at it. A full-duplex model never stops
        # listening and can only keep generating while frames keep arriving, so
        # cutting the input early would starve it mid-sentence.
        async for frame in feeder.stream(trace):
            buffer = np.concatenate([buffer, self._to_mimi(frame.samples, feeder.sr)])

            while len(buffer) >= MIMI_FRAME:
                block, buffer = buffer[:MIMI_FRAME], buffer[MIMI_FRAME:]
                result = await self.worker.call("moshi_step", block)
                if result is None:
                    continue
                tok, piece, out = result
                if out is None or not len(out):
                    continue
                session_audio.append(out)

                now = time.monotonic()
                loud = float(np.sqrt(np.mean(out ** 2))) >= self.rms_speech
                past_endpoint = (
                    feeder.t_endpoint is not None and now >= feeder.t_endpoint
                )

                if loud and not past_endpoint:
                    # Moshi talking over the user. Recorded, never counted as a
                    # response, and the reason the endpoint test exists. On the
                    # clips where it answers early and then has nothing left to
                    # say, this is the only record of what it said.
                    early.append(now)
                    if self._is_word(piece):
                        early_said.append(piece)
                    continue
                if not (loud or speaking):
                    continue

                if not speaking:
                    # The response starts on a frame that is loud and carries a
                    # word. Energy alone would start it on a `<unk>` with a
                    # trace of signal in it.
                    if not self._is_word(piece):
                        continue
                    speaking = True
                    trace.mark(OUTPUT_FIRST_AUDIO)

                if self._is_word(piece):
                    said.append(piece)
                chunk = AudioChunk(out, self.out_sr)

                # Quiet frames inside the reply are kept, because the text
                # stream goes quiet between words and dropping them would chop
                # the audio into syllables. Quiet frames after the reply are
                # held back until speech resumes and dropped if it never does,
                # so a response does not carry the silence gap on the end.
                if not loud:
                    pending.append((now, chunk))
                    continue
                last_speech = now
                for t_arrived, held in pending:
                    trace.mark_at(OUTPUT_CHUNK, t_arrived,
                                  duration_s=held.duration_s)
                    yield held
                pending = []
                trace.mark(OUTPUT_CHUNK, duration_s=chunk.duration_s)
                yield chunk

            if self._turn_over(feeder, last_speech):
                break

        # Two held frames go out after all, so the audio ends where the
        # measurement says it does. A word's last syllable decays below the
        # threshold before it has finished being heard, so cutting at the last
        # loud frame clips the end of the reply.
        if last_speech is not None:
            for t_arrived, tail in pending[:TAIL_FRAMES]:
                trace.mark_at(OUTPUT_CHUNK, t_arrived, duration_s=tail.duration_s)
                yield tail

        # The response ends where the audio does, not where the loop does. The
        # loop runs on for silence_gap_s after that, waiting to find out
        # whether the reply is over. The last loud frame's own 80ms, plus the
        # tail frames just yielded.
        if last_speech is not None:
            trace.mark_at(
                OUTPUT_END,
                last_speech + (1 + TAIL_FRAMES) * MIMI_FRAME / MIMI_SR,
            )
        else:
            trace.mark(OUTPUT_END)
        # Moshi's inner monologue is its answer in text, which makes it the
        # thing to score against reference_answer, the same as the cascade's
        # LLM response.
        trace.artifacts["response"] = "".join(said).replace("▁", " ").strip()
        if session_audio:
            trace.artifacts["session_audio"] = (
                np.concatenate(session_audio), self.out_sr
            )
        if early and feeder.t_endpoint is not None:
            trace.artifacts["early_speech_ms"] = [
                round((t - feeder.t_endpoint) * 1000) for t in early
            ]
            trace.artifacts["early_response"] = (
                "".join(early_said).replace("▁", " ").strip()
            )

    def _to_mimi(self, samples: np.ndarray, sr: int) -> np.ndarray:
        if sr == MIMI_SR:
            return samples
        import soxr

        return soxr.resample(samples, sr, MIMI_SR).astype(np.float32)

    def _turn_over(self, feeder: AudioFeeder, last_speech: float | None) -> bool:
        now = time.monotonic()
        if feeder.t_endpoint is None or now < feeder.t_endpoint:
            return False
        if now - feeder.t_endpoint > self.max_response_s:
            return True
        # last_speech, not the last frame. Frames never stop arriving, so a gap
        # measured on them would never open and every trial would run to
        # max_response_s.
        return last_speech is not None and now - last_speech > self.silence_gap_s
