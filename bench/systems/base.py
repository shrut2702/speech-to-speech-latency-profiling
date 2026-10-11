"""The interface every system under test implements.

Keeping cascade and Moshi behind one interface is what makes their numbers
comparable. The moment a system needs special-cased timing code, the metrics
stop meaning the same thing.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from ..feeder import AudioFeeder, Frame
from ..trace import Trace


async def drain_to_endpoint(
    feeder: AudioFeeder,
    trace: Trace,
    on_frame: Callable[[Frame], Awaitable[None]] | None = None,
) -> asyncio.Task:
    """Consumes frames up to the endpoint, then keeps draining in the background.

    Systems must react at the endpoint, never at end-of-stream. The file running
    out is an out-of-band signal a live system never receives, and waiting for it
    silently adds the whole trailing-silence pad to every measurement.

    Draining continues in the returned task because the audio does not stop when
    the user does: streaming ASR needs some trailing context to finalize, and a
    full-duplex model keeps listening regardless. Await the task before finishing
    the trial.
    """
    stream = feeder.stream(trace)

    async for frame in stream:
        if on_frame is not None:
            await on_frame(frame)
        if frame.past_endpoint:
            break

    async def _rest() -> None:
        async for frame in stream:
            if on_frame is not None:
                await on_frame(frame)

    return asyncio.create_task(_rest())


async def stop_drain(task: asyncio.Task) -> None:
    """Ends a background drain.

    The feeder keeps emitting silence for `silence_after_s`, which is sized for
    the longest expected response. A half-duplex cascade has no use for that
    audio once it has its transcript, so it cancels instead of waiting, and the
    trial ends when the response does rather than when the silence budget runs
    out.
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def paced(feeder: AudioFeeder, trace: Trace) -> AsyncIterator[Frame]:
    """The feeder's frames, released on the feeder's clock and not the reader's.

    Iterating the feeder directly ties the two together: a generator only
    produces its next frame when asked, so a reader that awaits a model between
    frames holds the microphone still while it does. The frame after is then
    late, and it reads as the host failing to feed in real time when the host
    was fine.

    Here the feeder runs as its own task and leaves frames in a queue, the way
    a microphone fills a buffer whether or not anyone is reading it. A slow
    reader now shows up as a backlog, which is recorded, and the feeder's lag
    is left meaning only what it claims to.
    """
    frames: asyncio.Queue = asyncio.Queue()

    async def feed() -> None:
        try:
            async for frame in feeder.stream(trace):
                frames.put_nowait(frame)
        finally:
            frames.put_nowait(None)

    task = asyncio.create_task(feed())
    backlog = 0
    try:
        while (frame := await frames.get()) is not None:
            # Frames already waiting behind this one.
            backlog = max(backlog, frames.qsize())
            yield frame
    finally:
        await stop_drain(task)
        trace.artifacts["feeder_backlog_ms"] = round(
            backlog * feeder.frame_len / feeder.sr * 1000
        )


@dataclass
class AudioChunk:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


class S2SSystem(ABC):
    """A speech-in, speech-out system.

    Implementations consume the feeder's frames and yield audio as it becomes
    available. They must emit the canonical trace events from `bench.trace` so
    metrics.py can find them.
    """

    name: str = "unnamed"

    @abstractmethod
    async def load(self) -> None:
        """Loads models onto the device. Not timed."""

    async def warmup(self, feeder_factory) -> None:
        """Runs a few throwaway trials.

        First-call latency includes compilation, kernel autotuning and lazy
        allocation, and will wreck the numbers if it lands in the results.
        """
        for _ in range(2):
            trace = Trace(clip_id="__warmup__", config=self.name, trial=-1, warmup=True)
            async for _chunk in self.run(feeder_factory(), trace):
                pass

    @abstractmethod
    def run(self, feeder: AudioFeeder, trace: Trace) -> AsyncIterator[AudioChunk]:
        """Consumes the feeder, yields output audio.

        Must emit at minimum:
          OUTPUT_FIRST_AUDIO on the first chunk,
          OUTPUT_CHUNK per chunk with meta duration_s,
          OUTPUT_END when finished.
        """

    async def unload(self) -> None:
        """Frees device memory. Called between configs."""
        return None
