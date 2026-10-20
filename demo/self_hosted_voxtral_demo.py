"""Live transcription demo: an AgentSession with Silero VAD and SelfHostedSTT.

Plays an audio file into the session at real-time pace (as a microphone would),
prints interim transcripts as they arrive and each final transcript with the time
from the VAD's end of speech to the transcript.

    python demo/self_hosted_voxtral_demo.py --audio demo.wav --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time

import aiohttp

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    StopResponse,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.voice.io import AudioInput
from livekit.plugins import mistralai, silero

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160  # 10 ms


def load_pcm(path: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        check=True,
        capture_output=True,
    ).stdout


class FileAudioInput(AudioInput):
    """Paces a file's audio out in 10 ms frames, like a microphone, then silence."""

    def __init__(self, pcm: bytes) -> None:
        super().__init__(label="FileAudioInput")
        self._pcm = pcm
        self._pos = 0
        self._next_at: float | None = None
        self.started_at = 0.0

    async def __anext__(self) -> rtc.AudioFrame:
        now = time.perf_counter()
        if self._next_at is None:
            self._next_at = self.started_at = now
        elif self._next_at > now:
            await asyncio.sleep(self._next_at - now)
        self._next_at += FRAME_SAMPLES / SAMPLE_RATE
        chunk = self._pcm[self._pos : self._pos + FRAME_SAMPLES * 2]
        self._pos += FRAME_SAMPLES * 2
        chunk = chunk.ljust(FRAME_SAMPLES * 2, b"\x00")
        return rtc.AudioFrame(chunk, SAMPLE_RATE, 1, FRAME_SAMPLES)


class TranscribeOnly(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Transcribe only.")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:  # type: ignore[no-untyped-def]
        raise StopResponse()


async def main(args: argparse.Namespace) -> None:
    pcm = load_pcm(args.audio)
    duration = len(pcm) / (SAMPLE_RATE * 2)
    audio_in = FileAudioInput(pcm)

    http = aiohttp.ClientSession()
    session = AgentSession(
        stt=mistralai.SelfHostedSTT(
            base_url=args.base_url, max_model_len=args.max_model_len, http_session=http
        ),
        vad=silero.VAD.load(),
        turn_handling={"turn_detection": "vad"},
    )
    session.input.audio = audio_in

    end_of_speech_at: float | None = None

    def t() -> str:
        return f"{time.perf_counter() - audio_in.started_at:5.2f}s"

    @session.on("user_state_changed")
    def _on_state(ev: UserStateChangedEvent) -> None:
        nonlocal end_of_speech_at
        if ev.new_state == "speaking":
            sys.stdout.write(f"\r\033[K\033[2m[{t()}] speech starts\033[0m\n")
        elif ev.old_state == "speaking":
            end_of_speech_at = time.perf_counter()
            sys.stdout.write(f"\r\033[K\033[2m[{t()}] end of speech (VAD)\033[0m\n")
        sys.stdout.flush()

    @session.on("user_input_transcribed")
    def _on_transcript(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            lag = ""
            if end_of_speech_at is not None:
                lag = f"  \033[36m{(time.perf_counter() - end_of_speech_at) * 1000:.0f} ms after end of speech\033[0m"
            sys.stdout.write(f"\r\033[K[{t()}] \033[1mFINAL\033[0m {ev.transcript}{lag}\n")
        else:
            sys.stdout.write(f"\r\033[K[{t()}] \033[2m… {ev.transcript[-90:]}\033[0m")
        sys.stdout.flush()

    print(f"SelfHostedSTT -> {args.base_url}  (Voxtral Realtime on vLLM)")
    print(f"playing {args.audio} ({duration:.1f}s) into an AgentSession with Silero VAD\n")
    await session.start(TranscribeOnly())
    await asyncio.sleep(duration + args.tail)
    await session.aclose()
    await http.close()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tail", type=float, default=2.0, help="seconds to wait after the audio")
    asyncio.run(main(parser.parse_args()))
