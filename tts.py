"""
tts.py
======

Gives MordBot a voice: Kokoro (ONNX, fully local) speech synthesis plus a
per-guild `VoiceSpeaker` that plays the synthesized audio in the voice
channel.

Playback coordination: a discord.py voice client can only play one source
at a time, and pausing music then calling play() would silently replace
the paused source. So before speaking, the current music track (if any)
is *interrupted* -- MusicPlayer remembers the playback position -- and
restarted from (almost) where it left off once the bot finishes talking.

Model files (downloaded from github.com/thewh1teagle/kokoro-onnx releases):
  models/kokoro-v1.0.onnx   models/voices-v1.0.bin
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile

import discord
import soundfile as sf
from kokoro_onnx import Kokoro

from music import MusicPlayer

TTS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "kokoro-v1.0.onnx")
TTS_VOICES_PATH = os.path.join(os.path.dirname(__file__), "models", "voices-v1.0.bin")

TTS_VOICE = os.getenv("TTS_VOICE", "am_michael")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
MAX_SPOKEN_CHARS = 600  # don't monologue: cap synthesis length


def load_tts() -> Kokoro:
    """Loads the Kokoro model once at startup (CPU; ~340MB of files)."""
    print("[tts] Loading Kokoro TTS model ...")
    kokoro = Kokoro(TTS_MODEL_PATH, TTS_VOICES_PATH)
    print(f"[tts] TTS ready (voice={TTS_VOICE}).")
    return kokoro


_URL_RE = re.compile(r"https?://\S+")
_MARKDOWN_RE = re.compile(r"[*_`#>|]")


def _cleanup_for_speech(text: str) -> str:
    """Replies are written for a text channel; make them read well aloud."""
    text = _URL_RE.sub("(link in chat)", text)
    text = _MARKDOWN_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > MAX_SPOKEN_CHARS:
        text = text[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0] + "…"
    return text


class VoiceSpeaker:
    """Speaks text in one guild's voice channel, ducking music around it."""

    def __init__(
        self,
        voice_client: discord.VoiceClient,
        music: MusicPlayer,
        kokoro: Kokoro,
        loop: asyncio.AbstractEventLoop,
    ):
        self.voice_client = voice_client
        self.music = music
        self.kokoro = kokoro
        self.loop = loop
        self._lock = asyncio.Lock()  # one utterance at a time per guild

    async def speak(self, text: str) -> None:
        text = _cleanup_for_speech(text)
        if not text:
            return

        async with self._lock:
            wav_path = await asyncio.to_thread(self._synthesize_to_wav, text)
            if wav_path is None:
                return

            resume_info = self.music.interrupt()
            done = asyncio.Event()

            def _after(error):
                if error:
                    print(f"[tts] Playback error: {error}")
                self.loop.call_soon_threadsafe(done.set)

            try:
                self.voice_client.play(discord.FFmpegPCMAudio(wav_path), after=_after)
                await done.wait()
            except discord.ClientException as e:
                print(f"[tts] Could not play speech: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                if resume_info is not None:
                    self.music.resume_interrupted(*resume_info)

    def _synthesize_to_wav(self, text: str) -> str | None:
        """Runs on a worker thread: Kokoro inference is CPU-bound."""
        try:
            samples, sample_rate = self.kokoro.create(
                text, voice=TTS_VOICE, speed=TTS_SPEED, lang="en-us"
            )
        except Exception as e:
            print(f"[tts] Synthesis failed: {e}")
            return None
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="mordbot_tts_")
        os.close(fd)
        sf.write(path, samples, sample_rate)
        return path
