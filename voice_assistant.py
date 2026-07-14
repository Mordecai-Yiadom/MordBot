"""
voice_assistant.py
===================

Discord voice-channel listening pipeline built on:
  - discord-ext-voice-recv  (receiving raw audio from a Discord call)
  - NVIDIA Nemotron ASR     (transcription, via Hugging Face `transformers`)
  - webrtcvad               (voice-activity detection, to find utterance boundaries)

How it works
------------
1. The bot joins a voice channel with a `voice_recv.VoiceRecvClient`.
2. `VoiceAssistantSink.write()` receives decoded 48kHz stereo PCM for every
   speaker, 20ms at a time (Discord/Opus's native frame size).
3. Each 20ms frame is downmixed to mono and run through WebRTC VAD.
4. Consecutive "speech" frames are buffered into an utterance; once enough
   trailing silence is seen (or a max duration is hit), the utterance is
   handed to a background thread for transcription.
5. The transcript is checked for the wake word. Once heard, the *next*
   utterance from that user is treated as a command and passed to whatever
   `on_command` callback you provide (wire up your actual assistant logic,
   e.g. an LLM call + TTS playback, there).

Model & access notes
---------------------
This targets `nvidia/nemotron-speech-streaming-en-0.6b` by default. It's the
English-only, openly-downloadable predecessor to NVIDIA's newer multilingual
`nvidia/nemotron-3.5-asr-streaming-0.6b`. Both use the same `transformers`
API (just change MODEL_ID below), but as of this writing the multilingual
model's Hugging Face access has been gated/ungated a few times during its
rollout -- check https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
yourself and make sure you're logged in with access before pointing this at
it. Either way, NVIDIA's models on the Hub require you to accept their
license once while logged in; after that, set the `HF_TOKEN` environment
variable (or run `huggingface-cli login`) so `from_pretrained` can download
the weights.

Why not "true" cache-aware streaming?
--------------------------------------
Nemotron ASR *does* support frame-by-frame cache-aware streaming (partial
transcripts while someone is still talking), but that low-level API
(`asr_model.conformer_stream_step()` + manually carried encoder cache state)
currently only exists in the NeMo toolkit, not in `transformers`, and is
genuinely fiddly -- get the chunk/cache bookkeeping slightly wrong and you
get garbled, boundary-artifact-y text. Since this bot only needs to notice a
wake word and then capture one command sentence, not render live captions,
this file instead uses VAD to find utterance boundaries and transcribes each
utterance in one shot with the documented, stable `transformers` API. If you
later want true live streaming captions, the reference implementation is
NeMo's `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`.

Python version note
---------------------
This deliberately avoids the stdlib `audioop` module (used in a lot of older
Discord audio snippets) since it was removed in Python 3.13. Resampling is
done with numpy + scipy instead.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional

import numpy as np
import torch
import webrtcvad
from scipy.signal import resample_poly

import davey
import discord
from discord.ext import voice_recv
from discord.ext.voice_recv import reader as _voice_recv_reader
from discord.ext.voice_recv.rtp import OPUS_SILENCE
from nacl.exceptions import CryptoError
from transformers import AutoModelForRNNT, AutoProcessor

# ---------------------------------------------------------------------------
# DAVE (E2EE) receive shim
# ---------------------------------------------------------------------------
# Since 2026-03-02 Discord *requires* the DAVE end-to-end encryption protocol
# on voice calls (connecting without it is rejected with close code 4017).
# discord.py 2.7+ negotiates DAVE and E2EE-encrypts what the bot *sends*, but
# discord-ext-voice-recv (as of 0.5.2a179) only strips transport encryption on
# receive, so incoming "opus" payloads are still E2EE frames -- garbage to the
# decoder (see voice-recv issue #53). This shim wraps the reader's RTP
# decryptor to additionally run each payload through discord.py's own
# DaveSession, which holds the call's E2EE keys. Remove once voice-recv
# supports DAVE natively.

_original_audioreader_init = _voice_recv_reader.AudioReader.__init__


def _audioreader_init_with_dave(self, sink, voice_client, *, after=None):
    _original_audioreader_init(self, sink, voice_client, after=after)
    transport_decrypt = self.decryptor.decrypt_rtp

    def decrypt_rtp_with_dave(packet):
        payload = transport_decrypt(packet)

        conn = voice_client._connection  # discord.py VoiceConnectionState
        dave = getattr(conn, "dave_session", None)
        if dave is None or getattr(conn, "dave_protocol_version", 0) == 0:
            return payload
        if payload == OPUS_SILENCE:
            return payload  # keepalive silence frames are not E2EE-wrapped

        user_id = voice_client._ssrc_to_id.get(packet.ssrc)
        if user_id is None:
            # Sender unknown (ssrc not mapped yet): undecryptable. Raising
            # CryptoError makes the reader drop just this packet.
            raise CryptoError(f"unknown ssrc {packet.ssrc}, cannot E2EE-decrypt")
        try:
            return dave.decrypt(user_id, davey.MediaType.audio, payload)
        except Exception as exc:
            raise CryptoError(f"DAVE decrypt failed for user {user_id}: {exc}") from exc

    self.decryptor.decrypt_rtp = decrypt_rtp_with_dave


_voice_recv_reader.AudioReader.__init__ = _audioreader_init_with_dave
# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
# Multilingual alternative (check access first -- see module docstring):
# MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

WAKE_WORD = "hey assistant"

DISCORD_SAMPLE_RATE = 48000  # Discord always sends/receives PCM at 48kHz
DISCORD_CHANNELS = 2         # ...stereo...
DISCORD_SAMPLE_WIDTH = 2     # ...16-bit signed PCM.
VAD_FRAME_MS = 20            # matches Discord/Opus's native 20ms frame size


# --------------------------------------------------------------------------
# Model loading -- call this ONCE at startup, never per-command.
# --------------------------------------------------------------------------

def load_asr_model(model_id: str = MODEL_ID):
    """
    Loads the Nemotron ASR model + processor once. Reuse the returned objects
    across every guild/voice session -- do NOT reload per command, the model
    weights are ~600M params.
    """
    print(f"[voice_assistant] Loading ASR model '{model_id}' ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForRNNT.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()
    print(f"[voice_assistant] Model ready on {model.device} (dtype={model.dtype}).")
    return model, processor


# --------------------------------------------------------------------------
# Per-speaker state
# --------------------------------------------------------------------------

@dataclass
class _UserAudioState:
    # "IDLE"      -> waiting to hear the wake word
    # "LISTENING" -> wake word just heard, the *next* utterance is the command
    state: str = "IDLE"
    utterance_pcm: bytearray = field(default_factory=bytearray)
    voiced_frames: int = 0
    silence_frames: int = 0
    speaking: bool = False
    last_frame_time: float = 0.0


# --------------------------------------------------------------------------
# The audio sink
# --------------------------------------------------------------------------

class VoiceAssistantSink(voice_recv.AudioSink):
    """
    Discord audio sink that VAD-segments each speaker's audio into utterances,
    transcribes each one with Nemotron ASR, and dispatches wake-word/command
    handling.

    IMPORTANT: `write()` is invoked by voice-recv's background audio-reader
    thread, NOT the bot's asyncio event loop. Anything here that needs to run
    async code (like calling your `on_command` callback) must hop back onto
    the loop with `asyncio.run_coroutine_threadsafe`.
    """

    def __init__(
        self,
        *,
        model,
        processor,
        loop: asyncio.AbstractEventLoop,
        on_command: Callable[[discord.abc.User, str], Awaitable[None]],
        wake_word: str = WAKE_WORD,
        vad_aggressiveness: int = 2,
        silence_timeout_s: float = 0.8,
        max_utterance_s: float = 15.0,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loop = loop
        self.on_command = on_command
        self.wake_word = wake_word.lower().strip()
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.max_utterance_s = max_utterance_s
        self.silence_timeout_s = silence_timeout_s

        self._frame_bytes_stereo = (
            int(DISCORD_SAMPLE_RATE * VAD_FRAME_MS / 1000) * DISCORD_CHANNELS * DISCORD_SAMPLE_WIDTH
        )
        self._silence_frames_needed = max(1, int(silence_timeout_s * 1000 / VAD_FRAME_MS))
        self._min_voiced_frames = 3  # ignore very short blips / clicks

        # ThreadPoolExecutor so blocking model.generate() calls don't stall
        # the audio-reader thread for other speakers.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="asr-infer"
        )

        self._users: Dict[int, _UserAudioState] = {}
        self._raw_buffers: Dict[int, bytearray] = {}
        self._user_objs: Dict[int, discord.abc.User] = {}

        # Discord stops sending packets when a user stops talking, so the
        # in-band silence counter alone never fires. This watchdog flushes an
        # utterance once no audio has arrived for silence_timeout_s.
        # The lock guards per-user state shared between voice-recv's audio
        # reader thread (write) and the event loop (watchdog).
        self._state_lock = threading.Lock()
        self._watchdog_future = asyncio.run_coroutine_threadsafe(
            self._silence_watchdog(), self.loop
        )

    # -- required OpusSink overrides ------------------------------------

    def wants_opus(self) -> bool:
        # Let voice-recv's decode pipeline handle Opus -> PCM: it manages
        # per-SSRC decoder state, packet ordering, and loss concealment.
        # Decoding packets by hand here produced badly garbled audio.
        return False

    def write(self, user, data: voice_recv.VoiceData):
        if user is None or not data.pcm:
            return

        st = self._users.setdefault(user.id, _UserAudioState())
        self._user_objs[user.id] = user
        raw = self._raw_buffers.setdefault(user.id, bytearray())
        raw.extend(data.pcm)

        # Consume as many complete 20ms stereo frames as we have.
        while len(raw) >= self._frame_bytes_stereo:
            frame = bytes(raw[: self._frame_bytes_stereo])
            del raw[: self._frame_bytes_stereo]
            self._process_frame(user, st, frame)

    def cleanup(self):
        self._watchdog_future.cancel()
        self._users.clear()
        self._raw_buffers.clear()
        self._user_objs.clear()
        self._executor.shutdown(wait=False)

    # -- internals ---------------------------------------------------------

    def _process_frame(self, user, st: _UserAudioState, stereo_frame: bytes):
        mono = self._downmix_to_mono(stereo_frame)  # 48kHz mono, 16-bit PCM bytes

        try:
            is_speech = self.vad.is_speech(mono, DISCORD_SAMPLE_RATE)
        except Exception:
            is_speech = False

        with self._state_lock:
            st.last_frame_time = time.monotonic()

            if is_speech:
                st.utterance_pcm.extend(mono)
                st.voiced_frames += 1
                st.silence_frames = 0
                st.speaking = True
            elif st.speaking:
                st.utterance_pcm.extend(mono)  # keep a little trailing silence for natural cadence
                st.silence_frames += 1

            utterance_seconds = len(st.utterance_pcm) / 2 / DISCORD_SAMPLE_RATE
            finished = st.speaking and (
                st.silence_frames >= self._silence_frames_needed
                or utterance_seconds >= self.max_utterance_s
            )

            if finished:
                self._finalize_utterance(user, st)

    def _finalize_utterance(self, user, st: _UserAudioState):
        """Flush the buffered utterance to transcription. Caller must hold _state_lock."""
        pcm_48k_mono = bytes(st.utterance_pcm)
        voiced_enough = st.voiced_frames >= self._min_voiced_frames
        st.utterance_pcm = bytearray()
        st.voiced_frames = 0
        st.silence_frames = 0
        st.speaking = False
        if voiced_enough:
            self._dispatch_transcription(user, st, pcm_48k_mono)

    async def _silence_watchdog(self):
        """
        Discord clients only transmit while someone is speaking, so trailing
        silence mostly never arrives as packets and the silence_frames counter
        stalls. Poll for users whose audio simply *stopped arriving* and flush
        their utterance after silence_timeout_s.
        """
        while True:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            with self._state_lock:
                for user_id, st in list(self._users.items()):
                    if st.speaking and now - st.last_frame_time >= self.silence_timeout_s:
                        user = self._user_objs.get(user_id)
                        if user is not None:
                            self._finalize_utterance(user, st)

    @staticmethod
    def _downmix_to_mono(stereo_bytes: bytes) -> bytes:
        stereo = np.frombuffer(stereo_bytes, dtype=np.int16).reshape(-1, 2)
        mono = stereo.astype(np.int32).mean(axis=1).astype(np.int16)
        return mono.tobytes()

    def _dispatch_transcription(self, user, st: _UserAudioState, pcm_48k_mono: bytes):
        future = self._executor.submit(self._transcribe, pcm_48k_mono)

        def _on_done(fut: concurrent.futures.Future):
            try:
                text = fut.result()
                print(f"[ASR Engine] Raw text from {user.name}: {text!r}")
            except Exception as exc:
                print(f"[voice_assistant] transcription failed: {exc}")
                return
            if text:
                asyncio.run_coroutine_threadsafe(
                    self._handle_transcript(user, st, text), self.loop
                )

        future.add_done_callback(_on_done)

    def _transcribe(self, pcm_48k_mono: bytes) -> str:
        """Runs on a worker thread. Uses the ASR pipeline to safely transcribe."""
        audio_i16 = np.frombuffer(pcm_48k_mono, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        target_sr = self.processor.feature_extractor.sampling_rate  # normally 16000
        if target_sr != DISCORD_SAMPLE_RATE:
            gcd = np.gcd(DISCORD_SAMPLE_RATE, target_sr)
            up, down = target_sr // gcd, DISCORD_SAMPLE_RATE // gcd
            audio_f32 = resample_poly(audio_f32, up, down).astype(np.float32)

        # Prepare inputs with the model's own processor (this also sets
        # streaming-specific params like num_lookahead_tokens correctly),
        # then let generate() handle the RNN-T transducer decoding.
        try:
            inputs = self.processor(
                audio_f32, sampling_rate=target_sr, return_tensors="pt"
            ).to(self.model.device)
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    inputs[key] = value.to(self.model.dtype)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs)
            # generate() returns a GenerateOutput dataclass; the ids are in .sequences
            output_ids = getattr(outputs, "sequences", outputs)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            print(f"[voice_assistant] ASR inference error: {e}")
            return ""

    async def _handle_transcript(self, user, st: _UserAudioState, text: str):
        lowered = text.lower()
        print(f"[voice_assistant] heard from {user}: {text!r} (state={st.state})")

        if st.state == "IDLE":
            if self.wake_word in lowered:
                remainder = lowered.split(self.wake_word, 1)[1].strip(" ,.!?")
                if remainder:
                    # Wake word + command arrived in the same breath.
                    await self.on_command(user, remainder)
                else:
                    st.state = "LISTENING"
            # else: not addressed to the assistant, ignore
        elif st.state == "LISTENING":
            st.state = "IDLE"
            await self.on_command(user, text)


# --------------------------------------------------------------------------
# Voice connection helpers
# --------------------------------------------------------------------------

class VoiceConnectHelper:
    """Handles connecting/disconnecting with VoiceRecvClient."""

    @staticmethod
    async def connect_to_voice(channel: discord.VoiceChannel) -> Optional[voice_recv.VoiceRecvClient]:
        try:
            voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            return voice_client
        except Exception as e:
            print(f"[voice_assistant] Failed to join voice channel: {e}")
            return None

    @staticmethod
    async def disconnect_voice(voice_client: Optional[discord.VoiceProtocol]):
        if voice_client is not None:
            try:
                await voice_client.disconnect(force=True)
                print("[voice_assistant] Disconnected from voice.")
            except Exception as e:
                print(f"[voice_assistant] Disconnect error: {e}")