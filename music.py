"""
music.py
========

Per-guild YouTube music playback for MordBot.

A `MusicPlayer` wraps the guild's voice client with a simple track queue.
Track lookup goes through yt-dlp (blocking, so it runs in a thread), and
playback streams the extracted audio URL through FFmpeg via discord.py's
`FFmpegPCMAudio`. Sending audio works fine while the VoiceRecvClient is
listening -- it's a normal VoiceClient underneath, and discord.py handles
the DAVE (E2EE) encryption of outgoing audio itself.

Every public control method returns a short human-readable string; these
double as tool results for the LLM agent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import discord
import yt_dlp

YTDL_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "default_search": "ytsearch1",
}

# Reconnect flags keep FFmpeg alive across YouTube CDN hiccups.
FFMPEG_BEFORE_OPTS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
FFMPEG_OPTS = "-vn"


def _format_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "?:??"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class Track:
    title: str
    webpage_url: str
    stream_url: str
    duration: Optional[float]
    requested_by: str

    def describe(self) -> str:
        return f"{self.title} [{_format_duration(self.duration)}]"


class MusicPlayer:
    """Track queue + playback controls for one guild's voice connection."""

    def __init__(self, voice_client: discord.VoiceClient, loop: asyncio.AbstractEventLoop):
        self.voice_client = voice_client
        self.loop = loop
        self.queue: list[Track] = []
        self.current: Optional[Track] = None
        self._ytdl = yt_dlp.YoutubeDL(YTDL_OPTS)
        # Where the current track started within the source (nonzero after a
        # speech interruption), used to compute the absolute resume position.
        self._seek_offset: float = 0.0
        self._pending_seek: float = 0.0

    # -- lookup ----------------------------------------------------------

    async def search(self, query: str, requested_by: str) -> Track:
        """Resolves a free-text query (or URL) to a playable Track. Raises on no match."""
        info = await asyncio.to_thread(self._ytdl.extract_info, query, download=False)
        if info is None:
            raise LookupError(f"No results for {query!r}")
        if "entries" in info:  # search result / playlist: take the first hit
            entries = [e for e in info["entries"] if e]
            if not entries:
                raise LookupError(f"No results for {query!r}")
            info = entries[0]
        return Track(
            title=info.get("title", "unknown title"),
            webpage_url=info.get("webpage_url", ""),
            stream_url=info["url"],
            duration=info.get("duration"),
            requested_by=requested_by,
        )

    # -- controls (each returns a short string, used as LLM tool results) --

    async def play(self, query: str, requested_by: str) -> str:
        try:
            track = await self.search(query, requested_by)
        except Exception as e:
            return f"Couldn't find anything to play for {query!r}: {e}"

        self.queue.append(track)
        if self.voice_client.is_playing() or self.voice_client.is_paused():
            return f"Queued at position {len(self.queue)}: {track.describe()}"
        self._start_next()
        return f"Now playing: {track.describe()}"

    def skip(self) -> str:
        if self.current is None:
            return "Nothing is playing."
        skipped = self.current.describe()
        # stop() fires the after-callback, which starts the next queued track
        self.voice_client.stop()
        return f"Skipped {skipped}."

    def pause(self) -> str:
        if not self.voice_client.is_playing():
            return "Nothing is playing."
        self.voice_client.pause()
        return f"Paused {self.current.describe() if self.current else 'playback'}."

    def resume(self) -> str:
        if not self.voice_client.is_paused():
            return "Playback isn't paused."
        self.voice_client.resume()
        return f"Resumed {self.current.describe() if self.current else 'playback'}."

    def stop(self) -> str:
        self.queue.clear()
        had_track = self.current is not None
        self.current = None  # cleared before stop() so the after-callback doesn't chain
        if self.voice_client.is_playing() or self.voice_client.is_paused():
            self.voice_client.stop()
        return "Stopped playback and cleared the queue." if had_track else "Nothing was playing."

    def interrupt(self) -> Optional[tuple[Track, float]]:
        """
        Stops the current track so the bot can speak, returning
        (track, position_seconds) for resume_interrupted(). Returns None if
        nothing needs resuming. Position comes from the audio player's frame
        counter (`_player.loops`, 20ms per frame -- private API, but the only
        place discord.py tracks playback progress).
        """
        if self.current is None:
            return None
        player = getattr(self.voice_client, "_player", None)
        played = player.loops * 0.02 if player else 0.0
        position = self._seek_offset + played
        track, self.current = self.current, None  # None => after-callback won't chain
        self.voice_client.stop()
        return (track, position)

    def resume_interrupted(self, track: Track, position: float) -> None:
        """Restarts an interrupted track at (roughly) where it left off."""
        self.queue.insert(0, track)
        self._pending_seek = max(0.0, position - 1.0)  # rewind 1s for continuity
        self._start_next()

    def queue_summary(self) -> str:
        lines = []
        if self.current:
            lines.append(f"Now playing: {self.current.describe()} (requested by {self.current.requested_by})")
        if self.queue:
            lines.append("Up next:")
            lines.extend(f"  {i}. {t.describe()}" for i, t in enumerate(self.queue, 1))
        return "\n".join(lines) if lines else "The queue is empty and nothing is playing."

    # -- internals ---------------------------------------------------------

    def _start_next(self) -> None:
        """Pops and plays the next queued track. Must run on the event loop thread."""
        if not self.queue or not self.voice_client.is_connected():
            self.current = None
            return

        track = self.queue.pop(0)
        self.current = track
        self._seek_offset, seek = self._pending_seek, self._pending_seek
        self._pending_seek = 0.0
        before_options = FFMPEG_BEFORE_OPTS
        if seek > 0:
            before_options = f"-ss {seek:.2f} {before_options}"
        source = discord.FFmpegPCMAudio(
            track.stream_url,
            before_options=before_options,
            options=FFMPEG_OPTS,
        )
        self.voice_client.play(source, after=self._on_track_end)
        print(f"[music] Now playing: {track.describe()}")

    def _on_track_end(self, error: Optional[Exception]) -> None:
        # Runs on discord.py's audio player thread, so hop back to the loop.
        if error:
            print(f"[music] Playback error: {error}")
        if self.current is None:  # stop() was called; don't chain
            return
        self.current = None
        self.loop.call_soon_threadsafe(self._start_next)
