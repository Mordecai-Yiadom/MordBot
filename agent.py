"""
agent.py
========

The LLM brain of MordBot. Voice commands (already transcribed by
voice_assistant.py) are handed to a Pydantic AI agent that can chat, search
the web, and control music playback via tools.

The model is served by LM Studio's local OpenAI-compatible server; set
LLM_BASE_URL / LLM_MODEL in .env (defaults below). Because it's the
standard OpenAI wire format, pointing this at any other provider later is
a config change, not a code change.

Conversation history is kept per guild as a list of per-run message chunks
(`result.new_messages()`), so trimming old turns always removes whole runs
and never orphans a tool call from its result. The system prompt is passed
as `instructions`, which Pydantic AI re-attaches on every run instead of
storing in history -- so it survives any amount of trimming.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import discord
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import tools
from music import MusicPlayer

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")

# Keep this many most-recent agent runs (one run = one voice command,
# including any tool calls it made) as conversation memory per guild.
MAX_HISTORY_RUNS = 10

INSTRUCTIONS = """\
You are MordBot, a voice assistant sitting in a Discord voice channel.

What you receive is speech-to-text output, prefixed with the speaker's
name (e.g. "Joe: play some jazz"). Transcripts may contain recognition
errors -- infer the intended meaning instead of taking odd words literally.

You can chat, search the web, and control music playback with your tools.
Use web_search for anything involving current events or facts you're
unsure of. When asked for music, just play it -- don't ask for
confirmation.

Your replies are posted to a text channel and may later be spoken aloud:
keep them to 1-3 conversational sentences, no markdown headings or lists
unless genuinely needed.
"""


@dataclass
class AgentDeps:
    """Per-command context handed to tools."""

    guild: discord.Guild
    user_name: str
    music: MusicPlayer


def _build_agent() -> Agent[AgentDeps, str]:
    model = OpenAIChatModel(
        LLM_MODEL,
        provider=OpenAIProvider(base_url=LLM_BASE_URL, api_key="lm-studio"),
    )
    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
        instructions=INSTRUCTIONS,
        retries=1,
    )

    @agent.tool
    async def play_music(ctx: RunContext[AgentDeps], query: str) -> str:
        """Play a song or add it to the queue. `query` is a song/artist description or a YouTube URL."""
        return await ctx.deps.music.play(query, requested_by=ctx.deps.user_name)

    @agent.tool
    async def skip_track(ctx: RunContext[AgentDeps]) -> str:
        """Skip the currently playing track."""
        return ctx.deps.music.skip()

    @agent.tool
    async def pause_music(ctx: RunContext[AgentDeps]) -> str:
        """Pause the currently playing track."""
        return ctx.deps.music.pause()

    @agent.tool
    async def resume_music(ctx: RunContext[AgentDeps]) -> str:
        """Resume paused music."""
        return ctx.deps.music.resume()

    @agent.tool
    async def stop_music(ctx: RunContext[AgentDeps]) -> str:
        """Stop playback entirely and clear the music queue."""
        return ctx.deps.music.stop()

    @agent.tool
    async def show_queue(ctx: RunContext[AgentDeps]) -> str:
        """Show what's currently playing and the upcoming music queue."""
        return ctx.deps.music.queue_summary()

    @agent.tool_plain
    async def web_search(query: str) -> str:
        """Search the web. Use for current events, facts, or anything you don't know."""
        return await tools.web_search(query)

    return agent


assistant_agent = _build_agent()


@dataclass
class GuildAgentSession:
    """Rolling conversation memory + run serialization for one guild."""

    runs: list[list[ModelMessage]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def history(self) -> list[ModelMessage]:
        return [msg for run in self.runs for msg in run]

    def remember(self, run_messages: list[ModelMessage]) -> None:
        self.runs.append(run_messages)
        del self.runs[:-MAX_HISTORY_RUNS]


async def handle_command(session: GuildAgentSession, deps: AgentDeps, text: str) -> str:
    """Runs one voice command through the agent and returns the reply text."""
    async with session.lock:
        result = await assistant_agent.run(
            f"{deps.user_name}: {text}",
            deps=deps,
            message_history=session.history or None,
        )
        session.remember(result.new_messages())
        return result.output
