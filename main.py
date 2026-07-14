"""
main.py

Discord bot entrypoint. Loads the ASR model once at startup, then exposes
`/join` and `/leave` slash commands that attach a VoiceAssistantSink to a
voice channel.

Note: this uses plain discord.py (not py-cord). discord-ext-voice-recv is
built against discord.py's VoiceClient internals, and discord.py / py-cord
cannot both be installed in the same environment (they both install as the
top-level `discord` package), so pick one -- this project uses discord.py.
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, voice_recv
from dotenv import load_dotenv

from agent import AgentDeps, GuildAgentSession, handle_command
from music import MusicPlayer
from voice_assistant import (
    VoiceAssistantSink,
    VoiceConnectHelper,
    WAKE_WORD,
    load_asr_model,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN is not set. Add it to your .env file.")

intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True  # required to join/track voice channels

bot = commands.Bot(command_prefix="!", intents=intents)

@dataclass
class GuildSession:
    """Everything one guild's assistant needs, torn down together on /leave."""

    voice_client: voice_recv.VoiceRecvClient
    sink: VoiceAssistantSink
    music: MusicPlayer
    agent_session: GuildAgentSession
    text_channel: Optional[discord.TextChannel]


active_sessions: dict[int, GuildSession] = {}

_synced = False


@bot.event
async def on_ready():
    global _synced
    print(f"Logged on as {bot.user} (id={bot.user.id})")

    # Needed for opus decode/encode; matches discord-ext-voice-recv's own example.
    if not discord.opus.is_loaded():
        discord.opus._load_default()

    if not _synced:
        try:
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} slash command(s).")
            _synced = True
        except Exception as e:
            print(f"Slash command sync failed: {e}")


async def on_voice_command(guild_id: int, user: discord.abc.User, text: str):
    """
    Called whenever the assistant hears a full command after the wake word.
    Runs the command through the LLM agent and posts the reply to the
    session's text channel.
    """
    print(f"[assistant] Command from {user}: {text!r}")

    session = active_sessions.get(guild_id)
    guild = bot.get_guild(guild_id)
    if session is None or guild is None:
        return

    member = guild.get_member(user.id)
    name = member.display_name if member else str(user)
    deps = AgentDeps(guild=guild, user_name=name, music=session.music)

    channel = session.text_channel
    try:
        if channel is not None:
            async with channel.typing():
                reply = await handle_command(session.agent_session, deps, text)
        else:
            reply = await handle_command(session.agent_session, deps, text)
    except Exception as e:
        print(f"[assistant] Agent error: {e!r}")
        reply = "⚠️ Sorry, I hit an error handling that."

    print(f"[assistant] Reply: {reply!r}")
    if channel is not None:
        await channel.send(f"🎙️ **{name}**: {text}\n💬 {reply}")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

@bot.tree.command(name="join", description="Join a voice channel to become an assistant")
@app_commands.describe(channel="The voice channel to join")
async def join_command(interaction: discord.Interaction, channel: discord.VoiceChannel):
    await interaction.response.defer(ephemeral=True)

    if interaction.guild is None:
        await interaction.followup.send("This command only works in a server.")
        return

    if interaction.guild.id in active_sessions:
        await interaction.followup.send(
            "I'm already active in a voice channel here. Use `/leave` first."
        )
        return

    voice_client = await VoiceConnectHelper.connect_to_voice(channel)
    if voice_client is None:
        await interaction.followup.send("❌ Could not join that voice channel.")
        return

    sink = VoiceAssistantSink(
        model=bot.asr_model,
        processor=bot.asr_processor,
        loop=bot.loop,
        on_command=partial(on_voice_command, interaction.guild.id),
        wake_word=WAKE_WORD,
    )
    voice_client.listen(sink)

    # Replies go where /join was invoked, falling back to any sendable channel.
    text_channel = interaction.channel
    if not isinstance(text_channel, discord.TextChannel) or not text_channel.permissions_for(
        interaction.guild.me
    ).send_messages:
        text_channel = discord.utils.find(
            lambda c: c.permissions_for(interaction.guild.me).send_messages,
            interaction.guild.text_channels,
        )

    active_sessions[interaction.guild.id] = GuildSession(
        voice_client=voice_client,
        sink=sink,
        music=MusicPlayer(voice_client, bot.loop),
        agent_session=GuildAgentSession(),
        text_channel=text_channel,
    )

    await interaction.followup.send(
        f"✅ Joined **{channel.name}** — say “{WAKE_WORD}” to get my attention."
    )


@bot.tree.command(name="leave", description="Leave the current voice channel")
async def leave_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    if interaction.guild is None or interaction.guild.id not in active_sessions:
        await interaction.followup.send("I'm not in a voice channel here.")
        return

    session = active_sessions.pop(interaction.guild.id)
    session.music.stop()
    await VoiceConnectHelper.disconnect_voice(session.voice_client)
    await interaction.followup.send("👋 Left the voice channel.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Loaded once here, shared across every /join in every guild.
    bot.asr_model, bot.asr_processor = load_asr_model()
    bot.run(BOT_TOKEN)