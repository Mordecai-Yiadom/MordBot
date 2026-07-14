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

import discord
from discord import app_commands
from discord.ext import commands, voice_recv
from dotenv import load_dotenv

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

# guild_id -> (voice_client, sink), so /leave can clean everything up
active_sessions: dict[int, tuple[voice_recv.VoiceRecvClient, VoiceAssistantSink]] = {}

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


async def on_voice_command(user: discord.abc.User, text: str):
    """
    Called whenever the assistant hears a full command after the wake word.
    This is where your actual assistant logic goes -- call an LLM, run a
    tool, generate TTS audio and play it back with `voice_client.play(...)`,
    etc. Currently just echoes what it heard into a text channel as a demo.
    """
    print(f"[assistant] Command from {user}: {text!r}")

    for guild in bot.guilds:
        member = guild.get_member(user.id)
        if member is None:
            continue
        channel = discord.utils.find(
            lambda c: isinstance(c, discord.TextChannel)
            and c.permissions_for(guild.me).send_messages,
            guild.text_channels,
        )
        if channel:
            await channel.send(f"🎙️ Heard from **{member.display_name}**: {text}")
        break


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
        on_command=on_voice_command,
        wake_word=WAKE_WORD,
    )
    voice_client.listen(sink)
    active_sessions[interaction.guild.id] = (voice_client, sink)

    await interaction.followup.send(
        f"✅ Joined **{channel.name}** — say “{WAKE_WORD}” to get my attention."
    )


@bot.tree.command(name="leave", description="Leave the current voice channel")
async def leave_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    if interaction.guild is None or interaction.guild.id not in active_sessions:
        await interaction.followup.send("I'm not in a voice channel here.")
        return

    voice_client, _sink = active_sessions.pop(interaction.guild.id)
    await VoiceConnectHelper.disconnect_voice(voice_client)
    await interaction.followup.send("👋 Left the voice channel.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Loaded once here, shared across every /join in every guild.
    bot.asr_model, bot.asr_processor = load_asr_model()
    bot.run(BOT_TOKEN)