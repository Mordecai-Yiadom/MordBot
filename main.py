import asyncio
import discord
from discord.ext import tasks
import os
from dotenv import load_dotenv
from ollama import AsyncClient

load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3")

if BOT_TOKEN:
    print("INFO discord token loaded")

ALLOWED_CHANNELS = {"mordbot-playground", "helmet-heroes-general", "damnation"}

with open("personality.txt", "r") as f:
    SYSTEM_PROMPT = f.read().strip()


class MyClient(discord.Client):

    DEBOUNCE_SECONDS = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ollama = AsyncClient()
        self.history = {}
        self.pending_messages = {}
        self.pending_tasks = {}

    async def on_ready(self):
        print(f"Logged on as {self.user}!")
        for guild in client.guilds:
            for channel in guild.channels:
                if channel.name == "mordbot-playground":
                    self.channel = channel
        self.patty_hatred.start()

    async def on_message(self, message):
        if message.author == self.user:
            return

        print(f"Message from {message.author}: {message.content}")

        if message.channel.name in ALLOWED_CHANNELS:
            channel_id = message.channel.id
            self.pending_messages.setdefault(channel_id, []).append(
                (message.author.display_name, message.content)
            )

            existing = self.pending_tasks.get(channel_id)
            if existing:
                existing.cancel()

            self.pending_tasks[channel_id] = asyncio.get_event_loop().create_task(
                self._debounced_reply(message.channel)
            )

    async def _debounced_reply(self, channel):
        await asyncio.sleep(self.DEBOUNCE_SECONDS)

        messages = self.pending_messages.pop(channel.id, [])
        self.pending_tasks.pop(channel.id, None)

        if not messages:
            return

        combined = "\n".join(f"{author}: {content}" for author, content in messages)
        if len(messages) > 1:
            combined += "\n\n(reply to all of the above with a single reaction, not one line per message)"

        async with channel.typing():
            reply = await self.get_ai_response(channel.id, combined)
        await channel.send(reply)

    async def get_ai_response(self, channel_id, user_message):
        history = self.history.setdefault(channel_id, [])

        history.append({
            "role": "user",
            "content": user_message,
        })

        response = await self.ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        )

        reply = response.message.content
        history.append({"role": "assistant", "content": reply})
        
        if len(history) > 30:
            self.history[channel_id] = history[-30:]

        return reply

    @tasks.loop(hours=1)
    async def patty_hatred(self):
        if self.channel:
            await self.channel.send("kill patrick hoban")


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.presences = True

client = MyClient(intents=intents)
client.run(BOT_TOKEN)
