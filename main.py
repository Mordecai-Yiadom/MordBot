import discord
import os
from dotenv import load_dotenv


#load discord bot token from .env file
load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if(BOT_TOKEN): 
    print("INFO discord token");

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')

        if(message.author != self.user): 
            await message.channel.send(message.content);


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.presences = True

client = MyClient(intents=intents)

client.run(BOT_TOKEN)


