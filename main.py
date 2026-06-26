import discord
import os
from dotenv import load_dotenv


#load discord bot token from .env file
load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

if(BOT_TOKEN): 
    print("INFO discord token");

TESTING_CHANNEL_NAME = "mordbot-playground"

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')

        if((message.author != self.user) and message.channel.name == TESTING_CHANNEL_NAME):
            await message.channel.send("https://tenor.com/view/sonic-devil-diabolique-evil-gif-9725651736562738158")


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.presences = True

client = MyClient(intents=intents)

client.run(BOT_TOKEN)


