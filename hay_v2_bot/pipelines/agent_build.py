import os

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from hay_v2_bot.components import dog_fact_tool, dog_image_tool
from hay_v2_bot.config import OPENAI_MODEL, PROXY_BASE_URL, OPENAI_API_KEY


def build_agent():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY или PROXY_API_KEY должен быть задан в .env")
    generator = OpenAIChatGenerator(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        api_key=Secret.from_token(OPENAI_API_KEY),
        api_base_url=PROXY_BASE_URL,
    )
    return Agent(
        chat_generator=generator,
        tools=[dog_fact_tool, dog_image_tool],
        system_prompt="""Ты — умный персональный помощник в Telegram. Ведёшь диалог естественно и учитываешь контекст предыдущих сообщений и загруженных пользователем документов.
Используй инструменты dog_fact и dog_image_describe по запросу (факты о собаках, картинки собак, описание пород).
Отвечай кратко, по-русски, дружелюбно. Если передан контекст (диалог или фрагменты документов) — опирайся на него.""",
        exit_conditions=["text"],
        max_agent_steps=10,
    )
