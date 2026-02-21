import os

from openai import OpenAI

from hay_v2_bot.config import OPENAI_MODEL, OPENAI_API_KEY, PROXY_BASE_URL


def build_file_summary(texts: list[str], max_chars: int = 10000) -> str:
    """Генерирует одно предложение — резюме содержимого документа."""
    if not texts:
        return "Документ не содержит текста."
    combined = "\n\n".join(texts)[:max_chars]
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=PROXY_BASE_URL)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Кратко резюмируй содержимое документа в одном предложении на русском. Только одно предложение.\n\nТекст:\n{combined}",
            }
        ],
        max_tokens=150,
    )
    return (resp.choices[0].message.content or "Резюме недоступно.").strip()
