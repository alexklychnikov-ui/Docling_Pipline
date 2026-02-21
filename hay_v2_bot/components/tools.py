import json
import urllib.request

from haystack import component
from haystack.tools.component_tool import ComponentTool
from openai import OpenAI

from hay_v2_bot.config import OPENAI_MODEL, OPENAI_API_KEY, PROXY_BASE_URL


@component
class DogFactTool:
    """Возвращает случайный факт о собаках (Dog API by kinduff)."""

    @component.output_types(result=str)
    def run(self) -> dict:
        try:
            with urllib.request.urlopen("https://dogapi.dog/api/v2/facts?limit=1", timeout=10) as r:
                data = json.loads(r.read().decode())
            facts = data.get("data", [])
            if facts and "attributes" in facts[0]:
                body = facts[0]["attributes"].get("body", "No fact available.")
                return {"result": body}
            return {"result": "Не удалось получить факт."}
        except Exception as e:
            return {"result": f"Ошибка API: {e}"}


@component
class DogImageDescribeTool:
    """Получает случайную картинку собаки (dog.ceo), отправляет в OpenAI Vision и возвращает описание породы."""

    @component.output_types(result=str)
    def run(self) -> dict:
        try:
            with urllib.request.urlopen("https://dog.ceo/api/breeds/image/random", timeout=10) as r:
                data = json.loads(r.read().decode())
            image_url = data.get("message")
            if not image_url:
                return {"result": "Не удалось получить ссылку на изображение."}
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=PROXY_BASE_URL)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Опиши собаку на фото: укажи породу (или предположение), краткую предысторию породы. Ответь на русском, кратко."},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=500,
            )
            text = (resp.choices[0].message.content or "").strip()
            return {"result": f"Фото: {image_url}\n\n{text}"}
        except Exception as e:
            return {"result": f"Ошибка: {e}"}


dog_fact_tool = ComponentTool(
    name="dog_fact",
    description="Получить случайный интересный факт о собаках. Вызывай, когда пользователь просит факт о собаках или хочет что-то интересное про собак.",
    component=DogFactTool(),
    outputs_to_string={"source": "result"},
)

dog_image_tool = ComponentTool(
    name="dog_image_describe",
    description="Получить случайную фотографию собаки и описание породы с краткой историей. Вызывай, когда пользователь просит картинку собаки или описание породы по фото.",
    component=DogImageDescribeTool(),
    outputs_to_string={"source": "result"},
)
