# Docling

Проект с интеграцией Docling для обработки документов и Telegram-ботом на Haystack.

## Структура

- **hay-telegram-bot.py** — исходная версия бота (монолит).
- **hay_v2_bot/** — вторая версия бота с модульной архитектурой:
  - `components/` — хранилище, эмбеддеры, Docling-лоадер, инструменты
  - `pipelines/` — индексация, генерация, резюме, агент
  - `bot/` — обработчики и запуск
  - `main.py` — точка входа

## Запуск бота (hay_v2_bot)

1. Создать виртуальное окружение и установить зависимости:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r hay_v2_bot/requirements.txt
   ```

2. В корне проекта создать `.env` с переменными:
   - `TELEGRAM_BOT_TOKEN` — токен бота
   - `PINECONE_API_KEY` — ключ Pinecone
   - `PROXY_API_KEY` или `OPENAI_API_KEY` — для OpenAI через proxyapi.ru
   - При необходимости: `PROXY_BASE_URL`, `OPENAI_MODEL`, `PINECONE_INDEX_NAME`, `CHUNKER_TOKENIZER`

3. Запуск из корня проекта:
   ```
   python hay_v2_bot/main.py
   ```

Бот обрабатывает текстовые сообщения (с учётом загруженных документов), принимает файлы (PDF, DOCX и др.), сохраняет контент в Pinecone, после загрузки файла отправляет краткое резюме и отвечает на вопросы по документам.

## Windows

При ошибках с симлинками HuggingFace см. **hay_v2_bot/README_FIX_WINDOWS.md**.
