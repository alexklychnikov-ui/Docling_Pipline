"""
Haystack v2 Telegram-бот: RAG по документам (Docling + Pinecone), инструменты (собаки), чат через прокси OpenAI.

Запуск (из корня проекта Docling):

  cd C:\\Python\\Projects\\Zerocode\\Docling
  python -m hay_v2_bot.main

Или напрямую:

  python hay_v2_bot/main.py

Требования: .env в корне Docling с TELEGRAM_BOT_TOKEN, PINECONE_API_KEY, OPENAI_API_KEY (или PROXY_API_KEY),
PROXY_BASE_URL (по умолчанию https://openai.api.proxyapi.ru/v1).
Логи пишутся в терминал и в hay_v2_bot/WORK_LOG.txt.
"""

import os
import sys
import platform
from pathlib import Path

# КРИТИЧНО: устанавливаем переменные ДО импорта любых библиотек HuggingFace
# Принудительно отключаем симлинки на Windows (решает проблему с правами)
if platform.system() == "Windows":
    # Устанавливаем переменные принудительно, даже если они уже есть в системе
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNINGS"] = "1"
    
    # АГРЕССИВНЫЙ monkey patch: перехватываем os.symlink и pathlib.symlink_to
    # Заменяем симлинки на копирование файлов (решает проблему с правами на Windows)
    import builtins
    import shutil
    _original_symlink = getattr(os, 'symlink', None)
    if _original_symlink:
        def _copy_instead_of_symlink(src, dst, target_is_directory=False):
            """Копирует файл/папку вместо создания симлинка. Правильно обрабатывает относительные пути."""
            try:
                # Разрешаем относительные пути относительно dst (как делает symlink)
                dst_path = Path(dst).resolve()
                if not Path(src).is_absolute():
                    # Относительный путь - разрешаем относительно родителя dst
                    src_path = (dst_path.parent / src).resolve()
                else:
                    src_path = Path(src).resolve()
                
                # Проверяем существование источника
                if not src_path.exists():
                    # Пробуем найти относительно текущей рабочей директории
                    alt_src = Path(src).resolve()
                    if alt_src.exists():
                        src_path = alt_src
                    else:
                        raise FileNotFoundError(f"Источник не найден: {src} (разрешено как {src_path})")
                
                # Удаляем существующий файл/папку, если есть
                if dst_path.exists():
                    if dst_path.is_dir():
                        shutil.rmtree(dst_path, ignore_errors=True)
                    else:
                        dst_path.unlink(missing_ok=True)
                
                # Создаем родительские директории
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Копируем файл или папку
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)
                return True
            except Exception as e:
                # Не выводим предупреждение для тестовых файлов (dummy_file)
                if "dummy_file" not in str(src):
                    print(f"[WARN] Ошибка при копировании вместо симлинка {src} -> {dst}: {e}")
                return False
        os.symlink = _copy_instead_of_symlink
    
    # Также патчим pathlib.Path.symlink_to
    try:
        from pathlib import Path as PathLib
        _original_symlink_to = PathLib.symlink_to
        def _noop_symlink_to(self, target, target_is_directory=False):
            """Копирует файл/папку вместо создания симлинка через pathlib. Правильно обрабатывает относительные пути."""
            try:
                # Разрешаем относительные пути относительно self (как делает symlink_to)
                self_path = PathLib(self).resolve()
                if not PathLib(target).is_absolute():
                    # Относительный путь - разрешаем относительно родителя self
                    target_path = (self_path.parent / target).resolve()
                else:
                    target_path = PathLib(target).resolve()
                
                # Проверяем существование источника
                if not target_path.exists():
                    # Пробуем найти относительно текущей рабочей директории
                    alt_target = PathLib(target).resolve()
                    if alt_target.exists():
                        target_path = alt_target
                    else:
                        raise FileNotFoundError(f"Источник не найден: {target} (разрешено как {target_path})")
                
                # Удаляем существующий файл/папку
                if self_path.exists():
                    if self_path.is_dir():
                        shutil.rmtree(self_path, ignore_errors=True)
                    else:
                        self_path.unlink(missing_ok=True)
                
                # Создаем родительские директории
                self_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Копируем
                if target_path.is_dir():
                    shutil.copytree(target_path, self_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(target_path, self_path)
            except Exception as e:
                # Не выводим предупреждение для тестовых файлов
                if "dummy_file" not in str(target):
                    print(f"[WARN] Ошибка при копировании через pathlib {target} -> {self}: {e}")
        PathLib.symlink_to = _noop_symlink_to
    except Exception:
        pass

# Настраиваем HuggingFace кэш в папку проекта
root = Path(__file__).resolve().parent.parent
hay_v2_root = Path(__file__).resolve().parent
hf_cache_dir = hay_v2_root / ".hf_cache"
hf_cache_dir.mkdir(exist_ok=True)
os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)

# АГРЕССИВНАЯ очистка: удаляем весь кэш Docling моделей при запуске на Windows
# Это нужно для перезагрузки моделей с копированием вместо симлинков
if platform.system() == "Windows":
    try:
        import shutil
        import glob
        # Удаляем ВСЕ модели Docling из кэша (принудительная очистка)
        patterns = ["models--docling-project--*", "models--docling-*"]
        deleted_count = 0
        for pattern in patterns:
            for cache_path in glob.glob(str(hf_cache_dir / pattern)):
                p = Path(cache_path)
                if p.exists():
                    try:
                        shutil.rmtree(p, ignore_errors=True)
                        print(f"[INFO] Удалён кэш для перезагрузки: {p.name}")
                        deleted_count += 1
                    except Exception as ex:
                        print(f"[WARN] Не удалось удалить {p.name}: {ex}")
        if deleted_count > 0:
            print(f"[INFO] Удалено {deleted_count} кэшей моделей Docling (будет перезагружено с копированием файлов)")
        # Проверяем переменные окружения
        print(f"[DEBUG] HF_HUB_DISABLE_SYMLINKS={os.environ.get('HF_HUB_DISABLE_SYMLINKS', 'НЕ УСТАНОВЛЕНО')}")
        print(f"[DEBUG] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', 'НЕ УСТАНОВЛЕНО')}")
    except Exception as e:
        print(f"[WARN] Не удалось очистить кэш: {e}")

# Добавляем корень проекта в sys.path для прямого запуска
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from hay_v2_bot.bot.run import run_bot

if __name__ == "__main__":
    run_bot()
