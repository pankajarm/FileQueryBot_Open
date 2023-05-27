from pathlib import Path

APP_NAME = "FileQueryBot"
MODEL = "gpt-3.5-turbo"
EMBED_MODEL = "text-embedding-ada-002"
PAGE_ICON = "🗄️"
PAGE_DESC = "Your Query is my Command!"

K = 10
FETCH_K = 20
CHUNK_SIZE = 1000
TEMPERATURE = 0.7
MAX_TOKENS = 3357

DATA_PATH = Path.cwd() / "data"