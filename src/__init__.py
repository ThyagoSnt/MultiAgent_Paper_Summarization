# src/__init__.py
from pathlib import Path

from dotenv import load_dotenv


# Discover project root: folder that contains "configuration", "scripts", etc.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Load .env from project root
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=False)
