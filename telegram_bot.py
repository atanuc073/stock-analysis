"""Telegram delivery — sends the daily summary to your chat."""
from __future__ import annotations
import logging
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)


def send_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing — skipping send")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram 4096-char limit per message; chunk if needed
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
    ok = True
    for chunk in chunks:
        try:
            r = requests.post(url, data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }, timeout=20)
            if r.status_code != 200:
                log.warning("Telegram send failed (%s): %s", r.status_code, r.text)
                ok = False
        except Exception as e:
            log.exception("Telegram send error: %s", e)
            ok = False
    return ok


def send_document(path: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        with open(path, "rb") as f:
            r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
                              files={"document": f}, timeout=60)
        return r.status_code == 200
    except Exception as e:
        log.exception("Telegram doc send error: %s", e)
        return False
