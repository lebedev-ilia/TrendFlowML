"""VK-канал: отправка сообщений владельцу и чтение его ответов через group long poll.

Проверено вручную (2026-07-06): messages.send -> ok; groups.getLongPollServer -> ok.
"""
from __future__ import annotations
import time
import random
import requests

import config

VK_API = "https://api.vk.com/method"

_CHAT_LOG_MAX_LINES = 400


class VKError(RuntimeError):
    pass


def log_chat(direction: str, text: str) -> None:
    """Дописать в state/agent1_chat.log (общий журнал переписки Первого агента с владельцем) —
    это то, что читает Второй агент (supervisor.py / assistant.py), чтобы знать текущий контекст.
    Подрезаем файл, чтобы не рос бесконечно на непрерывной сессии."""
    try:
        line = f"{__import__('datetime').datetime.now().isoformat(timespec='seconds')} {direction} {text[:1500]}\n"
        with open(config.AGENT1_CHAT_LOG, "a", encoding="utf-8") as f:
            f.write(line)
        # Раз в ~50 записей подрезаем хвостом (дёшево: считаем по размеру, не каждый вызов).
        if config.AGENT1_CHAT_LOG.stat().st_size > 400_000:
            lines = config.AGENT1_CHAT_LOG.read_text(encoding="utf-8").splitlines()[-_CHAT_LOG_MAX_LINES:]
            config.AGENT1_CHAT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _call(method: str, **params):
    params.setdefault("access_token", config.VK_TOKEN)
    params.setdefault("v", config.VK_API_VERSION)
    r = requests.post(f"{VK_API}/{method}", data=params, timeout=40)
    data = r.json()
    if "error" in data:
        raise VKError(f"{method}: {data['error'].get('error_msg')} ({data['error'].get('error_code')})")
    return data["response"]


def get_group_id() -> int:
    resp = _call("groups.getById")
    groups = resp["groups"] if isinstance(resp, dict) and "groups" in resp else resp
    g = groups[0] if isinstance(groups, list) else groups
    return int(g["id"])


def send_assistant(text: str) -> None:
    """Отправить владельцу через ВТОРОГО бота (ассистента), VK_TOKEN2."""
    if not config.VK_TOKEN2:
        return
    try:
        requests.post(f"{VK_API}/messages.send", data={
            "user_id": config.VK_OWNER_ID, "random_id": random.randint(1, 2_000_000_000),
            "message": text[:4000], "access_token": config.VK_TOKEN2, "v": config.VK_API_VERSION,
        }, timeout=40)
    except requests.RequestException:
        pass


def send(text: str) -> int:
    """Отправить сообщение владельцу. Возвращает message_id."""
    log_chat("AGENT1->", text)
    return _call(
        "messages.send",
        user_id=config.VK_OWNER_ID,
        random_id=random.randint(1, 2_000_000_000),
        message=text[:4000],
    )


def send_photo(path, caption: str = "") -> int:
    """Отправить фото владельцу (график/визуализация с диска). Стандартный VK-флоу:
    photos.getMessagesUploadServer -> POST файла на upload_url -> photos.saveMessagesPhoto ->
    messages.send с attachment=photoOWNER_ID. Возвращает message_id.

    Добавлено 2026-07-17 для deepdive_agent.py (свободный чат-разбор компонентов) — раньше в
    этом репозитории не было ни одного места, отправляющего VK-фото, только текст."""
    log_chat("AGENT1->", f"[фото] {path}" + (f" — {caption}" if caption else ""))
    upload = _call("photos.getMessagesUploadServer", peer_id=config.VK_OWNER_ID)
    upload_url = upload["upload_url"]
    with open(path, "rb") as f:
        r = requests.post(upload_url, files={"photo": f}, timeout=60)
    r.raise_for_status()
    uploaded = r.json()
    saved = _call(
        "photos.saveMessagesPhoto",
        photo=uploaded["photo"], server=uploaded["server"], hash=uploaded["hash"],
    )
    photo = saved[0] if isinstance(saved, list) else saved
    attachment = f"photo{photo['owner_id']}_{photo['id']}"
    return _call(
        "messages.send",
        user_id=config.VK_OWNER_ID,
        random_id=random.randint(1, 2_000_000_000),
        message=caption[:4000],
        attachment=attachment,
    )


class LongPoll:
    """Group Long Poll — читает входящие message_new от владельца."""

    def __init__(self):
        self.group_id = get_group_id()
        self._refresh()

    def _refresh(self):
        s = _call("groups.getLongPollServer", group_id=self.group_id)
        self.server, self.key, self.ts = s["server"], s["key"], s["ts"]

    def poll_once(self, wait: int = 25):
        """Вернуть список текстов новых сообщений от владельца (может быть пустым)."""
        try:
            r = requests.get(
                self.server,
                params={"act": "a_check", "key": self.key, "ts": self.ts, "wait": wait},
                timeout=wait + 10,
            )
            data = r.json()
        except requests.RequestException:
            return []
        if "failed" in data:
            # 1: устарел ts -> обновить ts; 2/3: переполучить key/server
            if data["failed"] == 1:
                self.ts = data.get("new_ts", self.ts)
            else:
                self._refresh()
            return []
        self.ts = data.get("ts", self.ts)
        texts = []
        for upd in data.get("updates", []):
            if upd.get("type") == "message_new":
                msg = upd["object"]["message"]
                if int(msg.get("from_id", 0)) == config.VK_OWNER_ID and msg.get("text"):
                    texts.append(msg["text"].strip())
        return texts


if __name__ == "__main__":
    # Ручной тест: python messenger.py "текст"
    import sys
    config.require("VK_TOKEN", config.VK_TOKEN)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    msg = sys.argv[1] if len(sys.argv) > 1 else "Тест messenger.py ✅"
    print("group_id:", get_group_id())
    print("sent msg_id:", send(msg))
