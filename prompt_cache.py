"""
Prompt change detection and caching.

Hashes all inputs that affect LLM output. If inputs are unchanged
between executions, returns the cached prompt without calling LM Studio.
"""

import hashlib
from typing import Optional


class PromptCache:
    _instance = None

    def __init__(self):
        self._hash: Optional[str] = None
        self._result: Optional[str] = None

    @classmethod
    def get(cls) -> "PromptCache":
        if cls._instance is None:
            cls._instance = PromptCache()
        return cls._instance

    def compute_hash(self, **kwargs) -> str:
        h = hashlib.sha256()
        for key in sorted(kwargs.keys()):
            val = kwargs[key]
            if val is None:
                val = ""
            h.update(f"{key}={val}".encode("utf-8", errors="replace"))
        return h.hexdigest()

    def check(self, input_hash: str) -> Optional[str]:
        if self._hash is not None and self._hash == input_hash:
            return self._result
        return None

    def store(self, input_hash: str, result: str):
        self._hash = input_hash
        self._result = result

    def clear(self):
        self._hash = None
        self._result = None
