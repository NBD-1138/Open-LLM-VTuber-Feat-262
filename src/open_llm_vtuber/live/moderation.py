import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional

LINK_RE = re.compile(r"https?://|\w+\.(com|net|org|io|gg)\b", re.I)
REPEAT_RE = re.compile(r"(.)\1{5,}")
CAPS_RE = re.compile(r"^[A-Z0-9\s\W]{20,}$")


@dataclass
class Strike:
    count: int
    last_ts: float


class ModEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.strikes = defaultdict(lambda: Strike(0, 0.0))

    def _normalize(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        return "".join(
            ch for ch in s if unicodedata.category(ch)[0] != "C"
        )  # drop controls

    def check(self, user: str, message: str) -> Tuple[bool, Optional[str]]:
        if not getattr(self.cfg, "enabled", True):
            return False, None
        msg = self._normalize(message)
        f = self.cfg.filters

        if f.get("block_links") and LINK_RE.search(msg):
            return True, "links"
        if f.get("block_repeats") and REPEAT_RE.search(msg):
            return True, "spam"
        if f.get("block_mass_caps") and CAPS_RE.match(msg):
            return True, "caps"
        for w in f.get("badwords", []):
            if w and w.lower() in msg.lower():
                return True, "badword"
        # slurs/Unicode confusables would be additional lists/patterns
        return False, None

    def apply(self, user: str) -> Tuple[str, Optional[int]]:
        now = time.time()
        rec = self.strikes[user]
        if now - rec.last_ts > self.cfg.decay_hours * 3600:
            rec.count = 0
        rec.count += 1
        rec.last_ts = now

        seq = self.cfg.timeout_sequence_seconds
        if rec.count == 1:
            return "warn", None
        elif 2 <= rec.count <= 1 + len(seq):
            dur = seq[rec.count - 2]
            return "timeout", dur
        else:
            return "ban", None
