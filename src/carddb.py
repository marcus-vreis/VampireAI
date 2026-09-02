"""Cache de identidade de carta por hash perceptual.

Uma carta é sempre o mesmo sprite. Perguntar ao VLM "que carta é esta?" toda vez
que ela aparece custa ~2.2s por leitura e é a maior fatia da latência de um
turno. Depois da primeira leitura, reconhecer vira busca em tabela.

O hash é um dHash 16x16 sobre o recorte da carta: robusto ao pulso de destaque
(que muda cor, não estrutura) e a um ou dois pixels de deslocamento do recorte.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

_HASH_SIDE = 16
# Limiar medido em recortes reais, sobre 256 bits: a MESMA carta deslocada de 1 a
# 3px (a variação do detector entre frames) fica em 4-36 bits de distância;
# cartas DIFERENTES ficam em 112-131. 60 é o meio, com folga de ~2x pros dois
# lados. Ver `_distance` e o teste `test_carta_diferente_do_mesmo_frame_nao_colide`.
_MAX_DISTANCE = 60


@dataclass
class CardRecord:
    nome: str
    mana: int | None
    descricao: str | None
    tipo: str


def perceptual_hash(card_bgr: np.ndarray) -> str:
    """dHash do recorte da carta, em hexadecimal."""
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (_HASH_SIDE + 1, _HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join(f"{b:02x}" for b in np.packbits(bits))


def _distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 10**6
    xa = np.frombuffer(bytes.fromhex(a), dtype=np.uint8)
    xb = np.frombuffer(bytes.fromhex(b), dtype=np.uint8)
    return int(np.unpackbits(xa ^ xb).sum())


class CardDB:
    """Tabela hash→carta, persistida em JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, CardRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries = {k: CardRecord(**v) for k, v in raw.items()}
        logger.debug("CardDB carregada com {} cartas", len(self._entries))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in self._entries.items()}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def lookup(self, card_bgr: np.ndarray) -> CardRecord | None:
        """Carta já conhecida com este visual, ou None."""
        h = perceptual_hash(card_bgr)
        if h in self._entries:
            return self._entries[h]
        best, best_d = None, _MAX_DISTANCE + 1
        for known, record in self._entries.items():
            d = _distance(h, known)
            if d < best_d:
                best, best_d = record, d
        return best if best_d <= _MAX_DISTANCE else None

    def remember(self, card_bgr: np.ndarray, record: CardRecord) -> None:
        self._entries[perceptual_hash(card_bgr)] = record
        self.save()

    def __len__(self) -> int:
        return len(self._entries)
