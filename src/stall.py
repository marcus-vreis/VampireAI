"""Detecção e recuperação de travamento.

O agente tem handler pra 12 estados, mas o jogo tem telas que nenhum deles cobre:
as duas confirmações que a evolução de carta abre, animações longas, diálogos que
não estavam mapeados. Nessas, o handler escolhido aperta o botão errado — ou
nenhum — e o loop fica preso repetindo a mesma coisa pra sempre.

Tratar cada tela desconhecida caso a caso exigiria ter frame de todas, o que não
temos. A rede genérica é mais barata e cobre o que ainda não vimos: se a tela não
mudou depois de N tentativas, escalona botões até destravar. X primeiro porque é
o "confirmar/avançar" do jogo e resolve a maioria das telas de aviso.

Não substitui handler correto — só evita que uma tela desconhecida termine a run.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# Assinatura grosseira do frame: 16x16 em tons de cinza. Insensível a partícula e
# a pulso de destaque, sensível a qualquer mudança real de tela.
_SIGNATURE_SIDE = 16
# Diferença média por pixel abaixo disto é "a mesma tela". Medido em pares reais:
# tela parada com só animação de fundo fica em 0.89-1.46; a MENOR mudança real que
# precisamos detectar é o cursor andando uma carta, em 4.44. Girar no mapa dá
# 17.62. O corte em 2.5 tem ~1.7x de folga pra cada lado.
_SAME_SCREEN_DELTA = 2.5


class Nudge(str, Enum):
    """Botão a apertar pra tentar destravar, em ordem de escalonamento."""

    CONFIRM = "confirm"
    CANCEL = "cancel"
    FORWARD = "walk_forward"


# Ordem escolhida por dano decrescente de acerto: X avança quase toda tela de
# aviso; □ saca dinheiro / volta; andar pra frente destrava o mapa.
_ESCALATION = (Nudge.CONFIRM, Nudge.CANCEL, Nudge.FORWARD)


@dataclass
class StallDetector:
    """Conta quantas iterações seguidas a tela ficou igual.

    `patience` é quantas repetições toleramos antes do primeiro empurrão. Duas é
    o mínimo razoável: uma repetição sozinha acontece o tempo todo em animação de
    ataque, e empurrar ali atrapalharia mais que ajudaria.
    """

    patience: int = 2
    max_nudges: int = len(_ESCALATION)
    _history: deque[np.ndarray] = None  # type: ignore[assignment]
    _repeats: int = 0
    _nudges: int = 0

    def __post_init__(self) -> None:
        self._history = deque(maxlen=1)

    def observe(self, frame: np.ndarray) -> None:
        """Registra o frame atual e atualiza a contagem de repetição."""
        current = _signature(frame)
        if self._history and _same_screen(self._history[0], current):
            self._repeats += 1
        else:
            self._repeats = 0
            self._nudges = 0
        self._history.append(current)

    @property
    def stuck(self) -> bool:
        return self._repeats >= self.patience

    def next_nudge(self) -> Nudge | None:
        """Próximo botão a tentar, ou None se as tentativas acabaram."""
        if not self.stuck or self._nudges >= self.max_nudges:
            return None
        nudge = _ESCALATION[self._nudges]
        self._nudges += 1
        return nudge

    @property
    def exhausted(self) -> bool:
        """Travado e sem mais botões a tentar — hora de abortar a run."""
        return self.stuck and self._nudges >= self.max_nudges

    def reset(self) -> None:
        self._repeats = 0
        self._nudges = 0
        self._history.clear()


def _signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray, (_SIGNATURE_SIDE, _SIGNATURE_SIDE), interpolation=cv2.INTER_AREA
    )
    return small.astype(np.float32)


def _same_screen(a: np.ndarray, b: np.ndarray) -> bool:
    return float(np.abs(a - b).mean()) < _SAME_SCREEN_DELTA
