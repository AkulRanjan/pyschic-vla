"""Search tree node (SPEC.md §6.1). Owner: Person 1 (Prakhar).

Virtual loss (SPEC.md §6.2): while ``virtual_loss`` simulations are in flight through a node,
PUCT treats it as having ``N + virtual_loss`` visits, the extra ones contributing value 0.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass(eq=False)
class Node:
    steps: list[str]
    prior: float = 1.0
    N: int = 0
    W: float = 0.0
    value: float | None = None
    terminal: bool = False
    final_answer: str | None = None
    children: list[Node] = field(default_factory=list)
    expanding: asyncio.Future[None] | None = None
    virtual_loss: int = 0

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def Q(self) -> float:
        """Mean backed-up value over real visits (0.0 when unvisited)."""
        return self.W / self.N if self.N else 0.0

    @property
    def n_eff(self) -> int:
        """Visits including in-flight simulations (virtual loss)."""
        return self.N + self.virtual_loss

    def q_eff(self, fpu_value: float) -> float:
        """Q used by PUCT: virtual visits count as value 0; ``fpu_value`` if never visited."""
        n = self.n_eff
        return self.W / n if n else fpu_value
