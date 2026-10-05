"""B1: greedy chain-of-thought. Also C, the Gemma 4 31B ceiling (spec §8.1).

C uses the same code with a generator pointing at a different
``base_url``/``model``. Spec §8.1 requires recording where C ran: pass it as
``where`` (provider, quantization) and it goes into the output.
"""

from __future__ import annotations

from thoughtzero.baselines.common import BaselineOutput
from thoughtzero.data.grading import extract_answer
from thoughtzero.types import Generator, Problem


async def run_cot(
    generator: Generator, problem: Problem, where: str | None = None
) -> BaselineOutput:
    """One greedy (temperature 0) full solution; the answer is its last ``\\boxed{}``."""
    (solution,) = await generator.sample_solutions(problem.question, n=1, temperature=0.0)
    extra = {"where": where} if where else {}
    return BaselineOutput(answer=extract_answer(solution), solutions=[solution], extra=extra)
