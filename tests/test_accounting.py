import asyncio

from thoughtzero.accounting import (
    current_ledger,
    ledger_scope,
    record_gemma,
    record_jev,
    record_search,
)


def test_record_without_ledger_is_noop() -> None:
    assert current_ledger.get() is None
    record_gemma(10, 20)
    record_jev(100, 0.01, cache_hit=False)
    record_search(expansions=1, depth=3)


def test_scope_collects_and_resets() -> None:
    with ledger_scope() as ledger:
        record_gemma(10, 20)
        record_gemma(5, 7)
        record_jev(100, 0.5, cache_hit=False)
        record_jev(100, 0.0, cache_hit=True)
        record_search(expansions=2, depth=4, terminal_leaves=1)
        record_search(depth=2)
    assert current_ledger.get() is None
    assert ledger.gemma_calls == 2
    assert ledger.gemma_prompt_tokens == 15
    assert ledger.gemma_completion_tokens == 27
    assert ledger.jev_calls == 1 and ledger.jev_cache_hits == 1
    assert ledger.jev_input_tokens == 100 and ledger.jev_usd == 0.5
    assert ledger.expansions == 2 and ledger.max_depth == 4 and ledger.terminal_leaves == 1
    assert ledger.wall_time_s >= 0


async def test_child_tasks_share_the_ledger() -> None:
    async def work() -> None:
        await asyncio.sleep(0)
        record_gemma(1, 1)

    with ledger_scope() as ledger:
        async with asyncio.TaskGroup() as tg:
            for _ in range(5):
                tg.create_task(work())
    assert ledger.gemma_calls == 5


async def test_concurrent_scopes_do_not_mix() -> None:
    async def problem(n: int) -> int:
        with ledger_scope() as ledger:
            for _ in range(n):
                await asyncio.sleep(0)
                record_gemma(0, 1)
        return ledger.gemma_completion_tokens

    assert await asyncio.gather(problem(3), problem(7)) == [3, 7]
