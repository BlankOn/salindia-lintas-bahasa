"""One model call on the GPU at a time, urgent work first.

Measured on an M4: the same model run three times concurrently gains only 2-7%
throughput, and two *different* models (Whisper + an 8B LLM) running together
slowed each other so badly that end-of-speech-to-translation went from ~3s to
~12s. So every model call in the process goes through this gate. Concurrency
still lives above it -- several sentences can each be at a different stage --
but the GPU only ever does one thing, and finals/translations jump ahead of
cosmetic partials.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from contextlib import asynccontextmanager

URGENT = 0
BACKGROUND = 1


class GpuGate:
    def __init__(self) -> None:
        self._busy = False
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._order = itertools.count()

    @property
    def urgent_waiting(self) -> bool:
        return any(p == URGENT and not f.done() for p, _, f in self._waiters)

    @asynccontextmanager
    async def hold(self, priority: int):
        await self._acquire(priority)
        try:
            yield
        finally:
            self._release()

    async def _acquire(self, priority: int) -> None:
        if not self._busy and not self._waiters:
            self._busy = True
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._order), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # We were handed the GPU just as we got cancelled: pass it on.
                self._release()
            raise

    def _release(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)  # ownership moves straight to the waiter
                return
        self._busy = False
