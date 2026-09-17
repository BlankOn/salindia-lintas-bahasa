"""Slide decks: an uploaded PDF, rendered page by page to JPEG with PDFium.

Rendering happens server-side so the UI needs no PDF library and everything
works offline. PDFium is not thread-safe, so every call runs on one dedicated
thread. It is CPU work and does not touch the GPU gate, so a page turn never
waits behind speech or translation.
"""

from __future__ import annotations

import asyncio
import io
import logging
import secrets
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_BYTES = 200 * 1024 * 1024
MAX_DECKS = 3            # decks kept in memory; oldest is dropped
MAX_CACHED_PAGES = 120   # rendered images kept per deck
WIDTH_STEP = 160         # widths are bucketed so resizes don't defeat the cache
MIN_WIDTH, MAX_WIDTH = 320, 3840


class SlideError(ValueError):
    pass


@dataclass
class Deck:
    id: str
    name: str
    doc: object  # pypdfium2.PdfDocument
    sizes: list[tuple[float, float]]
    rendered: OrderedDict = field(default_factory=OrderedDict)

    def meta(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pages": len(self.sizes),
            "sizes": [[round(w, 1), round(h, 1)] for w, h in self.sizes],
        }


def bucket_width(width: int) -> int:
    width = max(MIN_WIDTH, min(MAX_WIDTH, width))
    return -(-width // WIDTH_STEP) * WIDTH_STEP  # round up


class SlideStore:
    def __init__(self) -> None:
        self._decks: OrderedDict[str, Deck] = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdfium")

    # -- blocking side, on the pdfium thread ---------------------------------

    def _open(self, data: bytes, name: str) -> Deck:
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            raise SlideError(f"not a readable PDF ({exc})") from exc
        if len(doc) == 0:
            doc.close()
            raise SlideError("the PDF has no pages")
        sizes = [doc[i].get_size() for i in range(len(doc))]

        deck = Deck(id=secrets.token_urlsafe(9), name=name, doc=doc, sizes=sizes)
        self._decks[deck.id] = deck
        while len(self._decks) > MAX_DECKS:
            _, old = self._decks.popitem(last=False)
            old.doc.close()
        return deck

    def _render(self, deck: Deck, index: int, width: int) -> bytes:
        key = (index, width)
        if key in deck.rendered:
            deck.rendered.move_to_end(key)
            return deck.rendered[key]

        started = time.perf_counter()
        page_w, _ = deck.sizes[index]
        page = deck.doc[index]
        try:
            image = page.render(scale=width / page_w).to_pil()
        finally:
            page.close()
        buf = io.BytesIO()
        # subsampling=0 keeps coloured text edges crisp; JPEG is ~100x faster
        # to encode than PNG at slide sizes.
        image.convert("RGB").save(buf, "JPEG", quality=92, subsampling=0)
        data = buf.getvalue()

        deck.rendered[key] = data
        while len(deck.rendered) > MAX_CACHED_PAGES:
            deck.rendered.popitem(last=False)
        log.debug(
            "rendered %s p%d @%dpx in %.0fms",
            deck.id, index + 1, width, (time.perf_counter() - started) * 1000,
        )
        return data

    # -- async API ------------------------------------------------------------

    async def add(self, data: bytes, name: str) -> Deck:
        if len(data) > MAX_BYTES:
            raise SlideError("PDF is larger than 200 MB")
        if not data.startswith(b"%PDF"):
            raise SlideError("that file is not a PDF")
        loop = asyncio.get_running_loop()
        deck = await loop.run_in_executor(self._pool, self._open, data, name)
        log.info("slides %s: %r, %d pages", deck.id, name, len(deck.sizes))
        return deck

    def get(self, deck_id: str) -> Deck | None:
        return self._decks.get(deck_id)

    async def page(self, deck_id: str, number: int, width: int) -> bytes:
        deck = self._decks.get(deck_id)
        if deck is None:
            raise KeyError(deck_id)
        if not 1 <= number <= len(deck.sizes):
            raise SlideError(f"page {number} is out of range")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, self._render, deck, number - 1, bucket_width(width)
        )

    def shutdown(self) -> None:
        for deck in self._decks.values():
            deck.doc.close()
        self._pool.shutdown(wait=False, cancel_futures=True)
