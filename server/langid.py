"""Decide whether a finished sentence was Indonesian or English (auto mode).

The OpenAI API is asked to detect the language itself, but its answer isn't
always usable:

- ``whisper-1`` reports it (as a full name, e.g. "indonesian"), yet Whisper
  regularly labels Indonesian as Malay, a closely related language.
- the ``gpt-*-transcribe`` models don't report a language at all.

So the API's label is a hint, and the transcript decides: common function
words separate the two languages reliably even for short sentences, and
English loanwords inside Indonesian speech ("deploy", "meeting") don't count
because they aren't function words.
"""

from __future__ import annotations

import re

_ID_WORDS = frozenset(
    """
    yang dan di ke dari ini itu dengan untuk tidak nggak gak enggak ga ada akan
    sudah udah belum saya aku kamu anda kita kami mereka dia ia gue gua lu lo
    apa apakah bagaimana gimana kenapa mengapa kapan dimana mana siapa berapa
    juga saja aja lagi sangat banget bisa harus mau ingin sedang lagi masih
    karena soalnya jadi kalau kalo tapi tetapi atau pada oleh dalam tentang
    seperti sama bagi kepada hari ya dong sih kan nih deh kok loh lah pun
    tolong terima kasih selamat pagi siang sore malam baik bapak ibu mas mbak
    adalah merupakan para semua setiap belum pernah nanti tadi besok kemarin
    """.split()
)
_EN_WORDS = frozenset(
    """
    the a an and or but of to in on at for with from by about as into is are
    was were be been being am do does did have has had will would can could
    should shall may might must not no yes this that these those there here
    i you he she it we they me him her us them my your his its our their
    what which who whom whose when where why how all any some very just also
    so than then too if because while until please thank thanks hello hi
    good morning afternoon evening let us okay go get got make know think
    don't can't won't didn't doesn't isn't i'm it's we're you're they're
    let's that's what's there's i'll we'll you'll i've we've i'd
    """.split()
)

# Labels from the API that mean one of our two languages.
_API_LANG = {
    "id": "id", "indonesian": "id", "indonesia": "id",
    "ms": "id", "malay": "id",  # Whisper's usual confusion for Indonesian
    "en": "en", "english": "en",
}

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def resolve_language(text: str, api_language: str | None = None) -> str | None:
    """Return "id", "en", or None when neither language is evident."""
    words = _WORD.findall(text.lower())
    id_score = sum(w in _ID_WORDS for w in words)
    en_score = sum(w in _EN_WORDS for w in words)
    hint = _API_LANG.get((api_language or "").strip().lower())

    if id_score != en_score:
        # Clear textual evidence wins, unless the API agrees anyway.
        return "id" if id_score > en_score else "en"
    return hint  # a tie (often very short text): trust the API if it said anything


def direction_for(lang: str) -> str:
    return "id-en" if lang == "id" else "en-id"
