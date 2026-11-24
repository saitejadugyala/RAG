"""STEP 1 - Prepare the question for searching.

The question is turned into two things:
  * a vector (1536 numbers) with the SAME embedding model used at ingestion,
    for semantic search - finds chunks with the same meaning
  * a list of keywords (common words like "the", "how" removed),
    for keyword search - finds exact terms such as "4400" or "Contoso"
"""
from __future__ import annotations

import re

from ingestion.settings import Settings

from .errors import QuestionError
from .models import Query

TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")  # keeps "99.9", "text-embedding-3-small"

STOPWORDS = set("""
a an and are as at be been but by can could did do does for from get got had has have how
i if in into is it its me my of on or our should so than that the their them then there these
they this to was we were what when where which who why will with would you your yours about
any all also just more most much many may might must no not only other some such very
""".split())


def keywords(text: str) -> list[str]:
    """Lower-case words without stopwords, with a light plural fix (days -> day)."""
    words = []
    for word in TOKEN.findall(text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss") and word.isalpha():
            word = word[:-1]
        words.append(word)
    return words


def prepare_question(question: str, embedder, settings: Settings) -> Query:
    question = " ".join(question.split())  # tidy whitespace
    if len(question) < 3:
        raise QuestionError("The question is empty or too short.")
    if len(question) > 2000:
        raise QuestionError("The question is too long (max 2000 characters).")
    try:
        vector = embedder.embed([question])[0]
    except Exception as e:
        raise QuestionError(f"Could not embed the question: {e}") from e
    return Query(text=question, vector=vector, keywords=keywords(question))
