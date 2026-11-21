"""Search saved tweets with hybrid embeddings or optional keyword retrieval."""

from collections import Counter, defaultdict
import csv
from functools import lru_cache
import math
import os
import re
import unicodedata

from src.config import PROJECT_ROOT

STOP_WORDS = frozenset(
    "a an and are as at be been by can could did do does for from had has have "
    "he her him his how i in is it its me my of on or please she that the their "
    "them there they this to was were what when where which who will with would "
    "you your about tell said says say news latest update updates tweet tweets".split()
)


def tokenize(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return [term for term in re.findall(r"[^\W_]+", normalized)
            if len(term) > 1 and term not in STOP_WORDS]


@lru_cache(maxsize=2)
def load_keyword_index(path, modified_at):
    # The timestamp invalidates the cache when the scraped CSV changes.
    documents = []
    postings = defaultdict(list)
    lengths = []
    with open(path, newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if not row.get("text") or not row.get("url"):
                continue
            document_id = len(documents)
            documents.append(row)
            terms = Counter(tokenize(row["text"]))
            lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                postings[term].append((document_id, frequency))
    return documents, postings, lengths, sum(lengths) / max(len(documents), 1) or 1


def keyword_search(query, k=3):
    path = PROJECT_ROOT / "data" / "fabrizioromano_tweets.csv"
    documents, postings, lengths, average_length = load_keyword_index(
        str(path), path.stat().st_mtime_ns
    )
    scores = defaultdict(float)
    for term in set(tokenize(query)):
        matches = postings.get(term, [])
        inverse_frequency = math.log(1 + (len(documents) - len(matches) + 0.5) / (len(matches) + 0.5))
        for document_id, frequency in matches:
            length_weight = 1.2 * (0.25 + 0.75 * lengths[document_id] / average_length)
            scores[document_id] += inverse_frequency * frequency * 2.2 / (frequency + length_weight)
    selected = sorted(scores, key=lambda item: (-scores[item], item))[:max(k, 0)]
    results = [documents[item] for item in selected]
    context = [
        f"On {row.get('created at') or row.get('created_at', '')}, Fabrizio Romano tweeted: "
        f"{row['text']}, Tweet URL: {row['url']}"
        for row in results
    ]
    return context, [row["url"] for row in results]


def get_retrieval_mode():
    mode = (os.getenv("RETRIEVAL_MODE") or "").strip().lower() or "hybrid"
    if mode not in ("keyword", "hybrid"):
        raise ValueError("RETRIEVAL_MODE must be 'keyword' or 'hybrid'.")
    return mode


def search_tweets(query, k=3):
    if get_retrieval_mode() == "keyword":
        return keyword_search(query, k)

    # Load embedding and reranking models on the first search, not for greetings
    # or while rendering the initial page.
    from init_lancedb import initialize_database
    from src.vectordb import lancedb_hybrid_search

    initialize_database()
    return lancedb_hybrid_search(query, k)


def get_greeting_response(query):
    normalized = re.sub(r"[^\w\s]", "", query.casefold()).strip()
    normalized = " ".join(normalized.split())
    if normalized in ("hi", "hi there", "hello", "hello there", "hey", "hey there",
                      "good morning", "good afternoon", "good evening"):
        return "Hi! Ask me about a player, club, or transfer and I'll search the saved tweets."
    return None
