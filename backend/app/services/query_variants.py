from __future__ import annotations

import re

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "in",
    "of",
    "on",
    "the",
    "to",
    "using",
    "via",
    "what",
    "with",
}
_ACRONYMS = {
    "ai": "artificial intelligence",
    "llm": "large language model",
    "ml": "machine learning",
    "nlp": "natural language processing",
}


def _singularize(token: str) -> str:
    lower = token.casefold()
    if len(token) > 4 and lower.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


class QueryVariantService:
    """Small, deterministic lexical reformulations; no query-topic special cases."""

    max_variants = 4

    def generate(self, query: str) -> list[str]:
        original = " ".join(query.split())
        variants = [original]
        seen = {original.casefold()}

        def add(value: str) -> None:
            normalized = " ".join(value.split())
            if (
                normalized
                and normalized.casefold() not in seen
                and len(variants) < self.max_variants
            ):
                variants.append(normalized)
                seen.add(normalized.casefold())

        terms = [
            _singularize(token)
            for token in re.findall(r"[^\W_]+(?:-[^\W_]+)*", original)
            if token.casefold() not in _STOP_WORDS
        ]
        if not terms:
            return variants

        concise = " ".join(terms)
        add(concise)
        # Keep the query's subject and topic while removing an intervening modifier.
        compressed = [terms[0], *terms[2:]] if len(terms) >= 3 else terms
        add(" ".join(compressed))
        expanded = [_ACRONYMS.get(term.casefold(), term) for term in compressed]
        add(" ".join(expanded))
        return variants
