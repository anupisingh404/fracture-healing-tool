from __future__ import annotations
import logging

from backend.config import settings

logger = logging.getLogger(__name__)


def search_medical_literature(
    fracture_location: str,
    healing_category: str,
    biomarker_summary: str,
    max_results: int = 3,
) -> list[str]:
    """
    Search Tavily for relevant medical literature about fracture healing.
    Returns a list of text snippets to inject into the GPT-4o prompt.
    Falls back to empty list if TAVILY_API_KEY is not set.
    """
    api_key = settings.tavily_api_key
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — skipping web search.")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        query = (
            f"{fracture_location} fracture healing prognosis BSAP ALP P1NP biomarkers "
            f"callus {healing_category} outcome prediction"
        )
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )

        snippets: list[str] = []
        if response.get("answer"):
            snippets.append(f"[Web summary] {response['answer']}")
        for result in response.get("results", [])[:max_results]:
            content = result.get("content", "").strip()
            if content:
                snippets.append(f"[{result.get('title', 'Source')}] {content[:400]}")
        return snippets

    except Exception as exc:
        logger.warning(f"Tavily search failed: {exc}")
        return []
