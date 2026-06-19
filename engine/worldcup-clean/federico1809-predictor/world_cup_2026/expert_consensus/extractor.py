"""
extractor.py
------------
Use the Google Gemini API to extract structured football predictions
from article text.

IMPORTANT: GEMINI_API_KEY must be set as an environment variable before
running this module. Copy .env.example to .env and fill in your key, or
export it in your shell: export GEMINI_API_KEY=AIza-...
"""
from __future__ import annotations

import json
import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

load_dotenv()

from world_cup_2026.config import RAW_DATA_DIR
from world_cup_2026.data_ingestion.normalize import normalize_team_name

# ---------------------------------------------------------------------------
# WC2026 qualified teams — loaded once at import time from the fixture CSV
# ---------------------------------------------------------------------------
_TEAMS_CSV = RAW_DATA_DIR / "areezvisram12_fixture" / "teams.csv"


def _load_wc2026_teams() -> frozenset[str]:
    df = pd.read_csv(_TEAMS_CSV)
    qualified = df[df["is_placeholder"] == False]  # noqa: E712
    return frozenset(normalize_team_name(name) for name in qualified["team_name"])


WC2026_TEAMS: frozenset[str] = _load_wc2026_teams()

# ---------------------------------------------------------------------------
# Validation sets
# ---------------------------------------------------------------------------
_VALID_PREDICTION_TYPES = frozenset({
    "champion", "finalist", "semifinalist", "group_exit",
    "surprise", "disappointment", "strong", "weak",
    "general_positive", "general_negative",
})
_VALID_CONFIDENCES = frozenset({"certain", "likely", "possible", "unlikely"})

_MODEL = "models/gemini-2.5-flash"
_MAX_TEXT_CHARS = 12_000

_SYSTEM_PROMPT = (
    "You are a football (soccer) analyst. Given an article about the 2026 FIFA World Cup, "
    "extract every concrete prediction about national teams.\n\n"
    "Return ONLY a valid JSON array — no markdown, no explanation, no preamble.\n"
    "Each element must follow this exact schema:\n"
    "{\n"
    '  "team": "<national team name>",\n'
    '  "prediction_type": "<champion|finalist|semifinalist|group_exit|surprise|'
    'disappointment|strong|weak|general_positive|general_negative>",\n'
    '  "confidence": "<certain|likely|possible|unlikely>",\n'
    '  "quote": "<exact supporting quote from the article, max 50 words>",\n'
    '  "sentiment": <float from -1.0 to 1.0>\n'
    "}\n\n"
    "If no predictions are found, return an empty array: []"
)


def extract_predictions(article: dict) -> list[dict]:
    """Extract structured team predictions from a scraped article dict.

    Calls the Gemini API (reads GEMINI_API_KEY from environment),
    parses the JSON response, normalizes team names, and filters out teams
    not participating in WC2026.

    Args:
        article: dict returned by scraper.scrape_url — must contain
                 url, source_domain, title, date, text.

    Returns:
        List of prediction dicts. Each dict has fields:
        team, prediction_type, confidence, sentiment, quote,
        source, date, url.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))  # reads GEMINI_API_KEY from environment

    text_snippet = article["text"][:_MAX_TEXT_CHARS]
    user_message = f"Article title: {article['title']}\n\n{text_snippet}"

    logger.info(f"Extracting predictions from: {article['source_domain']}")

    try:
        response = client.models.generate_content(
            model=_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
        )
        raw_json = response.text.strip()
        # Strip markdown code fences that some model versions add despite instructions
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[-1]
            raw_json = raw_json.rsplit("```", 1)[0].strip()
    except Exception as e:
        logger.error(f"Gemini API error for {article['url']}: {e}")
        return []

    try:
        predictions = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e} — raw output: {raw_json[:500]}")
        return []

    if not isinstance(predictions, list):
        logger.warning("Model returned non-list JSON — skipping.")
        return []

    results = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue

        missing = [f for f in ("team", "prediction_type", "confidence", "quote", "sentiment")
                   if f not in pred]
        if missing:
            logger.debug(f"Prediction missing fields {missing} — skipping: {pred}")
            continue

        canonical = normalize_team_name(str(pred["team"]))
        if canonical not in WC2026_TEAMS:
            logger.debug(f"Team '{canonical}' not in WC2026 roster — skipping.")
            continue

        if pred["prediction_type"] not in _VALID_PREDICTION_TYPES:
            logger.debug(f"Unknown prediction_type '{pred['prediction_type']}' — skipping.")
            continue

        if pred["confidence"] not in _VALID_CONFIDENCES:
            logger.debug(f"Unknown confidence '{pred['confidence']}' — skipping.")
            continue

        try:
            sentiment = float(pred["sentiment"])
        except (TypeError, ValueError):
            sentiment = 0.0
        sentiment = max(-1.0, min(1.0, sentiment))

        results.append({
            "team": canonical,
            "prediction_type": pred["prediction_type"],
            "confidence": pred["confidence"],
            "sentiment": sentiment,
            "quote": str(pred["quote"])[:300],
            "source": article["source_domain"],
            "date": article["date"],
            "url": article["url"],
        })

    logger.info(
        f"Extracted {len(results)} valid predictions from {article['source_domain']}"
    )
    return results
