"""
normalize.py
------------
Master dictionary for normalizing team names across all data sources.

The same national team appears with different names depending on the source:
    - Kaggle results: "South Korea"
    - FIFA Rankings:  "Korea Republic"
    - StatsBomb:      "South Korea"

Strategy:
    - Only names that DIFFER between datasets need an alias entry.
    - Names that are consistent across datasets pass through unchanged.
    - WARNING is only logged for genuinely ambiguous/unknown cases.

Usage:
    from world_cup_2026.data_ingestion.normalize import normalize_team_name
    df["team"] = df["team"].map(normalize_team_name)
"""

from loguru import logger

# ---------------------------------------------------------------------------
# Alias dictionary — ONLY cross-dataset conflicts go here
# key   = alias as it appears in ONE source (lowercase)
# value = canonical name used consistently across most sources
# ---------------------------------------------------------------------------

_TEAM_ALIASES: dict[str, str] = {
    # Korea
    "korea republic":               "South Korea",
    "korea, republic of":           "South Korea",
    "korea rep.":                   "South Korea",
    "north korea":                  "Korea DPR",
    "korea dpr":                    "Korea DPR",
    "korea, dem. people's rep.":    "Korea DPR",

    # Iran
    "ir iran":                      "Iran",
    "islamic republic of iran":     "Iran",

    # China
    "china pr":                     "China",
    "china, pr":                    "China",

    # Turkey
    "türkiye":                      "Turkey",
    "turkiye":                      "Turkey",

    # Czechia
    "czech republic":               "Czechia",
    "the czech republic":           "Czechia",

    # Slovakia
    "slovak republic":              "Slovakia",

    # North Macedonia
    "north macedonia":              "North Macedonia",
    "macedonia":                    "North Macedonia",
    "republic of north macedonia":  "North Macedonia",

    # Bosnia
    "bosnia and herzegovina":       "Bosnia-Herzegovina",
    "bosnia & herzegovina":         "Bosnia-Herzegovina",

    # USA
    "united states":                "USA",
    "united states of america":     "USA",

    # Trinidad
    "trinidad & tobago":            "Trinidad and Tobago",

    # Ivory Coast
    "ivory coast":                  "Côte d'Ivoire",
    "cote d'ivoire":                "Côte d'Ivoire",

    # DR Congo
    "democratic republic of the congo": "DR Congo",
    "congo dr":                     "DR Congo",

    # Eswatini
    "swaziland":                    "Eswatini",

    # Kyrgyzstan
    "kyrgyzstan":                   "Kyrgyz Republic",

    # UAE
    "uae":                          "United Arab Emirates",

    # Russia
    "russian federation":           "Russia",

    # Germany (historical)
    "west germany":                 "Germany",

    # Republic of Ireland
    "republic of ireland":          "Ireland",

    # Curaçao
    "curaçao": "Curacao",

    # Cabo Verde
    "cabo verde":    "Cape Verde",
}

# Names that are already canonical — no transformation, no warning
# This is the complete set of names used consistently across our datasets
_PASSTHROUGH_NAMES: frozenset[str] = frozenset({
    "afghanistan", "albania", "algeria", "andorra", "angola",
    "argentina", "armenia", "australia", "austria", "azerbaijan",
    "bahrain", "bangladesh", "belgium", "bolivia", "botswana",
    "brazil", "bulgaria", "burkina faso", "burundi", "cabo verde",
    "cambodia", "cameroon", "canada", "chile", "china", "colombia",
    "comoros", "costa rica", "croatia", "cuba", "cyprus", "czechia",
    "denmark", "djibouti", "ecuador", "egypt", "el salvador",
    "england", "eritrea", "estonia", "ethiopia", "fiji", "finland",
    "france", "gabon", "georgia", "germany", "ghana", "gibraltar",
    "greece", "guatemala", "guinea", "guinea-bissau", "guyana",
    "haiti", "honduras", "hungary", "iceland", "india", "indonesia",
    "iran", "iraq", "ireland", "israel", "italy", "jamaica", "japan",
    "jordan", "kazakhstan", "kenya", "kosovo", "kuwait", "latvia",
    "lebanon", "lesotho", "liberia", "libya", "liechtenstein",
    "lithuania", "luxembourg", "madagascar", "malawi", "malaysia",
    "maldives", "mali", "malta", "mauritania", "mauritius", "mexico",
    "moldova", "mongolia", "montenegro", "morocco", "mozambique",
    "myanmar", "namibia", "nepal", "netherlands", "new zealand",
    "nicaragua", "niger", "nigeria", "north korea", "northern ireland",
    "norway", "oman", "pakistan", "palestine", "panama", "paraguay",
    "peru", "philippines", "poland", "portugal", "qatar", "romania",
    "russia", "rwanda", "san marino", "saudi arabia", "scotland",
    "senegal", "serbia", "singapore", "slovakia", "slovenia",
    "somalia", "south africa", "south korea", "south sudan", "spain",
    "sri lanka", "sudan", "suriname", "sweden", "switzerland",
    "tajikistan", "tanzania", "thailand", "togo", "tunisia",
    "turkey", "turkmenistan", "uganda", "ukraine", "uruguay",
    "usa", "uzbekistan", "venezuela", "vietnam", "wales",
    "yemen", "zambia", "zimbabwe",
    # WC2026 specific
    "curaçao", "côte d'ivoire", "dr congo", "eswatini",
    "north macedonia", "bosnia-herzegovina", "kyrgyz republic",
    "united arab emirates", "trinidad and tobago",
    "ir iran", "korea republic", "algeria", "austria", "jordan",
    "colombia", "ecuador", "ghana", "panama", "croatia",
    "norway", "senegal", "uzbekistan", "cabo verde",
    "american samoa", "anguilla", "antigua and barbuda", "aruba",
    "bahamas", "barbados", "belarus", "belize", "benin", "bermuda",
    "bhutan", "british virgin islands", "brunei darussalam",
    "cayman islands", "central african republic", "chad",
    "chinese taipei", "congo", "cook islands", "curacao",
    "czechoslovakia", "dominica", "dominican republic",
    "equatorial guinea", "faroe islands", "grenada", "guam",
    "hong kong", "laos", "macau", "montserrat",
    "netherlands antilles", "new caledonia", "papua new guinea",
    "puerto rico", "samoa", "sao tome and principe",
    "serbia and montenegro", "seychelles", "sierra leone",
    "solomon islands", "st kitts and nevis", "st lucia",
    "st vincent and the grenadines", "syria", "tahiti",
    "the gambia", "timor-leste", "tonga",
    "turks and caicos islands", "us virgin islands",
    "vanuatu", "yugoslavia", "zaire", "cape verde"
})

# Tracks unknown names already warned about — avoids log spam
_warned_unknown: set[str] = set()

def normalize_team_name(name: str) -> str:
    """Normalize a team name to its canonical form.

    Args:
        name: Raw team name string from any data source.

    Returns:
        Canonical team name. Passthrough if already canonical.
        WARNING only for genuinely unknown names.
    """
    if not isinstance(name, str):
        return name

    stripped = name.strip()
    key = stripped.lower()

    # Level 1 — explicit alias (cross-dataset conflict resolution)
    if key in _TEAM_ALIASES:
        return _TEAM_ALIASES[key]

    # Level 2 — already canonical or known consistent name, passthrough silently
    if key in _PASSTHROUGH_NAMES:
        return stripped

    # Level 3 — unknown, return as-is but warn once for manual review
    if stripped not in _warned_unknown:
        _warned_unknown.add(stripped)
        logger.warning(f"Unknown team name: '{stripped}' — add to normalize.py if needed.")
    return stripped
  

def normalize_dataframe_teams(df, columns: list[str]):
    """Apply normalize_team_name to one or more DataFrame columns.

    Args:
        df: pandas DataFrame.
        columns: List of column names to normalize.

    Returns:
        Copy of DataFrame with normalized team name columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(normalize_team_name)
            logger.info(f"Normalized column: '{col}'")
        else:
            logger.warning(f"Column '{col}' not found — skipping.")
    return df