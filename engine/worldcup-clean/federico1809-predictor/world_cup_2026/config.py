from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR = PROJ_ROOT / "models"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = PROJ_ROOT / "outputs" / "figures"
OUTPUTS_DIR = PROJ_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Create dirs if they don't exist
# ---------------------------------------------------------------------------
for _dir in [FIGURES_DIR, PREDICTIONS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
