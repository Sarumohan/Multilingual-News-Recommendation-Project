"""
Configuration module for the Advanced Multimodal and Multilingual News Recommendation Framework.
Provides centralized path management, model checkpoints, and hyperparameter configuration.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"
TESTS_DIR = BASE_DIR / "tests"

# Data Subdirectories
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

# Database Configuration
DATABASE_PATH = DATABASE_DIR / "news_recommender.db"

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "as": "Assamese"
}

DEFAULT_LANGUAGE = "en"

# Multilingual Text Model Settings
# Lightweight yet powerful multilingual model with high cross-lingual semantic alignment
DEFAULT_TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALTERNATIVE_TEXT_MODEL = "xlm-roberta-base"
TEXT_EMBEDDING_DIM = 384  # 384 for MiniLM-L12, 768 for XLM-RoBERTa

# Vision / Image Feature Extraction Settings
DEFAULT_VISION_MODEL = "resnet50"
IMAGE_EMBEDDING_DIM = 2048  # ResNet50 pooled feature dimension
IMAGE_TARGET_SIZE = (224, 224)

# Multimodal Fusion Settings
DEFAULT_FUSION_WEIGHTS = {
    "text": 0.70,
    "image": 0.30
}

# Recommendation Engine Settings
DEFAULT_TOP_K = 10
COLLABORATIVE_WEIGHT = 0.35
CONTENT_WEIGHT = 0.65
TEMPORAL_DECAY_HALF_LIFE_DAYS = 3.0  # User interest half life in days
TEMPORAL_DECAY_LAMBDA = 0.23105       # ln(2)/3.0 per day
FRESHNESS_HALF_LIFE_DAYS = 5.0       # News freshness half life in days
FRESHNESS_LAMBDA = 0.13863           # ln(2)/5.0 per day
DEFAULT_FRESHNESS_WEIGHT = 0.10      # Freshness contribution in integrated score
DEFAULT_POPULARITY_DEBIAS_GAMMA = 0.10 # Popularity debiasing dampener
DIVERSITY_LAMBDA = 0.7               # Maximal Marginal Relevance (MMR) trade-off parameter (1.0 = pure relevance, 0.0 = pure diversity)
DEFAULT_MMR_LAMBDA = 0.70

# Interaction Weights for User Interest Modeling
DEFAULT_INTERACTION_WEIGHTS = {
    "click": 1.0,
    "read": 1.5,
    "like": 2.0,
    "bookmark": 3.0,
    "share": 3.5
}

# Create essential directory trees if they don't exist
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EMBEDDINGS_DIR,
    DATA_SAMPLE_DIR,
    DATABASE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

