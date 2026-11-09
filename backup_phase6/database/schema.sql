-- Schema definition for Multimodal Multilingual News Recommendation Framework
-- SQLite Compatible Schema

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS news_articles (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    content TEXT,
    language TEXT NOT NULL,          -- 'en', 'hi', 'as'
    category TEXT NOT NULL,          -- e.g., 'technology', 'politics', 'sports', 'business', 'science'
    subcategory TEXT,
    source_name TEXT,
    source_url TEXT,
    image_url TEXT,
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    preferred_languages TEXT NOT NULL,  -- JSON list e.g. '["en", "hi", "as"]'
    preferred_categories TEXT NOT NULL, -- JSON list e.g. '["technology", "sports"]'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reading_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    read_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reading_duration_seconds INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES news_articles(article_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL,     -- 'click', 'like', 'share', 'bookmark'
    rating REAL DEFAULT 1.0,            -- Implicit/explicit weight
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES news_articles(article_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendation_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    recommendation_algorithm TEXT NOT NULL, -- 'content_based', 'collaborative', 'hybrid', 'cross_lingual', etc.
    relevance_score REAL NOT NULL,
    diversity_score REAL,
    explanation TEXT,
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES news_articles(article_id) ON DELETE CASCADE
);

-- Indices for fast querying during recommendation lookups
CREATE INDEX IF NOT EXISTS idx_news_lang_cat ON news_articles (language, category);
CREATE INDEX IF NOT EXISTS idx_history_user ON reading_history (user_id, read_timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON user_interactions (user_id, interaction_type);
CREATE INDEX IF NOT EXISTS idx_recom_logs_user ON recommendation_logs (user_id, recommended_at);

