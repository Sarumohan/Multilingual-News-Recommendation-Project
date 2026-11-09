"""
Article Card Component for Streamlit Dashboard.
Renders attractive, theme-adaptive cards with standardized image aspect ratios,
multilingual badges, interaction buttons (Like, Bookmark, Read), and XAI explanations
without relying on escaped HTML strings.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import streamlit as st
from PIL import Image, ImageOps

from config.config import SUPPORTED_LANGUAGES, BASE_DIR

@st.cache_data
def get_card_image(article_id: str, category: str) -> Image.Image:
    """
    Loads local article image, resizes and crops to exact (400, 240) aspect ratio.
    If image is missing or corrupted, returns category-themed placeholder.
    Guarantees exact uniform dimensions across all cards.
    """
    target_size = (400, 240)
    local_path = BASE_DIR / "data" / "images" / f"{article_id}.jpg"
    if local_path.exists():
        try:
            im = Image.open(local_path).convert("RGB")
            return ImageOps.fit(im, target_size, Image.Resampling.LANCZOS)
        except Exception:
            pass

    # Try category placeholder
    cat_lower = category.lower()
    cat_placeholder = BASE_DIR / "data" / "images" / f"placeholder_{cat_lower}.jpg"
    if cat_placeholder.exists():
        try:
            im = Image.open(cat_placeholder).convert("RGB")
            return ImageOps.fit(im, target_size, Image.Resampling.LANCZOS)
        except Exception:
            pass

    # Default fallback placeholder
    default_placeholder = BASE_DIR / "data" / "images" / "placeholder_default.jpg"
    if default_placeholder.exists():
        try:
            im = Image.open(default_placeholder).convert("RGB")
            return ImageOps.fit(im, target_size, Image.Resampling.LANCZOS)
        except Exception:
            pass

    # In-memory dark fallback
    return Image.new("RGB", target_size, color="#1e293b")


def render_article_card(
    article: Dict[str, Any],
    score: Optional[float] = None,
    explanation: Optional[str] = None,
    on_like: Optional[Callable[[str], None]] = None,
    on_bookmark: Optional[Callable[[str], None]] = None,
    on_read: Optional[Callable[[str], None]] = None,
    key_prefix: str = "card"
):
    """
    Renders an article card inside a theme-native bordered container.
    Guarantees no raw HTML escapes, consistent image dimensions, and aligned action buttons.
    """
    aid = article.get("article_id", "")
    title = article.get("title", "Untitled")
    abstract = article.get("abstract", article.get("title", ""))
    category = article.get("category", "General").title()
    lang_code = article.get("language", "en")
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
    published_at = article.get("published_at", "Recent")

    with st.container(border=True):
        # 1. Top Metadata Bar: Badges & Relevance Score
        b_col1, b_col2 = st.columns([3, 1])
        with b_col1:
            st.caption(f":blue-background[**{lang_name}**] :gray-background[**{category}**] • 🕒 {published_at}")
        with b_col2:
            if score is not None:
                st.caption(f":green-background[**Match: {score:.2f}**]")

        # 2. Standardized Image (400x240 Aspect Ratio)
        img = get_card_image(aid, category)
        st.image(img, use_container_width=True)

        # 3. Article Title (bold, wrapped)
        st.markdown(f"**{title}**")

        # 4. Description snippet
        if abstract and abstract != title:
            desc_text = f"{abstract[:160]}..." if len(abstract) > 160 else abstract
            st.caption(desc_text)

        # 5. XAI Explanation (if provided)
        if explanation:
            st.info(f"💡 {explanation}")

        # 6. Horizontally Aligned Action Buttons
        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            if st.button("👍 Like", key=f"{key_prefix}_like_{aid}", use_container_width=True):
                if on_like:
                    on_like(aid)
        with btn2:
            if st.button("🔖 Save", key=f"{key_prefix}_bm_{aid}", use_container_width=True):
                if on_bookmark:
                    on_bookmark(aid)
        with btn3:
            if st.button("👁️ Read", key=f"{key_prefix}_read_{aid}", use_container_width=True):
                if on_read:
                    on_read(aid)
