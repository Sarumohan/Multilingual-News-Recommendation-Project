"""
Multilingual Text Normalization and Preprocessing Module.
Preserves Unicode script integrity for English (Latin), Hindi (Devanagari), and Assamese (Eastern Nagari/Bengali-Assamese).
"""

import re
import html
from typing import Optional, Dict, Tuple

class MultilingualTextCleaner:
    """Cleans and normalizes multilingual news text while preserving script characteristics."""

    # Unicode ranges for script detection
    SCRIPT_RANGES = {
        "hi": (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, etc.)
        "as": (0x0980, 0x09FF),  # Eastern Nagari / Bengali-Assamese
        "en": (0x0020, 0x007E),  # Basic Latin / ASCII
    }

    # Distinctive Assamese characters in Eastern Nagari script
    # \u09F0 = ৱ (va), \u09F1 = ৰ (ra) - specifically used in Assamese orthography
    ASSAMESE_SPECIFIC_CHARS = {"\u09F0", "\u09F1", "ৱ", "ৰ"}

    def __init__(self):
        # Regex for removing HTML tags
        self.html_tag_re = re.compile(r"<[^>]+>")
        # Regex for URLs
        self.url_re = re.compile(r"https?://\S+|www\.\S+")
        # Regex for multiple whitespaces
        self.whitespace_re = re.compile(r"\s+")

    def clean_text(self, text: Optional[str], preserve_case: bool = False) -> str:
        """
        Normalize and clean news text.
        Removes HTML tags, URLs, extra whitespaces, while preserving native scripts.
        """
        if not text or not isinstance(text, str):
            return ""

        # Unescape HTML entities (e.g., &amp;, &quot;)
        cleaned = html.unescape(text)

        # Remove HTML markup
        cleaned = self.html_tag_re.sub(" ", cleaned)

        # Remove URLs
        cleaned = self.url_re.sub(" ", cleaned)

        # Remove zero-width spaces and non-printable control characters (except standard newlines/tabs)
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)

        # Standardize whitespace
        cleaned = self.whitespace_re.sub(" ", cleaned).strip()

        # Lowercase if not preserving case (primarily affects Latin/English)
        if not preserve_case:
            cleaned = cleaned.lower()

        return cleaned

    def detect_script_language(self, text: str) -> str:
        """
        Identify predominant language/script of the text ('en', 'hi', 'as').
        Analyzes character distribution across Unicode code points.
        """
        if not text or not isinstance(text, str):
            return "unknown"

        counts = {"hi": 0, "as": 0, "en": 0, "other": 0}
        has_assamese_specific = any(char in self.ASSAMESE_SPECIFIC_CHARS for char in text)

        for char in text:
            cp = ord(char)
            if self.SCRIPT_RANGES["hi"][0] <= cp <= self.SCRIPT_RANGES["hi"][1]:
                counts["hi"] += 1
            elif self.SCRIPT_RANGES["as"][0] <= cp <= self.SCRIPT_RANGES["as"][1]:
                counts["as"] += 1
            elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:  # A-Z, a-z
                counts["en"] += 1
            elif not char.isspace() and not char.isdigit():
                counts["other"] += 1

        total_script_chars = counts["hi"] + counts["as"] + counts["en"]
        if total_script_chars == 0:
            return "en"

        # If Eastern Nagari script is dominant
        if counts["as"] > counts["hi"] and counts["as"] > counts["en"]:
            # Assamese shares the Unicode Eastern Nagari block with Bengali
            return "as"
        
        # If Hindi/Devanagari is dominant
        if counts["hi"] > counts["as"] and counts["hi"] > counts["en"]:
            return "hi"

        # If Assamese specific letters are present even in mixed text
        if has_assamese_specific:
            return "as"

        # Default to English if Latin characters dominate
        if counts["en"] >= counts["hi"] and counts["en"] >= counts["as"]:
            return "en"

        return "en"

    def prepare_article_representation(
        self,
        title: Optional[str],
        abstract: Optional[str] = None,
        content: Optional[str] = None,
        max_content_chars: int = 500
    ) -> str:
        """
        Synthesize preprocessed composite representation for multimodal & transformer embeddings.
        Combines title, abstract, and head content.
        """
        clean_title = self.clean_text(title)
        clean_abstract = self.clean_text(abstract)
        
        parts = [clean_title]
        if clean_abstract and clean_abstract != clean_title:
            parts.append(clean_abstract)
            
        if content and len(parts) < 2:
            clean_content = self.clean_text(content[:max_content_chars])
            if clean_content:
                parts.append(clean_content)

        return " ".join(parts).strip()

