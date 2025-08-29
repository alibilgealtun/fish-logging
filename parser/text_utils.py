"""Text utility functions for the fish parser."""

import re
from typing import List


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text while preserving hyphenated words, then split them.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase tokens
    """
    # Find all words, including hyphenated ones
    tokens = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)?", text)

    # Split hyphenated tokens
    split_tokens = []
    for token in tokens:
        if "-" in token:
            # Split on hyphens and add non-empty parts
            parts = token.split("-")
            for part in parts:
                if part:
                    split_tokens.append(part.lower())
        else:
            split_tokens.append(token.lower())

    return split_tokens