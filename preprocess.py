"""
Text preprocessing module.
Cleans email texts and prepares them for the model.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def download_nltk_resources():
    """Downloads the required NLTK resources."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)


class TextPreprocessor:
    """
    Cleans and normalizes email texts.

    Steps:
    - Lowercasing
    - Removing URLs, numbers and punctuation
    - Removing stopwords
    - Stemming (optional)
    """

    def __init__(self, use_stemming: bool = False, language: str = "english"):
        """
        Args:
            use_stemming: Whether to apply stemming.
            language: Stopword language.
        """
        download_nltk_resources()
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.use_stemming = use_stemming

    def clean(self, text: str) -> str:
        """
        Cleans a single text.

        Args:
            text: Raw email text.

        Returns:
            Cleaned text.

        Time Complexity: O(n), where n is the number of characters.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords and optionally stem
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def clean_batch(self, texts: list) -> list:
        """
        Cleans a list of texts.

        Args:
            texts: List of raw texts.

        Returns:
            List of cleaned texts.

        Time Complexity: O(n * m), where n is the number of texts and m the average length.
        """
        return [self.clean(text) for text in texts]
