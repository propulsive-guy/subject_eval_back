""""
This gives the Thematic Scores Module
"""

"""
Inbuilt Modules
"""
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ThematicAnalyzer:
    """
    A singleton class to analyze the thematic similarity of two texts.

    It works by normalizing text into a "bag of words," where synonyms are
    standardized to a root word. This allows for comparing the core themes
    of a text, even if different vocabulary is used.

    The singleton pattern ensures resources and synonym maps are
    initialized only once.
    """

    _instance = None

    # A pre-defined list of synonym groups to standardize common words.
    SYNONYM_GROUPS = [
        {"quick", "fast", "rapid", "speedy", "swift", "prompt"},
        {"intelligent", "smart", "clever", "bright", "brilliant", "wise"},
        {"happy", "joyful", "cheerful", "glad", "delighted", "content", "pleased"},
        {"sad", "unhappy", "sorrowful", "depressed", "miserable", "down"},
        {"angry", "mad", "furious", "irate", "annoyed", "outraged"},
        {"car", "automobile", "vehicle", "ride"},
        {"job", "work", "occupation", "profession", "career"},
        {"house", "home", "residence", "dwelling", "abode"},
        {"big", "large", "huge", "gigantic", "massive", "enormous"},
        {"small", "little", "tiny", "miniature", "petite"},
        {"start", "begin", "commence", "initiate", "launch"},
        {"end", "finish", "conclude", "terminate", "complete"},
        {"run", "sprint", "jog", "dash"},
        {"walk", "stroll", "saunter", "amble"},
        {"see", "observe", "view", "watch", "spot", "glimpse"},
        {"say", "tell", "speak", "utter", "state", "declare"},
        {"think", "ponder", "consider", "reflect", "contemplate"},
        {"eat", "consume", "devour", "ingest", "feast"},
        {"help", "assist", "aid", "support", "serve"},
        {"buy", "purchase", "acquire", "obtain", "procure"},
        {"beautiful", "pretty", "gorgeous", "lovely", "attractive", "stunning"},
        {"ugly", "unattractive", "hideous", "unsightly"},
        {"important", "significant", "crucial", "vital", "essential"},
        {"hard", "difficult", "challenging", "tough"},
        {"easy", "simple", "effortless", "straightforward"},
    ]

    def __new__(cls, *args, **kwargs):
        """Implements the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ThematicAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes synonym map only once."""
        if not hasattr(self, 'initialized'):
            self.stop_words = {
                "the","is","in","and","or","of","to","a","that","for","on","with",
                "as","was","were","by","an","it","be","this","are"
            }
            self.synonym_map = {
                word: sorted(group)[0]
                for group in self.SYNONYM_GROUPS
                for word in group
            }
            self.initialized = True

    def _preprocess(self, text: str) -> str:
        """Basic tokenizer + synonym mapper without NLTK."""
        if not isinstance(text, str):
            return ""

        tokens = text.split()

        normalized_tokens = [
            self.synonym_map.get(token.lower().strip(string.punctuation), token.lower())
            for token in tokens
            if token.lower() not in self.stop_words and token not in string.punctuation
        ]

        return " ".join(normalized_tokens)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity on bag-of-words after preprocessing."""
        if not text1 or not text2:
            return 0.0

        text1_clean = self._preprocess(text1)
        text2_clean = self._preprocess(text2)

        if not text1_clean and not text2_clean:
            return 1.0
        if not text1_clean or not text2_clean:
            return 0.0

        vectorizer = CountVectorizer().fit([text1_clean, text2_clean])
        vectors = vectorizer.transform([text1_clean, text2_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(similarity)
