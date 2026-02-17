"""Entity extraction for automatic memory tagging.

Extracts named entities (people, organizations, technologies, etc.) from natural
language to populate the entities field in save_memory calls.

Phase 1 Implementation (v0.6.0):
- spaCy-based NER (Named Entity Recognition)
- Configurable entity types
- Technology-specific pattern matching for AI/dev terms
- Fallback to regex patterns if spaCy unavailable
"""

import re

try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment,misc]
    SPACY_AVAILABLE = False


class EntityExtractor:
    """Extract named entities from natural language.

    Uses spaCy for NER when available, falls back to regex patterns otherwise.

    Extracted entity types:
    - PERSON: People's names
    - ORG: Organizations, companies
    - PRODUCT: Products, technologies
    - GPE: Geopolitical entities (cities, countries)
    - DATE: Time references
    - EVENT: Named events
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize entity extractor.

        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
                       Download with: python -m spacy download en_core_web_sm
        """
        self.nlp: "Language | None" = None

        if SPACY_AVAILABLE and spacy is not None:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                # Model not downloaded - will use fallback patterns
                pass

        # Technology/AI-specific patterns (case-insensitive)
        self.tech_patterns = [
            # AI/ML Models
            r"\b(?:GPT-?\d+|Claude|Gemini|LLaMA|BERT|T5)\b",
            # Frameworks
            r"\b(?:PyTorch|TensorFlow|FastAPI|Django|React|Vue)\b",
            # Languages
            r"\b(?:Python|JavaScript|TypeScript|Rust|Go|Java)\b",
            # Protocols
            r"\b(?:MCP|HTTP|gRPC|WebSocket|REST)\b",
            # Databases
            r"\b(?:PostgreSQL|MongoDB|Redis|SQLite)\b",
        ]

        self.tech_regex = re.compile("|".join(self.tech_patterns), re.IGNORECASE)

        # Fallback patterns if spaCy unavailable
        self.fallback_patterns = [
            # Capitalized words (likely proper nouns)
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            # Email addresses (as identifiers)
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            # URLs (as references)
            r"https?://[^\s]+",
        ]

        self.fallback_regex = re.compile("|".join(self.fallback_patterns), re.MULTILINE)

    def extract(self, text: str, max_entities: int = 10) -> list[str]:
        """Extract named entities from text.

        Args:
            text: Natural language text to analyze
            max_entities: Maximum entities to return (default: 10)

        Returns:
            List of entity strings (lowercase, deduplicated)

        Example:
            >>> extractor = EntityExtractor()
            >>> entities = extractor.extract("Claude and GPT-4 are LLMs from Anthropic and OpenAI")
            >>> "claude" in entities
            True
            >>> "gpt-4" in entities
            True
        """
        entities: set[str] = set()

        # Extract technology-specific entities first (always run)
        tech_matches = self.tech_regex.findall(text)
        entities.update(m.lower() for m in tech_matches)

        # Use spaCy if available
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Focus on most useful entity types for memory
                if ent.label_ in {
                    "PERSON",
                    "ORG",
                    "PRODUCT",
                    "GPE",
                    "EVENT",
                }:
                    entities.add(ent.text.lower())
        else:
            # Fallback to regex patterns
            fallback_matches = self.fallback_regex.findall(text)
            entities.update(m.lower() for m in fallback_matches)

        # Remove common stopwords that slip through
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        entities = {e for e in entities if e not in stopwords}

        # Return top max_entities by length (longer = more specific)
        sorted_entities = sorted(entities, key=len, reverse=True)
        return sorted_entities[:max_entities]

    def is_available(self) -> bool:
        """Check if spaCy NER is available.

        Returns:
            True if spaCy model loaded, False if using fallback patterns
        """
        return self.nlp is not None
