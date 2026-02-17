"""
Detection logic for natural language activation.

This module implements confidence scoring using a weighted sigmoid formula
and provides high-level detection functions for save/recall intent.
"""

import math
import time
from typing import Literal

from cortexgraph.activation.config import ActivationConfig, get_signal_weight
from cortexgraph.activation.entity_extraction import extract_entities
from cortexgraph.activation.models import ActivationSignal, MessageAnalysis, RecallAnalysis
from cortexgraph.activation.patterns import PatternMatcher


def sigmoid(x: float) -> float:
    """Sigmoid activation function.

    Maps any real value to range [0, 1].

    Args:
        x: Input value (raw score)

    Returns:
        Sigmoid output in range [0, 1]

    Formula:
        σ(x) = 1 / (1 + e^(-x))
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        # Handle very negative values
        return 0.0 if x < 0 else 1.0


def calculate_confidence(signals: dict[str, float], bias: float) -> float:
    """Calculate confidence score using weighted sigmoid.

    Args:
        signals: Dictionary of signal_name -> weight
        bias: Bias term for calibration

    Returns:
        Confidence score in range [0, 1]

    Formula:
        confidence = σ(Σwᵢsᵢ + b)

    Where:
        σ = sigmoid function
        wᵢ = signal weight from config
        sᵢ = signal indicator (0 or 1, or count for entity_count)
        b = bias term (default: -2.0)

    Example:
        >>> signals = {"explicit_save_request": 1.0, "entity_count": 2.0}
        >>> calculate_confidence(signals, bias=-2.0)
        0.92  # High confidence
    """
    # Sum weighted signals
    raw_score = sum(signals.values()) + bias

    # Apply sigmoid
    confidence = sigmoid(raw_score)

    return confidence


def detect_save_intent(
    message: str, config: ActivationConfig, matcher: PatternMatcher
) -> MessageAnalysis:
    """Detect memory-worthy content in user message.

    Args:
        message: User message to analyze
        config: Activation configuration
        matcher: Pattern matching engine

    Returns:
        MessageAnalysis with decision support

    Example:
        >>> analysis = detect_save_intent("Remember this: I prefer PostgreSQL", config, matcher)
        >>> analysis.should_save
        True
        >>> analysis.confidence >= 0.7
        True
    """
    # Run pattern matching
    pattern_results = matcher.detect_all_signals(message)

    # Extract entities
    entities = extract_entities(message, max_entities=10)

    # Build signal dictionary
    signals: dict[str, float] = {}
    phrase_signals: dict[str, bool] = {}

    # Explicit save request (strongest signal)
    if pattern_results["save"].matched:
        signals["explicit_save_request"] = get_signal_weight(config, "explicit_save_request")
        phrase_signals["save_request"] = True

    # Importance markers
    if pattern_results["importance"].matched:
        # Different weights for critical vs important
        if any("critical" in p for p in pattern_results["importance"].matched_patterns):
            signals["critical_marker"] = get_signal_weight(config, "critical_marker")
            phrase_signals["critical_marker"] = True
        else:
            signals["important_marker"] = get_signal_weight(config, "important_marker")
            phrase_signals["importance_marker"] = True

    # Exclusion patterns (negative signal)
    if pattern_results["exclusion"].matched:
        # General questions should not be saved
        phrase_signals["exclusion_pattern"] = True
        signals["exclusion"] = -5.0  # Strong negative

    # Uncertainty markers (negative signal)
    if pattern_results["uncertainty"].matched:
        signals["uncertainty_marker"] = get_signal_weight(config, "uncertainty_marker")
        phrase_signals["uncertainty_marker"] = True

    # Entity count (more entities = more concrete/memorable)
    entity_count_weight = get_signal_weight(config, "entity_count")
    entity_contribution = min(len(entities) * entity_count_weight, 3.2)  # Cap at 4 entities
    if entity_contribution > 0:
        signals["entity_count"] = entity_contribution

    # Decision/preference detection (heuristic)
    decision_keywords = ["decided", "choice", "decision", "prefer", "preference"]
    if any(kw in message.lower() for kw in decision_keywords):
        signals["preference_statement"] = get_signal_weight(config, "preference_statement")
        phrase_signals["decision_marker"] = True

    # Calculate confidence
    confidence = calculate_confidence(signals, config.bias)

    # Make decision
    decision = config.thresholds.get_decision(confidence)
    should_save = decision == "auto"

    # Calculate strength suggestion (1.0-2.0)
    if confidence >= 0.9:
        suggested_strength = 2.0
    elif confidence >= 0.7:
        suggested_strength = 1.5
    else:
        suggested_strength = 1.0

    # Generate tags (simple heuristic from entities)
    suggested_tags: list[str] = []
    if any(
        tech in " ".join(entities).lower() for tech in ["database", "postgres", "mongodb", "redis"]
    ):
        suggested_tags.append("database")
    if any(tech in " ".join(entities).lower() for tech in ["api", "rest", "graphql", "http"]):
        suggested_tags.append("api")
    if "decision" in phrase_signals or "preference" in message.lower():
        suggested_tags.append("preference")

    # Build reasoning string
    signal_breakdown = ", ".join(
        f"{name.replace('_', ' ')}: {value:+.1f}" for name, value in signals.items()
    )
    raw_score = sum(signals.values()) + config.bias
    reasoning = (
        f"Signals: {signal_breakdown} | Raw score: {raw_score:.1f} → Confidence: {confidence:.3f}"
    )

    return MessageAnalysis(
        should_save=should_save,
        confidence=confidence,
        suggested_entities=entities,
        suggested_tags=suggested_tags,
        suggested_strength=suggested_strength,
        reasoning=reasoning,
        phrase_signals=phrase_signals,
    )


def detect_recall_intent(
    query: str, config: ActivationConfig, matcher: PatternMatcher
) -> RecallAnalysis:
    """Detect recall intent in user query.

    Args:
        query: User query to analyze
        config: Activation configuration
        matcher: Pattern matching engine

    Returns:
        RecallAnalysis with decision support

    Example:
        >>> analysis = detect_recall_intent("What did I say about databases?", config, matcher)
        >>> analysis.should_search
        True
        >>> "database" in analysis.suggested_query
        True
    """
    # Run pattern matching
    pattern_results = matcher.detect_all_signals(query)

    # Extract entities
    entities = extract_entities(query, max_entities=10)

    # Build signal dictionary
    signals: dict[str, float] = {}
    phrase_signals: dict[str, bool] = {}

    # Explicit recall request (strongest signal)
    if pattern_results["recall"].matched:
        signals["explicit_recall_request"] = get_signal_weight(config, "explicit_recall_request")
        phrase_signals["recall_request"] = True

    # Past reference detection (heuristic)
    past_keywords = ["said", "told", "discussed", "mentioned", "last time", "previously"]
    if any(kw in query.lower() for kw in past_keywords):
        signals["past_reference"] = 2.0
        phrase_signals["past_reference"] = True

    # Question markers (questions more likely to be recall)
    question_markers = ["what", "when", "where", "who", "which", "how"]
    if any(query.lower().startswith(qm) for qm in question_markers):
        signals["question_marker"] = 1.5
        phrase_signals["question_marker"] = True

    # Possessive references ("my X") suggest recall
    if any(poss in query.lower() for poss in ["my ", "our "]):
        signals["possessive_reference"] = 2.0
        phrase_signals["possessive_reference"] = True

    # Exclusion patterns (negative signal)
    if pattern_results["exclusion"].matched:
        # General "what is" questions are NOT memory recall
        signals["exclusion"] = -4.0
        phrase_signals["exclusion_pattern"] = True

    # Entity count (more entities = more specific query)
    entity_count_weight = get_signal_weight(config, "entity_count")
    entity_contribution = min(len(entities) * entity_count_weight, 3.2)
    if entity_contribution > 0:
        signals["entity_count"] = entity_contribution

    # Calculate confidence
    confidence = calculate_confidence(signals, config.bias)

    # Make decision
    decision = config.thresholds.get_decision(confidence)
    should_search = decision == "auto"

    # Extract search query (strip recall phrases, keep content)
    suggested_query = query
    for pattern in config.patterns.explicit_recall_triggers:
        suggested_query = suggested_query.replace(pattern, "").strip()

    # Generate tags from entities
    suggested_tags: list[str] = []
    if entities:
        # Use first few entities as tags
        suggested_tags = entities[:3]

    # Build reasoning string
    signal_breakdown = ", ".join(
        f"{name.replace('_', ' ')}: {value:+.1f}" for name, value in signals.items()
    )
    raw_score = sum(signals.values()) + config.bias
    reasoning = (
        f"Signals: {signal_breakdown} | Raw score: {raw_score:.1f} → Confidence: {confidence:.3f}"
    )

    return RecallAnalysis(
        should_search=should_search,
        confidence=confidence,
        suggested_query=suggested_query,
        suggested_tags=suggested_tags,
        suggested_entities=entities,
        reasoning=reasoning,
        phrase_signals=phrase_signals,
    )


def create_activation_signal(
    signal_type: Literal["save", "recall", "reinforce"],
    confidence: float,
    matched_patterns: list[str],
    context: str,
) -> ActivationSignal:
    """Create an ActivationSignal for internal processing.

    Args:
        signal_type: Type of signal ("save", "recall", "reinforce")
        confidence: Confidence score (0.0-1.0)
        matched_patterns: List of matched pattern strings
        context: Surrounding text context

    Returns:
        ActivationSignal instance

    Note:
        ActivationSignal is an internal model. Use detect_save_intent() or
        detect_recall_intent() for MCP tool integration.
    """
    return ActivationSignal(
        type=signal_type,
        confidence=confidence,
        matched_patterns=matched_patterns,
        context=context[:1000],  # Truncate to max length
        timestamp=int(time.time()),
    )
