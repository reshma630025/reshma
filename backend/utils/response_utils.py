"""
Response formatting and score normalization utilities for TrustGuard AI.
"""

def clamp_score(value: float) -> float:
    """Clamp score between 0.0 and 100.0 with float precision rounding."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(100.0, val)), 4)


def authenticity_classification(fake_prob: float) -> dict:
    """
    Returns probabilistic classification based on fake probability score (0..100).
    
    Score Semantics:
    0   = Very likely genuine
    100 = Very likely fake / manipulated
    
    Thresholds:
    <= 30: LIKELY GENUINE (safe)
    <  70: UNCERTAIN (warning)
    >= 70: LIKELY FAKE / MANIPULATED (danger)
    """
    score = clamp_score(fake_prob)
    if score <= 30.0:
        return {
            "label": "LIKELY GENUINE",
            "status": "safe",
            "classification": "LIKELY_GENUINE"
        }
    elif score < 70.0:
        return {
            "label": "UNCERTAIN — VERIFY MANUALLY",
            "status": "warning",
            "classification": "UNCERTAIN"
        }
    else:
        return {
            "label": "LIKELY FAKE / MANIPULATED",
            "status": "danger",
            "classification": "LIKELY_FAKE"
        }


def normalize_fake_probability(probabilities: dict, id2label: dict) -> float:
    """
    Explicitly normalizes model class probabilities into a single `fakeProbability` score (0.0 to 100.0).
    
    Invariant:
    0   = very likely genuine / real
    100 = very likely fake / synthetic / manipulated
    
    Parameters:
        probabilities: dict of label_index (or label_str) -> raw_prob (0.0 to 1.0)
        id2label: dict mapping int class ID to str label name (e.g., {0: "Real", 1: "Fake"})
    """
    if not probabilities:
        return 50.0

    # Build lower-cased lookup for probabilities
    prob_lookup = {}
    for k, v in probabilities.items():
        prob_lookup[str(k).strip().lower()] = float(v)

    # 1. First check if probabilities dict directly contains "fake" or "real" keys
    fake_val = None
    real_val = None

    for k, v in prob_lookup.items():
        if any(term in k for term in ["fake", "synthetic", "manipulated", "generated", "deepfake"]):
            fake_val = v
        elif any(term in k for term in ["real", "genuine", "authentic", "human", "original"]):
            real_val = v

    if fake_val is not None:
        return clamp_score(fake_val * 100.0)
    if real_val is not None:
        return clamp_score((1.0 - real_val) * 100.0)

    # 2. Check id2label
    if id2label:
        for class_id, label_name in id2label.items():
            lbl = str(label_name).strip().lower()
            cid = str(class_id).strip().lower()
            score = prob_lookup.get(cid, prob_lookup.get(lbl, 0.0))

            if any(term in lbl for term in ["fake", "synthetic", "manipulated", "generated", "deepfake"]):
                return clamp_score(score * 100.0)
            elif any(term in lbl for term in ["real", "genuine", "authentic", "human", "original"]):
                return clamp_score((1.0 - score) * 100.0)

    return 50.0
