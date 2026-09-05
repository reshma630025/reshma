"""
Text & Scam Detection Module for TrustGuard AI.
Analyzes message content for financial fraud, urgent threats, credential phishing,
lottery scams, and institutional impersonation.
"""
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("trustguard.text_detector")

SCAM_PATTERNS = [
    {
        "category": "Credential & OTP Phishing",
        "weight": 40,
        "regex": r"(?:(?:share|send|enter|verify|provide|forward|give|submit)\s+(?:your\s+)?(?:otp|one[- ]time|code|pin|password|2fa|credentials?))|(?:(?:otp|code|pin|password|2fa)\b.*?(?:share|send|enter|verify|provide))|(?:(?:click here|tap link|open link|verify here).*?(?:verify|update|unlock|reactivate|login|access).*?(?:account|card|bank|wallet|profile))",
        "label": "Credential / OTP Harvesting Attempt",
        "level": "high"
    },
    {
        "category": "Financial Solicitation & Extortion",
        "weight": 35,
        "regex": r"(?:wire transfer|western union|gift card|crypto|bitcoin|btc|usdt|ethereum|wallet address|payment via upi|send money to|deposit fee|processing fee|advance payment|transfer funds|pay (?:\$|₹|rs|usd)\s*\d+)",
        "label": "Direct Financial / Crypto / Wire Transfer Solicitation",
        "level": "high"
    },
    {
        "category": "Psychological Urgency & Threats",
        "weight": 30,
        "regex": r"(?:within (?:24|12|48|2|1) hours?|immediately|urgent|arrest warrant|legal action|police complaint|account (?:is )?(?:suspended|blocked|terminated|locked)|final notice|penalty will be charged|take legal steps)",
        "label": "High-Pressure Psychological Coercion / Urgency",
        "level": "mod"
    },
    {
        "category": "Lottery & Unsolicited Reward Scams",
        "weight": 35,
        "regex": r"(?:congratulations|you have won|selected as the winner|claim your (?:reward|prize|grant|lottery|bonus)|\$?(?:\d{1,3}(?:,\d{3})+|\d+)\s*(?:usd|dollars|pounds|cash|crypto))",
        "label": "Unsolicited Prize / Lottery / Reward Bait",
        "level": "high"
    },
    {
        "category": "Institutional Impersonation",
        "weight": 25,
        "regex": r"(?:irs|fbi|customs department|income tax department|microsoft tech support|apple security team|whatsapp support|telegram support|bank customer care|fraud department)",
        "label": "Authoritative Entity / Institutional Impersonation",
        "level": "mod"
    },
    {
        "category": "Suspicious Communication Links",
        "weight": 20,
        "regex": r"(?:wa\.me\/|t\.me\/|bit\.ly\/|tinyurl\.com\/|cutt\.ly\/|http:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|t\.co\/)",
        "label": "Obfuscated / Unofficial Communication Link",
        "level": "mod"
    }
]


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyzes raw text message for fraud indicators and computes risk score.
    """
    if not text or not text.strip():
        return {
            "success": False,
            "error": "No text content provided for analysis."
        }

    clean_text = text.strip()
    matched_indicators: List[Dict[str, Any]] = []
    accumulated_risk = 0.0

    # Test all scam pattern heuristics
    for pat in SCAM_PATTERNS:
        match = re.search(pat["regex"], clean_text, re.IGNORECASE)
        if match:
            accumulated_risk += pat["weight"]
            matched_indicators.append({
                "label": pat["label"],
                "detail": f"Matched pattern trigger: \"{match.group(0)[:60]}\"",
                "score": float(pat["weight"]),
                "level": pat["level"]
            })

    # Base linguistic checks (ALL CAPS shouting, excessive exclamation marks)
    caps_ratio = sum(1 for c in clean_text if c.isupper()) / max(1, len(clean_text))
    if caps_ratio > 0.40 and len(clean_text) > 25:
        accumulated_risk += 12.0
        matched_indicators.append({
            "label": "Aggressive Visual Styling",
            "detail": f"{int(caps_ratio*100)}% capitalized characters indicate coercive emphasis.",
            "score": 12.0,
            "level": "mod"
        })

    # Risk Score & Classification Logic (0..100)
    risk_score = min(100.0, max(0.0, accumulated_risk))

    if risk_score >= 50.0:
        classification = "SCAM"
        confidence = min(0.98, max(0.85, 0.70 + (risk_score / 200.0)))
        risk_level = "CRITICAL" if risk_score >= 80.0 else "VERY HIGH"
        explanation = f"High-confidence scam pattern match. Message exhibits multiple fraudulent vectors including {matched_indicators[0]['label'].lower()}."
    elif risk_score >= 21.0:
        classification = "SUSPICIOUS"
        confidence = 0.78
        risk_level = "HIGH" if risk_score >= 41.0 else "MODERATE"
        explanation = f"Potential fraud or suspicious intent detected. Exercise caution before clicking links or sharing information."
    else:
        classification = "SAFE"
        confidence = 0.94
        risk_level = "LOW"
        risk_score = max(4.0, risk_score)
        explanation = "No significant fraud, phishing, urgency coercion, or scam patterns detected in message content."

    # If safe, provide benign positive indicators
    if not matched_indicators:
        matched_indicators = [
            {
                "label": "Legitimate Linguistic Structure",
                "detail": "No suspicious financial requests or coercive urgency triggers identified.",
                "score": 5.0,
                "level": "safe"
            },
            {
                "label": "Zero Credential Harvesting Patterns",
                "detail": "No unauthorized OTP, password, or security token queries found.",
                "score": 0.0,
                "level": "safe"
            }
        ]

    return {
        "success": True,
        "type": "text",
        "classification": classification,
        "confidence": round(confidence, 2),
        "confidence_pct": round(confidence * 100, 1),
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "indicators": matched_indicators,
        "signals": matched_indicators,
        "explanation": explanation
    }
