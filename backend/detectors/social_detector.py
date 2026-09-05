"""
Social Media Fraud and Impersonation Detection Module for TrustGuard AI.
Analyzes social posts, influencer giveaways, fake investment drops, and viral engagement scams.
"""
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("trustguard.social_detector")

SOCIAL_FRAUD_PATTERNS = [
    {
        "category": "Fake Crypto & Asset Multiplier Schemes",
        "weight": 45,
        "regex": r"(?:send (?:0\.\d+|\d+)\s*(?:btc|eth|sol|usdt|bnb|crypto).*?(?:get|receive|back)|airdrop|giveaway.*?(?:wallet|metamask|connect|claim)|double your (?:crypto|money|bitcoin)|crypto (?:grant|bonus|reward))",
        "label": "Crypto Airdrop / Asset Doubling Scam Pattern",
        "level": "high"
    },
    {
        "category": "High-Pressure Engagement Bait & Direct Message Phishing",
        "weight": 25,
        "regex": r"(?:comment ['\"]?(?:ready|info|yes|earn|win)['\"]?|dm (?:me|us) for details|limited slots remaining|first \d+ people (?:get|receive)|message our (?:telegram|whatsapp) (?:admin|team))",
        "label": "High-Pressure DM Funnel / Engagement Bait",
        "level": "mod"
    },
    {
        "category": "Celebrity / Brand Impersonation",
        "weight": 35,
        "regex": r"(?:elon musk|mrbeast|binance official|vitalik|telegram support admin|whatsapp crypto group|guaranteed forex trading returns)",
        "label": "High-Profile Identity / Institutional Impersonation",
        "level": "high"
    },
    {
        "category": "Fake Giveaways & Free Goods",
        "weight": 30,
        "regex": r"(?:free iphone|free macbook|free gift cards|selected for exclusive bonus|click (?:the )?bio link to claim|exclusive voucher code)",
        "label": "Unrealistic Free Merchandise / Gift Lure",
        "level": "high"
    },
    {
        "category": "Suspicious Social Redirect Links",
        "weight": 20,
        "regex": r"(?:t\.me\/|wa\.me\/|bit\.ly\/|cutt\.ly\/|tinyurl\.com\/|rb\.gy\/|shorturl\.at\/)",
        "label": "Off-Platform Redirection Link",
        "level": "mod"
    }
]


def analyze_social_post(text: str, url: str = "") -> Dict[str, Any]:
    """
    Analyzes social media post content and associated links for viral fraud vectors.
    """
    combined = f"{text} {url}".strip()
    if not combined:
        return {
            "success": False,
            "error": "No social media text or link provided."
        }

    matched_indicators: List[Dict[str, Any]] = []
    accumulated_risk = 0.0

    for pat in SOCIAL_FRAUD_PATTERNS:
        match = re.search(pat["regex"], combined, re.IGNORECASE)
        if match:
            accumulated_risk += pat["weight"]
            matched_indicators.append({
                "label": pat["label"],
                "detail": f"Pattern match: \"{match.group(0)[:60]}\"",
                "score": float(pat["weight"]),
                "level": pat["level"]
            })

    # Excessive viral emphasis (exclamation marks or emojis)
    if combined.count("!") >= 3:
        accumulated_risk += 8.0

    risk_score = min(100.0, max(0.0, accumulated_risk))

    if risk_score >= 50.0:
        classification = "FRAUDULENT SOCIAL CONTENT"
        confidence = min(0.96, max(0.85, 0.70 + (risk_score / 200.0)))
        risk_level = "CRITICAL" if risk_score >= 80.0 else "VERY HIGH"
        explanation = f"High-confidence social media fraud detected. Content exhibits known viral scam formats ({matched_indicators[0]['label']})."
    elif risk_score >= 21.0:
        classification = "SUSPICIOUS SOCIAL CONTENT"
        confidence = 0.78
        risk_level = "HIGH" if risk_score >= 41.0 else "MODERATE"
        explanation = f"Social post exhibits engagement bait or unverified promotional claims. Avoid clicking unfamiliar links or sharing credentials."
    else:
        classification = "LOW RISK SOCIAL CONTENT"
        confidence = 0.94
        risk_level = "LOW"
        risk_score = max(4.0, risk_score)
        explanation = "Natural organic social media communication. Zero fake giveaways, crypto multipliers, or phishing funnel patterns detected."

    if not matched_indicators:
        matched_indicators = [
            {
                "label": "Organic Social Communication",
                "detail": "No artificial engagement funneling or coercive reward schemes found.",
                "score": 0.0,
                "level": "safe"
            },
            {
                "label": "Safe Platform Interaction",
                "detail": "Zero malicious off-platform redirect patterns identified.",
                "score": 0.0,
                "level": "safe"
            }
        ]

    return {
        "success": True,
        "type": "social",
        "classification": classification,
        "confidence": round(confidence, 2),
        "confidence_pct": round(confidence * 100, 1),
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "indicators": matched_indicators,
        "signals": matched_indicators,
        "explanation": explanation
    }
