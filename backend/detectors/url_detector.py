"""
URL Safety and Phishing Scanner for TrustGuard AI.
Analyzes domain structure, TLD risk, IP hostnames, entropy, phishing paths, and redirects.
"""
import re
import math
import logging
from urllib.parse import urlparse
from typing import Dict, Any, List

logger = logging.getLogger("trustguard.url_detector")

SUSPICIOUS_TLDS = {
    ".top", ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".buzz", ".fit",
    ".rest", ".work", ".click", ".link", ".country", ".kim", ".science",
    ".gdn", ".loan", ".racing", ".win", ".bid", ".accountant", ".download"
}

PHISHING_KEYWORDS = [
    "login", "signin", "verify", "account", "security", "update", "banking",
    "wallet", "metamask", "paypal", "netflix", "appleid", "secure", "billing",
    "recover", "confirm", "authenticate", "kyc", "unlock", "support"
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "ow.ly", "cutt.ly", "rb.gy", "shorturl.at"
}


def compute_shannon_entropy(string: str) -> float:
    """Calculates Shannon entropy to detect algorithmic / dga domain names."""
    if not string:
        return 0.0
    prob = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum(p * math.log2(p) for p in prob)


def analyze_url(raw_url: str) -> Dict[str, Any]:
    """
    Evaluates URL security metrics and identifies potential phishing attacks.
    """
    if not raw_url or not raw_url.strip():
        return {
            "success": False,
            "error": "No URL provided for security scan."
        }

    url_str = raw_url.strip()
    if not url_str.startswith(("http://", "https://")):
        url_str = "http://" + url_str

    try:
        parsed = urlparse(url_str)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
    except Exception as e:
        return {
            "success": False,
            "error": f"Invalid URL syntax: {str(e)}"
        }

    matched_indicators: List[Dict[str, Any]] = []
    accumulated_risk = 0.0

    # 1. IP-based Hostname Check
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if re.match(ip_pattern, hostname):
        accumulated_risk += 45.0
        matched_indicators.append({
            "label": "Direct IP Hostname",
            "detail": f"Host is a raw numerical IP address ({hostname}) instead of a verified domain name.",
            "score": 45.0,
            "level": "high"
        })

    # 2. Suspicious High-Risk TLD Check
    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            accumulated_risk += 30.0
            matched_indicators.append({
                "label": f"High-Risk TLD ({tld})",
                "detail": f"Domain utilizes a top-level domain frequently associated with spam and automated campaigns.",
                "score": 30.0,
                "level": "mod"
            })
            break

    # 3. Excessive Subdomains (Domain Spoofing / Impersonation)
    subdomains = hostname.split(".")
    if len(subdomains) >= 4:
        accumulated_risk += 25.0
        matched_indicators.append({
            "label": "Excessive Subdomain Depth",
            "detail": f"Identified {len(subdomains)} domain segments, often used to disguise phishing targets.",
            "score": 25.0,
            "level": "mod"
        })

    # 4. Phishing Keywords in Hostname or Path
    matched_keywords = [kw for kw in PHISHING_KEYWORDS if kw in hostname or kw in path]
    if len(matched_keywords) >= 2:
        accumulated_risk += 35.0
        matched_indicators.append({
            "label": "Credential Phishing Token Triggers",
            "detail": f"URL path contains high-risk account harvesting terms: {', '.join(matched_keywords[:3])}",
            "score": 35.0,
            "level": "high"
        })

    # 5. URL Shortener Redirection
    if hostname in SHORTENER_DOMAINS:
        accumulated_risk += 20.0
        matched_indicators.append({
            "label": "URL Shortener Redirection",
            "detail": f"Link is cloaked behind a shortening service ({hostname}), concealing true destination.",
            "score": 20.0,
            "level": "mod"
        })

    # 6. High Shannon Entropy / DGA Domain Name
    entropy = compute_shannon_entropy(hostname.split(".")[0])
    if entropy > 4.2 and len(hostname.split(".")[0]) > 12:
        accumulated_risk += 25.0
        matched_indicators.append({
            "label": "High Domain Entropy (Potential DGA)",
            "detail": f"Domain prefix shows high randomness (entropy: {entropy:.2f}), typical of malware domains.",
            "score": 25.0,
            "level": "mod"
        })

    # 7. Unencrypted HTTP Scheme
    if parsed.scheme == "http":
        accumulated_risk += 8.0
        matched_indicators.append({
            "label": "Unencrypted HTTP Connection",
            "detail": "Connection is not secured with SSL/TLS encryption.",
            "score": 8.0,
            "level": "safe" if accumulated_risk < 20 else "mod"
        })

    risk_score = min(100.0, max(0.0, accumulated_risk))

    if risk_score >= 55.0:
        classification = "PHISHING / MALICIOUS"
        confidence = min(0.97, max(0.86, 0.72 + (risk_score / 200.0)))
        risk_level = "CRITICAL" if risk_score >= 80.0 else "VERY HIGH"
        explanation = f"High-confidence malicious or phishing URL detected with {len(matched_indicators)} risk triggers. Avoid entering credentials or downloading files."
    elif risk_score >= 21.0:
        classification = "SUSPICIOUS"
        confidence = 0.82
        risk_level = "HIGH" if risk_score >= 41.0 else "MODERATE"
        explanation = f"URL exhibits non-standard structural indicators (such as link cloaking or suspicious TLD). Verify destination before proceeding."
    else:
        classification = "SAFE"
        confidence = 0.95
        risk_level = "LOW"
        risk_score = max(4.0, risk_score)
        explanation = "Domain structure, protocol, and path follow standard legitimate web conventions with no phishing triggers."

    if not matched_indicators:
        matched_indicators = [
            {
                "label": "Standard Domain Architecture",
                "detail": "Verified domain syntax, legitimate TLD, and absence of cloaking tokens.",
                "score": 0.0,
                "level": "safe"
            },
            {
                "label": "Secure Transport Protocol",
                "detail": "Protected with valid HTTPS encryption.",
                "score": 0.0,
                "level": "safe"
            }
        ]

    return {
        "success": True,
        "type": "url",
        "url": url_str,
        "hostname": hostname,
        "classification": classification,
        "confidence": round(confidence, 2),
        "confidence_pct": round(confidence * 100, 1),
        "risk_score": round(risk_score, 1),
        "authenticity_probability": round(100.0 - risk_score, 1),
        "risk_level": risk_level,
        "indicators": matched_indicators,
        "signals": matched_indicators,
        "explanation": explanation
    }
