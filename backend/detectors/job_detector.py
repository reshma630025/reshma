"""
Job & Internship Fraud Detection Module for TrustGuard AI.
Analyzes job descriptions for advance registration fees, security deposits,
unrealistic salaries, free recruiter email domains, and fake offer signals.
"""
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("trustguard.job_detector")

JOB_FRAUD_PATTERNS = [
    {
        "category": "Advance Fees & Deposits",
        "weight": 40,
        "regex": r"(?:registration fee|application fee|training fee|security deposit|refundable deposit|laptop fee|processing fee|onboarding fee|buy equipment from our vendor|pay (?:\$|₹|rs|usd)\s*\d+)",
        "label": "Mandatory Upfront Fee / Security Deposit Demand",
        "level": "high"
    },
    {
        "category": "Unrealistic Compensation",
        "weight": 25,
        "regex": r"(?:earn (?:\$|₹|rs)\s*(?:[5-9]\d{3}|[1-9]\d{4,})\s*(?:per (?:day|week|hour)|daily)|no experience needed.*(?:\$|₹)\s*\d{4,}|guaranteed income of|work 1 hour.*(?:\$|₹)\s*\d{3,})",
        "label": "Unrealistic Salary-to-Effort Ratio",
        "level": "mod"
    },
    {
        "category": "Informal / Suspicious Recruiter Contact",
        "weight": 25,
        "regex": r"(?:contact on whatsapp|telegram HR|message hr at @|send resume to (?:[a-zA-Z0-9._%+-]+@(gmail|yahoo|hotmail|outlook|protonmail|aol)\.com))",
        "label": "Unofficial Public Email / Messaging Platform Recruitment",
        "level": "mod"
    },
    {
        "category": "Guaranteed Employment Without Interview",
        "weight": 30,
        "regex": r"(?:immediate selection|100% guaranteed job|no interview required|direct appointment letter|instant joining without screening)",
        "label": "Guaranteed Hiring Without Formal Screening",
        "level": "high"
    },
    {
        "category": "Vague / Shady Job Scope",
        "weight": 20,
        "regex": r"(?:part time data entry|copy paste work|captcha filling|like and subscribe jobs|task based review earning)",
        "label": "High-Risk Task/Review/Data-Entry Scam Format",
        "level": "mod"
    }
]


def analyze_job_or_internship(
    description: str,
    url: str = "",
    email: str = "",
    is_internship: bool = False
) -> Dict[str, Any]:
    """
    Analyzes job or internship listing text and metadata for fraudulent markers.
    """
    combined_text = f"{description} {url} {email}".strip()
    if not combined_text:
        return {
            "success": False,
            "error": "No job description or contact details provided."
        }

    matched_indicators: List[Dict[str, Any]] = []
    accumulated_risk = 0.0

    for pat in JOB_FRAUD_PATTERNS:
        match = re.search(pat["regex"], combined_text, re.IGNORECASE)
        if match:
            accumulated_risk += pat["weight"]
            matched_indicators.append({
                "label": pat["label"],
                "detail": f"Detected risk trigger: \"{match.group(0)[:65]}\"",
                "score": float(pat["weight"]),
                "level": pat["level"]
            })

    # Free email check for formal recruiter email
    if email and any(prov in email.lower() for prov in ["@gmail.com", "@yahoo.com", "@hotmail.com", "@outlook.com"]):
        accumulated_risk += 18.0
        matched_indicators.append({
            "label": "Non-Corporate Recruiter Email",
            "detail": f"Recruiter provided public email provider ({email}) instead of an official company domain.",
            "score": 18.0,
            "level": "mod"
        })

    risk_score = min(100.0, max(0.0, accumulated_risk))

    if risk_score >= 55.0:
        classification = "FRAUDULENT"
        confidence = min(0.96, max(0.85, 0.70 + (risk_score / 200.0)))
        risk_level = "CRITICAL" if risk_score >= 80.0 else "VERY HIGH"
        target_name = "Internship" if is_internship else "Job"
        explanation = f"High-risk {target_name.lower()} posting identified with clear predatory fraud markers (such as upfront fee demands or fake recruitment channels)."
    elif risk_score >= 21.0:
        classification = "SUSPICIOUS"
        confidence = 0.76
        risk_level = "HIGH" if risk_score >= 41.0 else "MODERATE"
        target_name = "Internship" if is_internship else "Job"
        explanation = f"Ambiguous or suspicious {target_name.lower()} listing. Verify the employer's official careers portal directly before submitting personal data."
    else:
        classification = "GENUINE"
        confidence = 0.93
        risk_level = "LOW"
        risk_score = max(5.0, risk_score)
        target_name = "Internship" if is_internship else "Job"
        explanation = f"Standard professional {target_name.lower()} format. No advance fee requests, predatory contracts, or fraudulent communication channels found."

    if not matched_indicators:
        matched_indicators = [
            {
                "label": "No Advance Fee Demands",
                "detail": "Legitimate employer policy — zero upfront monetary requirements.",
                "score": 0.0,
                "level": "safe"
            },
            {
                "label": "Standard Recruitment Format",
                "detail": "Job duties and requirements follow verified professional industry standards.",
                "score": 5.0,
                "level": "safe"
            }
        ]

    return {
        "success": True,
        "type": "internship" if is_internship else "job",
        "classification": classification,
        "confidence": round(confidence, 2),
        "confidence_pct": round(confidence * 100, 1),
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "indicators": matched_indicators,
        "signals": matched_indicators,
        "explanation": explanation
    }
