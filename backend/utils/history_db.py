"""
SQLite database and dynamic statistics tracker for TrustGuard AI.
Stores scan history and computes live metrics for the dashboard with user-scoped isolation.
"""
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "trustguard.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite schema if it doesn't already exist and migrates columns."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                timestamp REAL NOT NULL,
                scan_type TEXT NOT NULL,
                content_label TEXT NOT NULL,
                classification TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                is_threat INTEGER NOT NULL DEFAULT 0,
                is_deepfake INTEGER NOT NULL DEFAULT 0,
                is_scam INTEGER NOT NULL DEFAULT 0,
                explanation TEXT,
                indicators_json TEXT,
                frame_results_json TEXT DEFAULT '[]',
                segment_results_json TEXT DEFAULT '[]'
            )
        """)
        
        # Check and migrate columns if missing from legacy tables
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(scan_history)").fetchall()]
        if "user_id" not in cols:
            conn.execute("ALTER TABLE scan_history ADD COLUMN user_id INTEGER DEFAULT 1")
        if "frame_results_json" not in cols:
            conn.execute("ALTER TABLE scan_history ADD COLUMN frame_results_json TEXT DEFAULT '[]'")
        if "segment_results_json" not in cols:
            conn.execute("ALTER TABLE scan_history ADD COLUMN segment_results_json TEXT DEFAULT '[]'")
            
        conn.commit()


def record_scan(
    scan_type: str,
    content_label: str,
    classification: str,
    confidence: float,
    risk_score: float,
    risk_level: str,
    user_id: Optional[int] = 1,
    explanation: str = "",
    indicators_json: str = "[]",
    frame_results_json: str = "[]",
    segment_results_json: str = "[]"
) -> int:
    """Inserts a completed scan record into the database."""
    init_db()
    
    is_threat = 1 if risk_score > 30.0 or classification in ["FAKE", "FRAUDULENT", "SCAM", "PHISHING", "SUSPICIOUS"] else 0
    is_deepfake = 1 if scan_type in ["image", "video", "audio"] and risk_score > 30.0 else 0
    is_scam = 1 if scan_type in ["text", "job", "internship", "url", "ocr", "social"] and risk_score > 30.0 else 0
    u_id = user_id if user_id is not None else 1

    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO scan_history (
                user_id, timestamp, scan_type, content_label, classification,
                confidence, risk_score, risk_level, is_threat,
                is_deepfake, is_scam, explanation, indicators_json,
                frame_results_json, segment_results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            u_id, time.time(), scan_type, content_label, classification,
            confidence, risk_score, risk_level, is_threat,
            is_deepfake, is_scam, explanation, indicators_json,
            frame_results_json, segment_results_json
        ))
        conn.commit()
        return cursor.lastrowid


def get_stats(user_id: Optional[int] = None) -> Dict[str, int]:
    """Calculates live dashboard statistics from stored history."""
    init_db()
    with get_db() as conn:
        if user_id is not None:
            total = conn.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ?", (user_id,)).fetchone()[0]
            threats = conn.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND is_threat = 1", (user_id,)).fetchone()[0]
            deepfakes = conn.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND is_deepfake = 1", (user_id,)).fetchone()[0]
            scams = conn.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND is_scam = 1", (user_id,)).fetchone()[0]
        else:
            total = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
            threats = conn.execute("SELECT COUNT(*) FROM scan_history WHERE is_threat = 1").fetchone()[0]
            deepfakes = conn.execute("SELECT COUNT(*) FROM scan_history WHERE is_deepfake = 1").fetchone()[0]
            scams = conn.execute("SELECT COUNT(*) FROM scan_history WHERE is_scam = 1").fetchone()[0]

        return {
            "scans": total,
            "threats": threats,
            "deepfakes": deepfakes,
            "scams": scams
        }


def get_history(user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches recent scan history."""
    init_db()
    with get_db() as conn:
        if user_id is not None:
            rows = conn.execute("""
                SELECT id, user_id, timestamp, scan_type, content_label, classification,
                       confidence, risk_score, risk_level, explanation,
                       frame_results_json, segment_results_json
                FROM scan_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, user_id, timestamp, scan_type, content_label, classification,
                       confidence, risk_score, risk_level, explanation,
                       frame_results_json, segment_results_json
                FROM scan_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return [dict(r) for r in rows]


def clear_history(user_id: Optional[int] = None):
    """Clears scan history."""
    init_db()
    with get_db() as conn:
        if user_id is not None:
            conn.execute("DELETE FROM scan_history WHERE user_id = ?", (user_id,))
        else:
            conn.execute("DELETE FROM scan_history")
        conn.commit()


# Initialize on import
init_db()
