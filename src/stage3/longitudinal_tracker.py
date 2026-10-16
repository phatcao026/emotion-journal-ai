"""
src/stage3/longitudinal_tracker.py
----------------------------------
Persistent Longitudinal Mood Tracking & Trajectory Analytics for Emotion Journal AI.
Uses SQLite for zero-configuration, robust local persistence.

Capabilities:
    - Log emotion sessions with rich affective metadata & biomarkers
    - Compute rolling mood trajectories (e.g., 7-day, 30-day)
    - Track emotional volatility / stability index
    - Detect consecutive negative emotion streaks (for crisis / burnout alerts)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.stage3.emotion_result import EmotionResult

logger = logging.getLogger(__name__)

VALENCE_MAP = {
    "JOY": 1.0,
    "CALM": 0.8,
    "HOPE": 0.9,
    "CONNECTION": 0.9,
    "NEUTRAL": 0.0,
    "ANXIETY": -0.6,
    "FEAR": -0.8,
    "ANGER": -0.7,
    "DISGUST": -0.8,
    "SADNESS": -0.8,
    "LONELINESS": -0.9,
    "GUILT_SHAME": -0.9,
}


class LongitudinalTracker:
    """Manages longitudinal journal session records and computes temporal mood trends."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_dir = Path("data/journal_records")
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_dir / "mood_tracker.db")
        else:
            self.db_path = str(db_path)
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indices if not already present."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS journal_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE,
                        timestamp TEXT NOT NULL,
                        primary_label TEXT NOT NULL,
                        primary_confidence REAL NOT NULL,
                        primary_entropy REAL,
                        sub_label TEXT NOT NULL,
                        sub_confidence REAL NOT NULL,
                        gate_audio REAL,
                        gate_text REAL,
                        dominant_modality TEXT,
                        f0_mean REAL,
                        f0_std REAL,
                        f0_slope REAL,
                        jitter_local REAL,
                        shimmer_local REAL,
                        hnr_db REAL,
                        pause_ratio REAL,
                        speech_rate_wpm REAL,
                        transcript TEXT,
                        audio_duration_s REAL,
                        valence_score REAL,
                        model_version TEXT
                    )
                """)
                for col_def in [
                    "f0_std REAL",
                    "f0_slope REAL",
                    "jitter_local REAL",
                    "shimmer_local REAL",
                    "speech_rate_wpm REAL",
                ]:
                    try:
                        conn.execute(f"ALTER TABLE journal_sessions ADD COLUMN {col_def}")
                    except sqlite3.OperationalError:
                        pass
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON journal_sessions (timestamp)")
        finally:
            conn.close()

    def log_session(self, result: EmotionResult) -> int:
        """Insert an EmotionResult into the database."""
        valence = VALENCE_MAP.get(result.sub_label.upper(), VALENCE_MAP.get(result.primary_label.upper(), 0.0))
        sess_id = result.session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO journal_sessions (
                        session_id, timestamp, primary_label, primary_confidence, primary_entropy,
                        sub_label, sub_confidence, gate_audio, gate_text, dominant_modality,
                        f0_mean, f0_std, f0_slope, jitter_local, shimmer_local, hnr_db,
                        pause_ratio, speech_rate_wpm, transcript, audio_duration_s,
                        valence_score, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sess_id,
                    result.timestamp,
                    result.primary_label,
                    result.primary_confidence,
                    result.primary_entropy,
                    result.sub_label,
                    result.sub_confidence,
                    result.gate_audio,
                    result.gate_text,
                    result.dominant_modality,
                    result.f0_mean,
                    result.f0_std,
                    result.f0_slope,
                    result.jitter_local,
                    result.shimmer_local,
                    result.hnr_db,
                    result.pause_ratio,
                    result.speech_rate_wpm,
                    result.transcript,
                    result.audio_duration_s,
                    valence,
                    result.model_version,
                ))
                return cursor.lastrowid or 0
        finally:
            conn.close()

    def get_history(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch past journal entries up to `days` ago."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT * FROM journal_sessions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def compute_mood_trends(self, days: int = 7) -> Dict[str, Any]:
        """Compute rolling analytics across the last `days` days."""
        history = self.get_history(days=days, limit=100)
        if not history:
            return {
                "total_sessions": 0,
                "days_analyzed": days,
                "primary_distribution": {},
                "mean_valence": 0.0,
                "stability_score": 1.0,
                "consecutive_negative_streak": 0,
                "summary": "Chưa có đủ dữ liệu nhật ký trong khoảng thời gian này.",
            }

        total = len(history)
        primary_counts: Dict[str, int] = {}
        sub_counts: Dict[str, int] = {}
        valences: List[float] = []

        for r in history:
            p = r["primary_label"]
            s = r["sub_label"]
            primary_counts[p] = primary_counts.get(p, 0) + 1
            sub_counts[s] = sub_counts.get(s, 0) + 1
            valences.append(float(r["valence_score"] or 0.0))

        mean_val = float(sum(valences) / len(valences))
        variance = float(sum((v - mean_val) ** 2 for v in valences) / len(valences)) if len(valences) > 1 else 0.0
        # Stability: 1.0 = highly stable mood, 0.0 = high volatility
        stability_score = max(0.0, min(1.0, 1.0 - math.sqrt(variance)))

        # Consecutive negative days streak (grouped by calendar date)
        # History is ordered by timestamp DESC
        daily_records: Dict[str, List[Dict[str, Any]]] = {}
        for r in history:
            date_key = str(r["timestamp"])[:10]
            if date_key not in daily_records:
                daily_records[date_key] = []
            daily_records[date_key].append(r)

        # Dates ordered descending
        sorted_dates = sorted(daily_records.keys(), reverse=True)
        streak = 0
        for d in sorted_dates:
            day_entries = daily_records[d]
            # If the majority or any severe negative emotion dominated this day
            neg_count = sum(1 for e in day_entries if e["primary_label"] in ("SADNESS", "ANXIETY", "ANGER"))
            if neg_count > len(day_entries) / 2.0:
                streak += 1
            else:
                break

        # Dominant mood
        dominant_p = max(primary_counts.items(), key=lambda x: x[1])[0]

        summary = (
            f"Trong {days} ngày qua ({total} phiên): Cảm xúc chủ đạo là '{dominant_p}' "
            f"({primary_counts[dominant_p]}/{total} lần). "
            f"Điểm cân bằng cảm xúc: {mean_val:+.2f} (độ ổn định: {stability_score*100:.0f}%)."
        )

        return {
            "total_sessions": total,
            "days_analyzed": days,
            "primary_distribution": {k: v / total for k, v in primary_counts.items()},
            "sub_distribution": {k: v / total for k, v in sub_counts.items()},
            "mean_valence": mean_val,
            "stability_score": stability_score,
            "consecutive_negative_streak": streak,
            "summary": summary,
        }
