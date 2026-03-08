"""
Metrics Tracker - Suivi des métriques du système
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class MetricsTracker:
    """Collecte et stocke les métriques du système."""

    def __init__(self, data_dir: str = "data/monitoring"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.api_calls_file = self.data_dir / "api_calls.json"
        self.latencies_file = self.data_dir / "latencies.json"
        self.predictions_file = self.data_dir / "predictions.json"

        # Charger données existantes
        self.api_calls = self._load_json(self.api_calls_file, [])
        self.latencies = self._load_json(self.latencies_file, [])
        self.predictions = self._load_json(self.predictions_file, [])

    def _load_json(self, filepath: Path, default):
        """Charge fichier JSON."""
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return default
        return default

    def _save_json(self, filepath: Path, data):
        """Sauvegarde fichier JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def track_api_call(
        self,
        service: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        latency: float,
        success: bool = True,
    ):
        """Enregistre un appel API."""
        call = {
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "latency": latency,
            "success": success,
        }
        self.api_calls.append(call)
        self._save_json(self.api_calls_file, self.api_calls)

    def track_latency(
        self, component: str, operation: str, duration: float, metadata: Optional[Dict] = None
    ):
        """Enregistre une latence."""
        latency = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {},
        }
        self.latencies.append(latency)
        self._save_json(self.latencies_file, self.latencies)

    def track_prediction(
        self,
        severity: str,
        age: int,
        sex: str,
        symptoms: List[str],
        red_flags: List[str],
        confidence: float,
    ):
        """Enregistre une prédiction."""
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "patient": {"age": age, "sex": sex},
            "symptoms": symptoms,
            "red_flags": red_flags,
            "confidence": confidence,
        }
        self.predictions.append(prediction)
        self._save_json(self.predictions_file, self.predictions)

    def get_api_stats(self) -> Dict:
        """Statistiques API."""
        if not self.api_calls:
            return {
                "total_calls": 0,
                "total_tokens_input": 0,
                "total_tokens_output": 0,
                "avg_latency": 0,
                "success_rate": 0,
            }

        total_calls = len(self.api_calls)
        total_tokens_input = sum(c["tokens_input"] for c in self.api_calls)
        total_tokens_output = sum(c["tokens_output"] for c in self.api_calls)
        avg_latency = sum(c["latency"] for c in self.api_calls) / total_calls
        success_rate = sum(1 for c in self.api_calls if c["success"]) / total_calls

        return {
            "total_calls": total_calls,
            "total_tokens_input": total_tokens_input,
            "total_tokens_output": total_tokens_output,
            "avg_latency": avg_latency,
            "success_rate": success_rate,
        }

    def get_latency_stats(self) -> Dict:
        """Statistiques latences."""
        if not self.latencies:
            return {}

        stats = {}
        for lat in self.latencies:
            comp = lat["component"]
            if comp not in stats:
                stats[comp] = []
            stats[comp].append(lat["duration"])

        result = {}
        for comp, durations in stats.items():
            result[comp] = {
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "count": len(durations),
            }

        return result

    def get_prediction_stats(self) -> Dict:
        """Statistiques prédictions."""
        if not self.predictions:
            return {"total": 0, "by_severity": {}, "avg_confidence": 0}

        severity_count = {}
        for pred in self.predictions:
            sev = pred["severity"]
            severity_count[sev] = severity_count.get(sev, 0) + 1

        avg_confidence = sum(p["confidence"] for p in self.predictions) / len(self.predictions)

        return {
            "total": len(self.predictions),
            "by_severity": severity_count,
            "avg_confidence": avg_confidence,
        }

    def reset(self):
        """Réinitialise toutes les métriques."""
        self.api_calls = []
        self.latencies = []
        self.predictions = []

        self._save_json(self.api_calls_file, [])
        self._save_json(self.latencies_file, [])
        self._save_json(self.predictions_file, [])

    def export_csv(self, output_dir: str = "data/monitoring/export"):
        """Export CSV des métriques."""
        import csv

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export API calls
        if self.api_calls:
            with open(output_path / "api_calls.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.api_calls[0].keys())
                writer.writeheader()
                writer.writerows(self.api_calls)

        # Export latencies
        if self.latencies:
            with open(output_path / "latencies.csv", "w", newline="", encoding="utf-8") as f:
                rows = []
                for lat in self.latencies:
                    row = {
                        "timestamp": lat["timestamp"],
                        "component": lat["component"],
                        "operation": lat["operation"],
                        "duration": lat["duration"],
                    }
                    rows.append(row)

                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

        # Export predictions
        if self.predictions:
            with open(output_path / "predictions.csv", "w", newline="", encoding="utf-8") as f:
                rows = []
                for pred in self.predictions:
                    row = {
                        "timestamp": pred["timestamp"],
                        "severity": pred["severity"],
                        "age": pred["patient"]["age"],
                        "sex": pred["patient"]["sex"],
                        "confidence": pred["confidence"],
                        "num_red_flags": len(pred["red_flags"]),
                    }
                    rows.append(row)

                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

        return str(output_path)


# Instance globale
_tracker = None


def get_tracker() -> MetricsTracker:
    """Récupère l'instance globale du tracker."""
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker()
    return _tracker
