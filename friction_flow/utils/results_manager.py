import json
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultsManager:
    """Manages simulation results and metrics"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, List[float]] = {
            "emotional_intensity": [],
            "group_cohesion": [],
            "interaction_count": [],
        }
        self.events: List[Dict[str, Any]] = []
        
    def log_metric(self, metric_name: str, value: float) -> None:
        """Log a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def log_event(self, event: Dict[str, Any]) -> None:
        """Log a simulation event"""
        event_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **event
        }
        self.events.append(event_with_timestamp)
        
    def save_results(self) -> None:
        """Save all results to files"""
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save events
        events_file = self.output_dir / "events.json"
        with open(events_file, 'w') as f:
            json.dump(self.events, f, indent=2)
            
        # Generate summary
        summary = self._generate_summary()
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info("Results saved to %s", self.output_dir)
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the simulation results"""
        return {
            "metrics_summary": {
                name: {
                    "mean": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "count": len(values)
                }
                for name, values in self.metrics.items()
            },
            "event_counts": {
                event_type: len([e for e in self.events if e["type"] == event_type])
                for event_type in set(e["type"] for e in self.events)
            },
            "timestamp": datetime.now().isoformat()
        } 