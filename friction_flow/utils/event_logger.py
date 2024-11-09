from typing import Dict, Any, List
import json
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EventLogger:
    """Logs and analyzes simulation events"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        
    def log_event(self, event: Dict[str, Any]) -> None:
        """Log a simulation event"""
        event_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **event
        }
        self.events.append(event_with_timestamp)
        
        if len(self.events) >= 1000:
            self._flush_events()
            
    def _flush_events(self) -> None:
        """Write events to disk and clear memory"""
        if not self.events:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"events_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.events, f, indent=2)
            self.events = []
            logger.info("Flushed events to %s", filename)
        except Exception as e:
            logger.error("Failed to flush events: %s", str(e))
            
    def analyze_events(self, event_type: str) -> Dict[str, Any]:
        """Analyze events of a specific type"""
        relevant_events = [
            event for event in self.events
            if event.get("type") == event_type
        ]
        
        return {
            "count": len(relevant_events),
            "first_occurrence": relevant_events[0]["timestamp"] if relevant_events else None,
            "last_occurrence": relevant_events[-1]["timestamp"] if relevant_events else None,
        } 