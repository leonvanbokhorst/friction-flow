import json
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
import numpy as np
import networkx as nx

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
        
    def analyze_time_series(self, metric_name: str) -> Dict[str, Any]:
        """Analyze time series data for a specific metric"""
        if metric_name not in self.metrics:
            return {}
            
        values = np.array(self.metrics[metric_name])
        return {
            'trend': np.polyfit(range(len(values)), values, 1)[0],
            'volatility': np.std(values),
            'peak': np.max(values),
            'trough': np.min(values),
            'mean': np.mean(values),
            'median': np.median(values)
        }
        
    def analyze_network_metrics(self) -> Dict[str, Any]:
        """Calculate network metrics from interaction events"""
        G = nx.Graph()
        
        # Build network from interaction events
        for event in self.events:
            if event['type'] == 'interaction':
                G.add_edge(
                    event['agent_a'],
                    event['agent_b'],
                    weight=event.get('intensity', 1.0)
                )
                
        return {
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'average_path_length': nx.average_shortest_path_length(G)
            if nx.is_connected(G) else float('inf'),
            'modularity': self._calculate_modularity(G)
        }
        
    def _calculate_modularity(self, G: nx.Graph) -> float:
        """Calculate network modularity using community detection"""
        try:
            communities = nx.community.greedy_modularity_communities(G)
            return nx.community.modularity(G, communities)
        except:
            return 0.0
        
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in metrics"""
        return {
            metric_name: {
                "moving_average": np.convolve(
                    values, 
                    np.ones(10)/10, 
                    mode='valid'
                ).tolist(),
                "trend": np.polyfit(
                    range(len(values)), 
                    values, 
                    1
                )[0]
            }
            for metric_name, values in self.metrics.items()
        }
        
    def analyze_metric_correlations(self) -> Dict[str, float]:
        """Analyze correlations between different metrics"""
        correlations = {}
        metrics_array = np.array([
            self.metrics["emotional_intensity"],
            self.metrics["group_cohesion"],
            self.metrics["interaction_count"]
        ])
        
        corr_matrix = np.corrcoef(metrics_array)
        metric_names = list(self.metrics.keys())
        
        for i in range(len(metric_names)):
            for j in range(i+1, len(metric_names)):
                key = f"{metric_names[i]}_{metric_names[j]}"
                correlations[key] = corr_matrix[i,j]
                
        return correlations