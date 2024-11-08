import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path

class BeliefVisualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set clean, modern style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12
        })

    def plot_belief_evolution(self, analysis: Dict, evidence_list: List[str], topic: str):
        """Creates a multi-panel visualization of belief evolution."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.5, 1])
        
        # Extract data
        shifts = analysis['belief_shifts']
        steps = [s['step'] for s in shifts]
        magnitudes = [s['shift_magnitude'] for s in shifts]
        conf_changes = [s['confidence_change'] for s in shifts]
        
        # Plot 1: Belief Shift Magnitudes
        ax1.plot(steps, magnitudes, 'o-', color='#2196F3', linewidth=2, 
                label='Belief Similarity', markersize=6)
        ax1.set_ylabel('Similarity to Previous Belief')
        ax1.set_title(f'Belief Evolution for Topic: {topic}')
        ax1.set_ylim(0, 1.05)  # Give a little padding at the top
        
        # Add evidence annotations more cleanly
        for i, evidence in enumerate(evidence_list[1:], 1):
            # Alternate above/below for readability with more space
            if i % 2 == 0:
                y_pos = -0.1
                rotation = 45
                va = 'top'
            else:
                y_pos = 1.1
                rotation = -45
                va = 'bottom'
            
            # Truncate evidence text more aggressively
            trunc_evidence = evidence[:30] + "..." if len(evidence) > 30 else evidence
            
            ax1.annotate(trunc_evidence, 
                        xy=(i, magnitudes[i-1]),
                        xytext=(i, y_pos),
                        ha='right',
                        va=va,
                        rotation=rotation,
                        fontsize=8,
                        arrowprops=dict(
                            arrowstyle='->',
                            connectionstyle='arc3,rad=0.2',
                            alpha=0.6
                        ))

        # Plot 2: Confidence Changes
        cumulative_conf = np.cumsum(conf_changes)
        ax2.plot(steps, cumulative_conf, 'o-', color='#4CAF50', linewidth=2, 
                label='Cumulative Confidence', markersize=6)
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Confidence Change')
        ax2.legend(loc='upper right')

        # Adjust layout
        plt.tight_layout(h_pad=2)
        
        # Save with high DPI for better quality
        plt.savefig(
            self.output_dir / f'{topic.lower().replace(" ", "_")}_evolution.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()