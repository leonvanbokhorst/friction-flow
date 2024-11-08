import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from scipy.interpolate import make_interp_spline
import logging

logger = logging.getLogger(__name__)

class BeliefVisualizer:
    """Visualizes belief evolution and confidence changes over time."""
    
    def __init__(self):
        # Use a built-in style that's clean and modern
        plt.style.use('fivethirtyeight')  # Alternative options: 'bmh', 'ggplot', or just remove this line
        self.fig_size = (12, 8)
        
    def plot_belief_evolution(self, analysis: Dict, evidence_list: List[str], topic: str):
        """
        Creates a two-panel visualization showing belief shifts and confidence changes.
        
        Args:
            analysis: Dictionary containing belief shift analysis
            evidence_list: List of evidence statements processed
            topic: The topic being analyzed
        """
        if not analysis.get("belief_shifts"):
            logger.warning("No belief shifts to visualize")
            return
            
        try:
            # Extract data from analysis
            steps = [shift["step"] for shift in analysis["belief_shifts"]]
            magnitudes = [shift["shift_magnitude"] for shift in analysis["belief_shifts"]]
            conf_changes = [shift["confidence_change"] for shift in analysis["belief_shifts"]]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[1.5, 1])
            fig.suptitle(f'Belief Evolution for Topic: {topic}', fontsize=12)
            
            # Create smooth curves - adjust spline degree based on number of points
            x_smooth = np.linspace(min(steps), max(steps), 300)
            k = min(3, len(steps) - 1)  # Automatically adjust degree based on available points
            
            # Top plot: Belief similarity with smooth spline
            if len(steps) > 2:
                spl = make_interp_spline(steps, magnitudes, k=k)
                y_smooth = spl(x_smooth)
                ax1.plot(x_smooth, y_smooth, '-', color='#2196F3', linewidth=2.5, alpha=0.8)
            else:
                # Fall back to simple line for few points
                ax1.plot(steps, magnitudes, '-', color='#2196F3', linewidth=2.5, alpha=0.8)
                
            ax1.plot(steps, magnitudes, 'o', color='#2196F3', markersize=6, 
                    label='Belief Similarity', zorder=5)
            
            # Add evidence annotations (only show every nth point to avoid crowding)
            n = max(1, len(evidence_list) // 5)  # Show at most 5 annotations
            for i, evidence in enumerate(evidence_list[1:], 1):  
                if i % n == 0:
                    ax1.annotate(f'E{i}', 
                               xy=(i, magnitudes[i-1]),
                               xytext=(0, 10),
                               textcoords='offset points',
                               ha='center',
                               fontsize=8)
            
            # Bottom plot: Confidence changes
            cumulative_conf = np.cumsum(conf_changes)
            if len(steps) > 2:
                spl_conf = make_interp_spline(steps, cumulative_conf, k=k)
                y_smooth_conf = spl_conf(x_smooth)
                ax2.plot(x_smooth, y_smooth_conf, '-', color='#4CAF50', linewidth=2.5, alpha=0.8)
            else:
                # Fall back to simple line for few points
                ax2.plot(steps, cumulative_conf, '-', color='#4CAF50', linewidth=2.5, alpha=0.8)
                
            ax2.plot(steps, cumulative_conf, 'o', color='#4CAF50', markersize=6,
                    label='Cumulative Confidence', zorder=5)
            
            # Customize top plot
            ax1.set_ylabel('Similarity to Previous Belief')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.legend()
            
            # Customize bottom plot
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('Confidence Change')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            ax2.legend()
            
            # Adjust layout
            plt.tight_layout(h_pad=2)
            
            # Save with error handling
            output_path = f'{topic.lower().replace(" ", "_")}_evolution.png'
            try:
                plt.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches='tight'
                )
            except Exception as e:
                raise IOError(f"Failed to save plot to {output_path}: {str(e)}") from e
            finally:
                plt.close()
                
        except Exception as e:
            plt.close()  # Ensure figure is closed even if error occurs
            raise e