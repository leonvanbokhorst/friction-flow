import click
import logging
from pathlib import Path
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import asyncio
import json
from datetime import datetime

from friction_flow.config.simulation_config import SimulationConfig
from friction_flow.core.simulation_coordinator import SimulationCoordinator
from friction_flow.inputs.agent_inputs import SimulationInputs
from friction_flow.utils.input_visualizer import InputVisualizer

console = Console()

def setup_logging(verbose: bool) -> None:
    """Configure logging with rich output"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

@click.group()
def cli():
    """Social Simulation CLI"""
    pass

@cli.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to configuration file'
)
@click.option(
    '--inputs',
    '-i',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to input file'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    default='simulation_results',
    help='Output directory for results'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def run(config: Path, inputs: Path, output: Path, verbose: bool) -> None:
    """Run the social simulation"""
    setup_logging(verbose)
    asyncio.run(_run_simulation(config, inputs, output))

async def _run_simulation(config: Path, inputs: Path, output: Path) -> None:
    """Async implementation of simulation run"""
    try:
        # Load configuration
        console.print("[bold blue]Loading configuration...[/]")
        with open(config) as f:
            config_dict = yaml.safe_load(f)
        sim_config = SimulationConfig.parse_obj(config_dict)
        
        # Load inputs
        console.print("[bold blue]Loading inputs...[/]")
        with open(inputs) as f:
            input_dict = yaml.safe_load(f)
        sim_inputs = SimulationInputs.parse_raw(json.dumps(input_dict))
        
        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simulation
        console.print("[bold blue]Initializing simulation...[/]")
        coordinator = SimulationCoordinator(sim_config, sim_inputs, output_dir)
        await coordinator.initialize_simulation()
        
        # Run simulation with progress bar
        console.print("[bold blue]Running simulation...[/]")
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[green]Running simulation...",
                total=sim_config.duration
            )
            
            def update_progress(event: dict) -> None:
                if event['type'] == 'step_completed':
                    progress.update(task, advance=1)
                    
            coordinator.attach_observer(update_progress)
            await coordinator.run_simulation(sim_config.duration)
            
        console.print("[bold green]Simulation completed successfully![/]")
        console.print(f"[bold green]Results saved to: {output_dir}[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        raise click.ClickException(str(e))

@cli.command()
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    required=True,
    help='Path to create new configuration file'
)
def generate_config(output: Path) -> None:
    """Generate a default configuration file"""
    config = SimulationConfig()
    
    with open(output, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
        
    console.print(f"[bold green]Configuration file generated at: {output}[/]")

@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
def analyze(results_dir: Path) -> None:
    """Analyze simulation results"""
    console = Console()
    
    try:
        # Load metrics
        metrics_file = results_dir / "metrics.json"
        events_file = results_dir / "events.json"
        summary_file = results_dir / "summary.json"
        
        if not all(f.exists() for f in [metrics_file, events_file, summary_file]):
            raise click.ClickException("Missing result files in directory")
            
        with open(metrics_file) as f:
            metrics = json.load(f)
        with open(events_file) as f:
            events = json.load(f)
        with open(summary_file) as f:
            summary = json.load(f)
            
        # Print analysis
        console.print("\n[bold blue]Simulation Analysis[/]")
        
        # Metrics Summary
        console.print("\n[bold]Metrics Summary:[/]")
        for metric, stats in summary["metrics_summary"].items():
            console.print(f"\n{metric}:")
            console.print(f"  Mean: {stats['mean']:.3f}")
            console.print(f"  Min:  {stats['min']:.3f}")
            console.print(f"  Max:  {stats['max']:.3f}")
            console.print(f"  Count: {stats['count']}")
            
        # Event Analysis    
        console.print("\n[bold]Event Counts:[/]")
        for event_type, count in summary["event_counts"].items():
            console.print(f"  {event_type}: {count}")
            
        # Timeline
        console.print("\n[bold]Timeline:[/]")
        start_time = datetime.fromisoformat(events[0]["timestamp"])
        end_time = datetime.fromisoformat(events[-1]["timestamp"])
        duration = end_time - start_time
        console.print(f"  Duration: {duration}")
        console.print(f"  Events/second: {len(events)/duration.total_seconds():.2f}")
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing results: {str(e)}[/]")
        raise click.ClickException(str(e))

@cli.command()
@click.option(
    '--num-agents',
    '-n',
    type=int,
    default=50,
    help='Number of agents to generate'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    required=True,
    help='Path to create input file'
)
def generate_inputs(num_agents: int, output: Path) -> None:
    """Generate random simulation inputs"""
    inputs = SimulationInputs.generate_random(num_agents)
    
    # Convert to dict and ensure enums are serialized as strings
    input_dict = inputs.model_dump(mode='json')
    
    with open(output, 'w') as f:
        yaml.safe_dump(input_dict, f, default_flow_style=False)
        
    console.print(f"[bold green]Input file generated at: {output}[/]")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    default='input_visualization',
    help='Output directory for visualizations'
)
def visualize_inputs(input_file: Path, output: Path) -> None:
    """Visualize simulation inputs"""
    try:
        # Load inputs
        with open(input_file) as f:
            input_dict = yaml.safe_load(f)
        inputs = SimulationInputs.parse_obj(input_dict)
        
        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        visualizer = InputVisualizer(inputs)
        visualizer.create_network_graph(output_dir / "network.png")
        
        console.print(f"[bold green]Visualizations saved to: {output_dir}[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli() 