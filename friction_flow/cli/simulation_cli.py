import click
import logging
from pathlib import Path
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import asyncio

from friction_flow.config.simulation_config import SimulationConfig
from friction_flow.core.simulation_coordinator import SimulationCoordinator

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
def run(config: Path, output: Path, verbose: bool) -> None:
    """Run the social simulation"""
    setup_logging(verbose)
    asyncio.run(_run_simulation(config, output))

async def _run_simulation(config: Path, output: Path) -> None:
    """Async implementation of simulation run"""
    try:
        # Load configuration
        console.print("[bold blue]Loading configuration...[/]")
        with open(config) as f:
            config_dict = yaml.safe_load(f)
        sim_config = SimulationConfig.parse_obj(config_dict)
        
        # Initialize simulation
        console.print("[bold blue]Initializing simulation...[/]")
        coordinator = SimulationCoordinator(sim_config)
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

if __name__ == '__main__':
    cli() 