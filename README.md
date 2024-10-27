# Friction Flow

## Project Overview

Friction Flow is an advanced Python-based research project aimed at developing a framework for analyzing and simulating complex human behavior and group dynamics based on Narrative Field Dynamics. This project leverages AI and machine learning techniques, with a focus on integrating Large Language Models (LLMs) for natural language-based decision making and interactions.

## Key Features

1. **Multi-Agent Systems**: Simulates emergent behavior in complex social systems.
2. **Psychological Modeling**: Incorporates advanced models of individual and group psychology.
3. **LLM Integration**: Utilizes state-of-the-art language models for natural language processing and generation.
4. **Distributed Computing**: Employs event-driven architectures for scalable simulations.
5. **Machine Learning Components**: Includes neural networks and other ML techniques for behavior prediction and analysis.

## Technical Stack

 **Python**: Core programming language (version >= 3.12 recommended)
- **PyTorch**: For neural network components and tensor operations
- **Transformers**: For integration with pre-trained language models
- **Ray**: For distributed computing
- **FastAPI**: For service endpoints
- **Redis**: For state management
- **Ollama**: For local LLM integration
- **ChromaDB**: For vector storage and similarity search

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-organization/friction-flow.git
   cd friction-flow
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment:
   - Ensure you have the necessary language models and embeddings set up as specified in `src/config.py`.

4. Run the simulation:

   ```bash
   python src/nfs_simple_lab_scenario.py
   ```

## Project Structure

- `src/`: Contains the core source code
  - `nfs_story_waves.py`: Simulation components for narrative field dynamics
  - `nfs_simple_lab_scenario.py`: Example scenario implementation
  - `language_models.py`: Interfaces for various language models
  - `config.py`: Configuration settings
- `tests/`: Unit and integration tests
- `pocs/`: Proof of concept implementations
- `.github/`: Issue templates and CI/CD workflows

## Development Guidelines

- Follow PEP 8 style guide and use Black for code formatting.
- Implement type hints as per PEP 484.
- Maintain a minimum of 80% test coverage.
- Adhere to SOLID principles and maintain clear separation of concerns.
- Use meaningful commit messages following the conventional commits format.

## Testing

Run the test suite using pytest:

```bash
pytest tests/
```

## CI/CD

The project uses GitHub Actions for continuous integration. The workflow includes:
- Setting up Python 3.12.6
- Installing dependencies
- Running tests

## Contributing

We welcome contributions to the Friction Flow project. Please read our contributing guidelines before submitting pull requests. Key points:

- No commented-out code in the main branch
- No TODOs in the main branch
- Clear variable and function naming
- Adherence to DRY and SOLID principles

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgments

This project builds upon research in cognitive science, complex systems theory, social network analysis, and organizational behavior. We acknowledge the contributions of the open-source community and the developers of the libraries and tools used in this project.
