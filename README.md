# Friction Flow

## Project Overview

Friction Flow is an advanced Python-based research project aimed at developing a framework for analyzing and simulating complex human behavior and group interaction based on Narrative Field Dynamics. This project leverages AI and machine learning techniques, with a focus on integrating Large Language Models (LLMs) for natural language-based decision making and interactions.

## Key Features

1. **Multi-Agent Systems**: Simulates emergent behavior in complex social systems using Graph Attention Networks (GAT)
2. **Psychological Modeling**: Incorporates models of individual and group psychology with emotional states
3. **LLM Integration**: Utilizes language models for natural language processing and generation
4. **Meta-Learning**: Implements Model-Agnostic Meta-Learning (MAML) for rapid adaptation
5. **Social Network Analysis**: Advanced relationship modeling with quantum-inspired dynamics

## Technical Stack

**Python**: Core programming language (version >= 3.12.6 recommended)

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
   git clone https://github.com/leonvanbokhorst/friction-flow
   cd friction-flow
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment:
   - Ensure you have the necessary language models and embeddings set up as specified in `src/config.py`

4. Run the demo:

   ```bash
   python src/gat_social_network.py
   ```

## Core Components

### Social Network Analysis (gat_social_network.py)

Reference to implementation:

Key features:

- Graph Attention Network (GAT) for relationship modeling
- Multi-head attention mechanisms
- Community detection using Louvain method
- Real-time visualization of social dynamics
- Comprehensive metrics tracking

### Meta-Learning Framework (maml_model_agnostic_meta_learning.py)

Reference to implementation:

Key features:

- Model-Agnostic Meta-Learning (MAML) implementation
- Adaptive learning rate scheduling
- Skip connections for improved gradient flow
- Comprehensive visualization capabilities
- Task-specific adaptation

### Narrative Field Dynamics

The project implements three core approaches to narrative field dynamics:

1. **Story Waves**: Quantum-inspired approach to modeling narrative dynamics
2. **Three Story Evolution**: Detailed evolution of interacting stories with emotional states
3. **Simple Lab Scenario**: Practical application in simulated environments

## Development Guidelines

- Follow PEP 8 style guide and use Black for code formatting
- Implement type hints as per PEP 484
- Maintain a minimum of 80% test coverage
- Adhere to SOLID principles
- Use meaningful commit messages following conventional commits format

## Testing

Run the test suite using pytest:

```bash
pytest tests/
```

## CI/CD

The project uses GitHub Actions for continuous integration with:

- Python 3.12.6 setup
- Dependency installation
- Automated testing
- Code quality checks

## Contributing

We welcome contributions. Key points:

- No commented-out code in main branch
- No TODOs in main branch
- Clear variable and function naming
- Adherence to DRY and SOLID principles

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgments

This project builds upon research in cognitive science, complex systems theory, social network analysis, and organizational behavior. Special thanks to the open-source community and the developers of the libraries and tools used in this project.
