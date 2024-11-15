from setuptools import setup, find_packages

setup(
    name="mas_learn",
    version="0.1",
    packages=find_packages(),
    install_requires=["torch", "langchain", "chromadb", "sentence-transformers"],
)

# setup(
#     name="friction-flow",
#     version="0.1.0",
#     packages=find_packages(),
#     install_requires=["click>=8.0.0", "pydantic>=2.0.0", "PyYAML>=6.0", "rich>=10.0.0"],
#     entry_points={
#         "console_scripts": [
#             "friction-flow=friction_flow.cli.simulation_cli:cli",
#         ],
#     },
# )
