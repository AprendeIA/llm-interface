"""Setup script for LLM Interface package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback requirements if file not found
    requirements = [
        "langgraph>=0.1.0",
        "pydantic>=2.0.0", 
        "langchain-community>=0.0.1",
        "langchain-ollama>=0.0.1",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "PyYAML>=6.0"
    ]

setup(
    name="llm-interface",
    version="1.0.0",
    author="Aprende IA",
    author_email="aprendetodoia@example.com",
    description="A flexible Python library for handling multiple LLM providers with LangGraph integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/AprendeIA/llm-interface",
    packages=find_packages(include=['llm_interface', 'llm_interface.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "crewai": [
            "crewai>=0.1.0",
        ],
        "autogen": [
            "pyautogen>=0.2.0",
        ],
        "semantic-kernel": [
            "semantic-kernel>=0.4.0",
        ],
        "all-frameworks": [
            "crewai>=0.1.0",
            "pyautogen>=0.2.0",
            "semantic-kernel>=0.4.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "llm-interface=llm_interface.cli:main",
    #     ],
    # },
    include_package_data=True,
    package_data={
        "llm_interface": ["*.yaml", "*.yml"],
    },
    keywords="llm, langchain, ai, machine learning, openai, anthropic, azure, ollama",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-interface/issues",
        "Source": "https://github.com/yourusername/llm-interface",
        "Documentation": "https://llm-interface.readthedocs.io/",
    },
)