# Airflow DAG Agent

🤖 An autonomous AI agent system that collaboratively designs, implements, and manages Apache Airflow DAGs through natural language interaction.

## Overview

This project implements a multi-agent AI system that transforms natural language requirements into production-ready Apache Airflow DAGs. Powered by the CrewAI framework, it deploys a team of specialized AI agents that collaborate intelligently to:

1. Analyze requirements and existing codebase
2. Design optimal DAG architecture
3. Implement the DAG following best practices
4. Review and validate the implementation

## Features

- 🤖 AI-powered DAG generation from natural language requirements
- 🔍 Smart codebase analysis to reuse existing patterns and components
- 📝 Comprehensive documentation and inline comments
- ✅ Automatic validation and best practice checks
- 🐳 Integrated Docker support for Airflow deployment
- 🔄 Semantic code search with multiple embedding options
- 🧪 Automated testing of generated DAGs

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- An API key for either Google AI (Gemini) or OpenAI
- Apache Airflow 2.7.1+ (handled via Docker)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd airflow-dag-agent
```

2. Run the setup script:
```bash
chmod +x setup-env.sh
./setup-env.sh
```

3. Create a `.env` file with your API keys:
```env
GOOGLE_API_KEY=your_google_api_key
# or
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Basic Usage

Run the generator with default settings:

```bash
python main.py
```

### Advanced Usage

```bash
python main.py \
  --codebase /path/to/your/codebase \
  --provider google \
  --container airflow-webserver-1 \
  --embedding-provider local
```

### Command Line Options

- `--codebase`: Path to analyze for existing DAGs and patterns
- `--provider`: LLM provider to use (`google` or `openai`)
- `--api-key`: API key for the LLM provider
- `--container`: Name of the Airflow container
- `--no-embeddings`: Disable semantic search
- `--embedding-provider`: Choose embedding provider (`google`, `openai`, or `local`)

## Project Structure

- `client.py`: Main client interface for DAG generation
- `context.py`: Codebase analysis and search functionality
- `tools.py`: AI agent tools for code search and file operations
- `crew_setup.py`: AI agent and crew configuration
- `docker_manager.py`: Docker and Airflow container management
- `main.py`: Command-line interface and user interaction
- `setup-env.sh`: Environment setup script

## Key Components

### AI Agents

1. **Analyst Agent**: Analyzes requirements and existing codebase
2. **Architect Agent**: Designs DAG structure and component selection
3. **Developer Agent**: Implements the DAG following the design
4. **Reviewer Agent**: Validates and suggests improvements

### Tools

- `CodeSearchTool`: Semantic code search in the codebase
- `FileReadTool`: File content access and analysis

## How It Works

1. **Analysis Phase**: The Analyst Agent examines your requirements and searches the codebase for relevant patterns and components.

2. **Design Phase**: The Architect Agent creates a DAG structure that meets your requirements while following Airflow best practices.

3. **Implementation Phase**: The Developer Agent generates the actual DAG code, including proper error handling and documentation.

4. **Review Phase**: The Reviewer Agent validates the DAG, checks for best practices, and suggests improvements.

## Error Handling

The system includes robust error handling for:
- API authentication issues
- Rate limiting
- Timeouts
- Validation errors
- Docker/Airflow deployment issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Support

For issues and feature requests, please use the GitHub issue tracker.

## Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the AI agent orchestration framework
- [Apache Airflow](https://airflow.apache.org/) for the workflow management platform
