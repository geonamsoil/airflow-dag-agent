#!/bin/bash

# Colors for terminal output
GREEN="\033[0;32m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BLUE}🚀 Setting up DAG Generator environment...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists, create if it doesn't
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${CYAN}🔧 Creating virtual environment...${NC}"
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Activate virtual environment
echo -e "${CYAN}🔌 Activating virtual environment...${NC}"
source "$SCRIPT_DIR/venv/bin/activate"

# Upgrade pip, setuptools, and wheel
echo -e "${CYAN}🔄 Upgrading pip, setuptools, and wheel...${NC}"
pip install --upgrade pip setuptools wheel

# Install critical dependencies first
echo -e "${CYAN}📦 Installing critical dependencies...${NC}"
pip install python-dotenv
pip install cython

# Try to install PyYAML separately (often causes issues on Python 3.12)
echo -e "${CYAN}📦 Installing PyYAML...${NC}"
pip install PyYAML --no-build-isolation || echo -e "${YELLOW}⚠️ PyYAML installation failed, but we'll continue...${NC}"

# Install langchain-community (missing dependency)
echo -e "${CYAN}📦 Installing langchain-community...${NC}"
pip install langchain-community || echo -e "${YELLOW}⚠️ langchain-community installation failed${NC}"

# Install core dependencies one by one
echo -e "${CYAN}📦 Installing core dependencies...${NC}"
pip install crewai || echo -e "${YELLOW}⚠️ crewai installation failed${NC}"
pip install langchain || echo -e "${YELLOW}⚠️ langchain installation failed${NC}"
pip install langchain-google-genai || echo -e "${YELLOW}⚠️ langchain-google-genai installation failed${NC}"
pip install langchain-openai || echo -e "${YELLOW}⚠️ langchain-openai installation failed${NC}"
pip install pydantic || echo -e "${YELLOW}⚠️ pydantic installation failed${NC}"
pip install numpy || echo -e "${YELLOW}⚠️ numpy installation failed${NC}"
pip install openai || echo -e "${YELLOW}⚠️ openai installation failed${NC}"
pip install google-generativeai || echo -e "${YELLOW}⚠️ google-generativeai installation failed${NC}"
pip install requests || echo -e "${YELLOW}⚠️ requests installation failed${NC}"
pip install tqdm || echo -e "${YELLOW}⚠️ tqdm installation failed${NC}"

# Try to install docker-compose (optional)
echo -e "${CYAN}📦 Installing docker-compose...${NC}"
pip install docker-compose || echo -e "${YELLOW}⚠️ docker-compose installation failed, but we'll continue...${NC}"

# Try to install apache-airflow (optional for CLI)
echo -e "${CYAN}📦 Installing apache-airflow...${NC}"
pip install apache-airflow || echo -e "${YELLOW}⚠️ apache-airflow installation failed, but we'll continue...${NC}"

# Install the local package in development mode
echo -e "${CYAN}📦 Installing local package in development mode...${NC}"
cd "$SCRIPT_DIR"
pip install -e . || echo -e "${YELLOW}⚠️ Local package installation failed. Creating a setup.py file...${NC}"

# If the local package installation failed, create a setup.py file and try again
if [ ! -f "$SCRIPT_DIR/setup.py" ]; then
    echo -e "${CYAN}📝 Creating setup.py file...${NC}"
    cat > "$SCRIPT_DIR/setup.py" << 'EOF'
from setuptools import setup, find_packages

setup(
    name="crew_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "crewai",
        "langchain",
        "langchain-google-genai",
        "langchain-openai",
        "langchain-community",
        "pydantic",
        "numpy",
        "openai",
        "google-generativeai",
        "python-dotenv",
        "requests",
        "tqdm",
    ],
)
EOF
    echo -e "${CYAN}📦 Trying to install local package again...${NC}"
    pip install -e . || echo -e "${RED}❌ Failed to install local package. You may need to run the script with sudo.${NC}"
fi

# Create a flag file to indicate that we've attempted installation
touch "$SCRIPT_DIR/venv/.requirements_attempted"

# Print instructions for running the main.py script
echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo -e "${YELLOW}To run the DAG Generator, use:${NC}"
echo -e "${CYAN}source $SCRIPT_DIR/venv/bin/activate && python $SCRIPT_DIR/main.py${NC}"
