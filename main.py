import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import argparse
import logging
from typing import Optional
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Load environment variables early for logging configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
    debug_logging = os.getenv("DEBUG_LOGGING", "FALSE").upper() == "TRUE"
except ImportError:
    debug_logging = False

# Configure logging based on DEBUG_LOGGING setting
if debug_logging:
    logging.basicConfig(level=logging.DEBUG)
    print("Debug logging enabled")
else:
    logging.basicConfig(level=logging.INFO)
    # Suppress verbose messages from libraries
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)



# Handle import errors gracefully
try:
    from dotenv import load_dotenv
except ImportError:
    print("\033[0;33m⚠️ Warning: python-dotenv not found. Environment variables will not be loaded from .env file.\033[0m")
    def load_dotenv():
        print("\033[0;33m⚠️ Using fallback load_dotenv that does nothing.\033[0m")
        pass

try:
    # Try direct import from the local directory
    from client import DAGGeneratorClient
except ImportError as e:
    print(f"\033[0;31m❌ {e}.\033[0m")
    sys.exit(1)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Emojis for better user experience
class Emoji:
    ROCKET = '🚀'
    ROBOT = '🤖'
    BRAIN = '🧠'
    CHECK = '✅'
    ERROR = '❌'
    WAIT = '⏳'
    SEARCH = '🔍'
    WRENCH = '🔧'
    SPARKLES = '✨'
    THINKING = '💭'
    DOCKER = '🐳'
    AIRFLOW = '🌬️'
    PYTHON = '🐍'
    FOLDER = '📁'
    DOCUMENT = '📄'

def print_header():
    """Print a fancy header for the application."""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    print(f"{Emoji.ROCKET} Airflow DAG Generator {Emoji.ROBOT}")
    print("=" * 80)
    print(f"{Colors.ENDC}")

def print_step(message: str, emoji: str = Emoji.ROCKET):
    """Print a step in the process with an emoji."""
    print(f"\n{Colors.BLUE}{emoji} {message}{Colors.ENDC}")

def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}{Emoji.CHECK} {message}{Colors.ENDC}")

def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}{Emoji.ERROR} {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}{Emoji.WAIT} {message}{Colors.ENDC}")

def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.CYAN}{message}{Colors.ENDC}")

class Spinner:
    """A class to manage a spinner animation in a separate thread."""
    def __init__(self, message):
        """Initialize the spinner.
        
        Args:
            message (str): The message to display alongside the spinner.
        """
        self.message = message
        self.spinner_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        self.running = False
        self.thread = None
        self.start_time = None
    
    def _spin(self):
        """The spinner animation function that runs in a thread."""
        i = 0
        self.start_time = time.time()
        
        while self.running:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            timeformat = f"{mins:02d}:{secs:02d}"
            
            sys.stdout.write(f"\r{Colors.CYAN}{self.spinner_chars[i]} {self.message} ({timeformat}){Colors.ENDC}")
            sys.stdout.flush()
            i = (i + 1) % len(self.spinner_chars)
            time.sleep(0.1)
    
    def start(self):
        """Start the spinner animation in a separate thread."""
        if self.thread is not None:
            return  # Already running
        
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True  # Thread will exit when main thread exits
        self.thread.start()
    
    def stop(self):
        """Stop the spinner animation."""
        self.running = False
        if self.thread:
            # Give the thread a moment to exit cleanly
            time.sleep(0.2)
            self.thread = None
        
        # Clear the spinner line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

def generate_docker_compose():
    """Generate a docker-compose.yml file for Airflow if it doesn't exist."""
    docker_compose_path = os.path.join(os.getcwd(), "docker-compose.yml")
    
    if os.path.exists(docker_compose_path):
        print_info(f"Docker Compose file already exists at {docker_compose_path}")
        return docker_compose_path
    
    print_step("Generating Docker Compose file for Airflow", Emoji.DOCKER)
    
    # Basic Airflow docker-compose configuration
    docker_compose_content = '''
# Basic Airflow docker-compose.yml
version: '3'

x-airflow-common: &airflow-common
  image: apache/airflow:2.7.1
  environment:
    - AIRFLOW__CORE__EXECUTOR=LocalExecutor
    - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
    - AIRFLOW__CORE__FERNET_KEY=
    - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true
    - AIRFLOW__CORE__LOAD_EXAMPLES=false
    - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
  depends_on:
    - postgres

services:
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler

  airflow-init:
    <<: *airflow-common
    command: version
    environment:
      - _AIRFLOW_DB_UPGRADE=true
      - _AIRFLOW_WWW_USER_CREATE=true
      - _AIRFLOW_WWW_USER_USERNAME=airflow
      - _AIRFLOW_WWW_USER_PASSWORD=airflow

volumes:
  postgres-db-volume:
'''
    
    # Create directories for Airflow
    os.makedirs(os.path.join(os.getcwd(), "dags"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "plugins"), exist_ok=True)
    
    # Write docker-compose.yml
    with open(docker_compose_path, "w") as f:
        f.write(docker_compose_content)
    
    print_success(f"Docker Compose file created at {docker_compose_path}")
    print_info("You can start Airflow with 'docker-compose up -d'")
    
    return docker_compose_path

def get_api_key(args):
    """Get API key from args, environment, or prompt user."""
    # Try to get from args
    api_key = args.api_key
    
    # Try to get from environment
    if not api_key:
        if args.provider.lower() == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
        else:  # openai
            api_key = os.getenv("OPENAI_API_KEY")
    
    # Prompt user if still not found
    if not api_key:
        provider_name = "Google" if args.provider.lower() == "google" else "OpenAI"
        api_key = input(f"{Colors.CYAN}Please enter your {provider_name} API key: {Colors.ENDC}")
    
    return api_key

def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Airflow DAGs using CrewAI")
    parser.add_argument("--codebase", type=str, default="E:\\agents\\cew\\crew-ai\\crew_ai\\dags",
                        help="Path to the codebase to analyze (default: current directory)")
    parser.add_argument("--provider", type=str, choices=["google", "openai"], default="google",
                        help="LLM provider to use (default: google)")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider")
    parser.add_argument("--container", type=str, default="airflow-airflow-webserver-1",
                        help="Name of the Airflow container (default: airflow-airflow-webserver-1)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Disable embeddings for semantic search")
    parser.add_argument("--embedding-provider", type=str, choices=["google", "openai", "local"], default="local",
                        help="Embedding provider to use (default: local)")
    
    args = parser.parse_args()
    
    print_header()
    
    # Generate docker-compose.yml if it doesn't exist
    try:
        docker_compose_path = generate_docker_compose()
    except Exception as e:
        print_warning(f"Failed to generate Docker Compose file: {str(e)}")
        docker_compose_path = os.path.join(os.getcwd(), "docker-compose.yml")
    
    # Get API key
    api_key = get_api_key(args)
    if not api_key:
        print_error("No API key provided. Exiting.")
        sys.exit(1)
    
    print_step("Initializing DAG Generator", Emoji.BRAIN)
    print_info(f"Using {args.provider.capitalize()} as the LLM provider")
    print_info(f"Analyzing codebase at {args.codebase}")
    
    try:
        # Initialize the client
        client = DAGGeneratorClient(
            api_key=api_key,
            codebase_path=args.codebase,
            airflow_container_name=args.container,
            llm_provider=args.provider,
            airflow_compose_file=docker_compose_path,
            use_embeddings=not args.no_embeddings,
            embedding_provider=args.embedding_provider
        )
        
        print_success("DAG Generator initialized successfully")
        
        # Get user requirements
        print_step("What kind of DAG would you like to generate?", Emoji.THINKING)
        requirements = input(f"{Colors.CYAN}Please describe your DAG requirements: {Colors.ENDC}")
        
        if not requirements.strip():
            print_error("No requirements provided. Exiting.")
            sys.exit(1)
        
        # Generate the DAG
        print_step("Generating DAG based on your requirements", Emoji.SPARKLES)
        print_info("This may take a few minutes...")
        
        # Start spinner
        spinner = Spinner("Generating DAG")
        spinner.start()
        
        try:
            # Generate the DAG
            result = client.generate_dag(requirements)
        finally:
            # Stop spinner - this will be called even if an exception occurs
            spinner.stop()
        
        if result["success"]:
            print_success("DAG generated successfully!")
            print_info(f"DAG ID: {result['dag_id']}")
            
            # Save DAG to dags folder before deployment
            dags_dir = os.path.join(os.getcwd(), "dags")
            os.makedirs(dags_dir, exist_ok=True)
            dag_file_path = os.path.join(dags_dir, f"{result['dag_id']}.py")
            try:
                with open(dag_file_path, "w", encoding="utf-8") as dag_file:
                    dag_file.write(result["dag_code"])
                print_success(f"DAG saved to {dag_file_path}")
            except Exception as e:
                print_warning(f"Failed to save DAG to {dag_file_path}: {e}")
            
            # Show any validation warnings
            if result.get("validation_warnings"):
                print_warning("Validation Warnings:")
                for warning in result["validation_warnings"]:
                    print_warning(f"- {warning}")
                print("")
            
            # Show analysis report
            if result.get("analysis_report"):
                print_step("Analysis Report", Emoji.DOCUMENT)
                print_info(result["analysis_report"][:500] + "..." if len(result["analysis_report"]) > 500 else result["analysis_report"])
            
            # Show the generated code
            print_step("Generated DAG Code", Emoji.PYTHON)
            print(f"{Colors.CYAN}```python\n{result['dag_code']}\n```{Colors.ENDC}")
            
            # Ask if user wants to deploy
            deploy = input(f"\n{Colors.CYAN}Do you want to deploy this DAG to Airflow? (y/n): {Colors.ENDC}").lower()
            
            if deploy == 'y':
                print_step("Deploying DAG to Airflow", Emoji.AIRFLOW)
                
                # Start spinner
                deploy_spinner = Spinner("Deploying DAG")
                deploy_spinner.start()
                
                try:
                    # Deploy the DAG
                    deploy_result = client.deploy_and_test_dag(
                        dag_code=result["dag_code"],
                        dag_id=result["dag_id"]
                    )
                finally:
                    # Stop spinner - this will be called even if an exception occurs
                    deploy_spinner.stop()
                
                if deploy_result["deployed"]:
                    print_success("DAG deployed successfully!")
                    print_info(f"You can view your DAG at http://localhost:8080/dags/{result['dag_id']}")
                    
                    if deploy_result.get("test_success"):
                        print_success("DAG syntax test passed!")
                    else:
                        print_warning(f"DAG syntax test failed: {deploy_result.get('error', 'Unknown error')}")
                else:
                    print_error(f"Failed to deploy DAG: {deploy_result.get('error', 'Unknown error')}")
            else:
                print_info("DAG not deployed.")
        else:
            print_error(f"Failed to generate DAG: {result.get('error_category', 'Error')}")
            print_error(f"Details: {result.get('error', 'Unknown error')}")
            
            if result.get('suggestion'):
                print_info(f"Suggestion: {result.get('suggestion')}")
    
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print_step("All done!", Emoji.ROCKET)

if __name__ == "__main__":
    main()
