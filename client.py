# Client interface for interacting with the DAG generation crew.
import warnings
warnings.filterwarnings('ignore')
import logging
import os
import re
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt
import time

from crew_setup import create_dag_generator_crew

logger = logging.getLogger(__name__)

class AgentOutput(BaseModel):
    """Structured output schema for agent results"""
    dag_code: str
    analysis: Optional[str] = None
    review: Optional[str] = None

class DAGGeneratorClient:
    """Client interface for interacting with the DAG generation crew.
    
    This class provides a high-level interface for generating, deploying, and testing
    Airflow DAGs using the CrewAI framework. It orchestrates the collaboration between
    different AI agents to analyze requirements, design, implement, and review DAGs.
    
    Attributes:
        api_key (str): The API key for the LLM provider.
        codebase_path (str): Path to the codebase to analyze.
        airflow_container_name (str): Name of the Airflow container for deployment.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        embedding_provider (str): The embedding provider to use ('google', 'openai', or 'local').
        crew: The CrewAI crew for DAG generation.
        docker_manager: The Docker manager for container operations.
        codebase_context: The codebase context for code analysis.
    """
    def __init__(self, 
                 api_key: str, 
                 codebase_path: str, 
                 airflow_container_name: str = "airflow-airflow-webserver-1",
                 llm_provider: str = "google",
                 airflow_compose_file: Optional[str] = None,
                 use_embeddings: bool = True,
                 embedding_provider: str = "local",
                 requirements: str = ""):
        """Initialize the DAGGeneratorClient.
        
        Args:
            api_key (str): The API key for the LLM provider (Google or OpenAI).
            codebase_path (str): Path to the codebase to analyze.
            airflow_container_name (str): Name of the Airflow container for deployment.
            llm_provider (str): The LLM provider to use ('google' or 'openai').
            airflow_compose_file (Optional[str]): Path to docker-compose.yml for Airflow.
            use_embeddings (bool): Whether to use embeddings for semantic search.
            embedding_provider (str): The embedding provider to use ('google', 'openai', or 'local').
            requirements (str): User requirements for the DAG in natural language.
            
        Raises:
            ValueError: If the API key or codebase path is invalid.
        """
        logger.info(f"Initializing DAGGeneratorClient with codebase: {codebase_path}, "
                   f"Airflow container: {airflow_container_name}, LLM provider: {llm_provider}")
        
        if not api_key or api_key == "your_api_key":
            logger.error("API Key is missing or using the placeholder value.")
            raise ValueError(f"Please provide a valid {llm_provider.capitalize()} API Key.")
            
        if not codebase_path or codebase_path == "/path/to/your/codebase" or not os.path.isdir(codebase_path):
            logger.error(f"Codebase path ('{codebase_path}') is missing, placeholder, or not a valid directory.")
            raise ValueError("Please provide a valid path to the codebase directory.")
            
        self.api_key = api_key
        self.codebase_path = codebase_path
        self.airflow_container_name = airflow_container_name
        self.llm_provider = llm_provider.lower()
        self.use_embeddings = use_embeddings
        self.embedding_provider = embedding_provider.lower()
        self.requirements = requirements
        
        # Validate LLM provider
        if self.llm_provider not in ["google", "openai"]:
            logger.warning(f"Unknown LLM provider: {llm_provider}, defaulting to Google")
            self.llm_provider = "google"
            
        # Validate embedding provider
        if self.embedding_provider not in ["google", "openai", "local"]:
            logger.warning(f"Unknown embedding provider: {embedding_provider}, defaulting to local")
            self.embedding_provider = "local"
        
        # Remove crew initialization from here
        self.crew = None
        self.docker_manager = None
        self.codebase_context = None

    def _extract_code_from_output(self, text: Optional[str]) -> Optional[str]:
        """Extracts Python code block from LLM markdown output using regex.
        
        Args:
            text (Optional[str]): The text to extract code from.
            
        Returns:
            Optional[str]: The extracted Python code, or None if no code was found.
        """
        if not text:
            return None
            
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```(?:python)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)
        
        if matches:
            # Return the longest code block (assuming it's the most complete)
            return max(matches, key=len).strip()
            
        # If no code blocks found, try to extract based on Python syntax
        if 'import airflow' in text or 'from airflow' in text:
            # Extract lines that look like Python code
            lines = text.split('\n')
            code_lines = []
            in_code_section = False
            
            for line in lines:
                if line.strip().startswith('import ') or line.strip().startswith('from ') or \
                   line.strip().startswith('def ') or line.strip().startswith('class ') or \
                   '=' in line or line.strip().startswith('#'):
                    in_code_section = True
                    code_lines.append(line)
                elif in_code_section and line.strip():
                    code_lines.append(line)
                    
            if code_lines:
                return '\n'.join(code_lines)
                
        return None

    def _get_dag_id_from_code(self, code: str, default_id: str = "generated_dag") -> str:
        """Attempts to extract the dag_id from the generated Python code.
        
        Args:
            code (str): The DAG code to extract the ID from.
            default_id (str): The default ID to use if extraction fails.
            
        Returns:
            str: The extracted DAG ID, or the default ID if extraction fails.
        """
        if not code:
            return default_id
            
        # Try to find dag_id in DAG initialization
        dag_id_pattern = r'["\']dag_id["\']\s*:\s*["\']([\w_-]+)["\']'
        matches = re.findall(dag_id_pattern, code)
        if matches:
            return matches[0]
            
        # Alternative pattern: dag = DAG('dag_id', ...)
        alt_pattern = r'DAG\s*\(\s*["\']([\w_-]+)["\']'
        matches = re.findall(alt_pattern, code)
        if matches:
            return matches[0]
            
        return default_id

    def _sanitize_input(self, requirements: str) -> str:
        """Ensure input is safe and reasonable size"""
        return requirements.strip()[:5000]  # Max 5000 chars

    def _validate_agent_output(self, raw_output: dict) -> AgentOutput:
        """Validate agent output matches expected schema"""
        try:
            return AgentOutput(**raw_output)
        except ValidationError as e:
            logger.error(f"Agent output validation failed: {e}")
            raise ValueError(f"Invalid agent output structure: {e.errors()}")

    @retry(stop=stop_after_attempt(3))
    def generate_dag(self, requirements: str) -> Dict[str, Any]:
        """Generate DAG with enhanced validation and error handling
        
        Args:
            requirements (str): User requirements for the DAG in natural language
            
        Returns:
            Dict[str, Any]: Dictionary with generation results including success status,
                           DAG code, DAG ID, and analysis report.
        """
        requirements = self._sanitize_input(requirements)
        
        logger.info(f"Generating DAG for requirements: {requirements[:100]}...")
        start_time = time.time()
        
        # Always re-initialize crew and context with latest requirements
        self.crew, self.docker_manager, self.codebase_context = create_dag_generator_crew(
            api_key=self.api_key,
            codebase_path=self.codebase_path,
            airflow_container_name=self.airflow_container_name,
            llm_provider=self.llm_provider,
            airflow_compose_file=None,
            use_embeddings=self.use_embeddings,
            embedding_provider=self.embedding_provider,
            requirements=requirements
        )
        
        try:
            # Progress update - Starting crew execution
            logger.info("Starting CrewAI agents to generate DAG...")
            
            # Run the crew to generate the DAG with retry for API calls
            try:
                result = self.crew.kickoff(inputs={"requirements": requirements})
                logger.info("CrewAI execution completed successfully")
            except Exception as e:
                logger.error(f"Error during CrewAI execution: {str(e)}")
                raise
            
            # Progress update - Processing task outputs
            logger.info("Processing agent outputs...")
            
            # Get the results from specific tasks
            # In the latest CrewAI version, tasks_output is a list of TaskOutput objects
            tasks_output = result.tasks_output or []
            
            # Improved task output extraction with more robust identification
            task_outputs_dict = {}
            
            # First pass: try to map tasks by ID if available
            for task_output in tasks_output:
                if hasattr(task_output, 'id') and task_output.id:
                    logger.debug(f"Found task with ID: {task_output.id}")
                    task_outputs_dict[task_output.id] = str(task_output)
                
            # Second pass: fallback to description-based matching if needed
            if not all(k in task_outputs_dict for k in ['analyze', 'implement', 'review']):
                logger.debug("Using description-based task matching")
                for task_output in tasks_output:
                    if hasattr(task_output, 'description'):
                        task_desc = task_output.description.lower()
                        task_content = str(task_output) if task_output else ""
                        
                        if "analyze" in task_desc and "requirements" in task_desc:
                            logger.debug("Found analysis task output")
                            task_outputs_dict['analyze'] = task_content
                        elif "implement" in task_desc and "airflow" in task_desc:
                            logger.debug("Found implementation task output")
                            task_outputs_dict['implement'] = task_content
                        elif "review" in task_desc and "validate" in task_desc:
                            logger.debug("Found review task output")
                            task_outputs_dict['review'] = task_content
            
            # Extract the results
            analysis_result = task_outputs_dict.get('analyze', '')
            implementation_result = task_outputs_dict.get('implement', '')
            review_result = task_outputs_dict.get('review', '')
            
            # Fallback to string(result) if we couldn't extract task outputs
            if not implementation_result:
                logger.warning("Could not extract implementation task output, using string representation of result")
                implementation_result = str(result)

            # Progress update - Extracting DAG code
            logger.info("Extracting DAG code from agent output...")
            
            # Extract code from implementation task result
            dag_code = self._extract_code_from_output(implementation_result)
            
            # If we couldn't extract any code, try to extract from the entire result
            if not dag_code and isinstance(str(result), str):
                logger.warning("Attempting to extract code from string representation of result as fallback")
                dag_code = self._extract_code_from_output(str(result))
            
            # If we still couldn't extract any code, return error
            if not dag_code:
                logger.error("Failed to extract DAG code from any output")
                return {
                    "success": False,
                    "error": "Failed to extract DAG code from generation result"
                }
            
            # Progress update - Validating DAG code
            logger.info("Validating generated DAG code...")
            
            # Basic validation checks
            validation_errors = []
            
            # Check for required imports
            if "from airflow import DAG" not in dag_code:
                validation_errors.append("Missing required import: 'from airflow import DAG'")
            
            # Check for DAG instantiation
            if "DAG(" not in dag_code:
                validation_errors.append("Missing DAG instantiation")
                
            # Check for operators
            if "Operator" not in dag_code and "operator" not in dag_code.lower():
                validation_errors.append("No Airflow operators found in DAG code")
            
            # Extract the DAG ID from the code
            dag_id = self._get_dag_id_from_code(dag_code)
            if not dag_id or dag_id == "generated_dag":
                logger.warning(f"Using generic DAG ID: {dag_id}")
            else:
                logger.info(f"Extracted DAG ID: {dag_id}")
            
            # Construct a clean analysis report
            analysis_report = analysis_result
            if review_result:
                analysis_report += "\n\n## Review Report\n" + review_result
            
            # Add validation results to the report if there are any errors
            if validation_errors:
                validation_warnings = "\n\n## Validation Warnings\n"
                for error in validation_errors:
                    validation_warnings += f"- {error}\n"
                analysis_report += validation_warnings
                logger.warning(f"Validation warnings: {', '.join(validation_errors)}")
            
            # Progress update - Finalizing results
            logger.info(f"DAG generation completed successfully in {time.time() - start_time:.2f} seconds")
            
            return {
                "success": True,
                "dag_code": dag_code,
                "dag_id": dag_id,
                "analysis_report": analysis_report,
                "validation_warnings": validation_errors if validation_errors else None,
                "execution_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error generating DAG: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Traceback: {error_traceback}")
            
            # Categorize the error for better user feedback
            error_type = type(e).__name__
            error_message = str(e)
            error_category = "Unknown error"
            suggestion = "Please try again or modify your requirements."
            
            if "API key" in error_message or "authentication" in error_message.lower():
                error_category = "API Authentication Error"
                suggestion = "Please check your API key and ensure it's valid."
            elif "timeout" in error_message.lower():
                error_category = "Timeout Error"
                suggestion = "The request took too long. Try simplifying your requirements or try again later."
            elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
                error_category = "Rate Limit Error"
                suggestion = "You've exceeded the API rate limit. Please wait a while and try again."
            elif "validation" in error_message.lower():
                error_category = "Validation Error"
                suggestion = "There was an issue with the generated DAG structure. Try more specific requirements."
            
            return {
                "success": False,
                "error": error_message,
                "error_type": error_type,
                "error_category": error_category,
                "suggestion": suggestion,
                "input_sample": requirements[:500],
                "execution_time": time.time() - start_time
            }

    def deploy_and_test_dag(self, dag_code: Optional[str], dag_id: Optional[str]) -> Dict[str, Any]:
        """Deploys the generated DAG to Airflow and runs a basic check.
        
        This method will attempt to start Airflow if it's not already running.
        
        Args:
            dag_code (Optional[str]): The Python code string for the DAG.
            dag_id (Optional[str]): The DAG ID to use for deployment and testing.
            
        Returns:
            Dict[str, Any]: A dictionary with deployment and test results.
            { "deployed": bool, "test_result": str | None, "error": str | None }
        """
        if not dag_code:
            return {"deployed": False, "error": "No DAG code provided"}
            
        if not dag_id:
            dag_id = self._get_dag_id_from_code(dag_code)
            
        logger.info(f"Deploying DAG with ID: {dag_id}")
        
        # Check if Airflow is running, if not try to start it
        if not self.docker_manager.is_airflow_running():
            logger.info("Airflow is not running, attempting to start it")
            start_result = self.docker_manager.start_airflow()
            
            if not start_result.get("started", False):
                return {
                    "deployed": False,
                    "error": f"Failed to start Airflow: {start_result.get('error', 'Unknown error')}"
                }
                
            logger.info(f"Airflow start result: {start_result.get('status', 'Unknown')}")
            
        # Deploy the DAG
        deploy_result = self.docker_manager.deploy_dag(dag_code, dag_id)
        
        if not deploy_result.get("deployed", False):
            return deploy_result
            
        # Test the DAG syntax
        test_result = self.docker_manager.test_dag_syntax(dag_id)
        
        return {
            "deployed": True,
            "dag_id": dag_id,
            "test_success": test_result.get("success", False),
            "test_result": test_result.get("result", None),
            "test_details": test_result.get("details", None),
            "error": test_result.get("error", None)
        }
    
    def list_available_containers(self) -> List[Dict[str, str]]:
        """Lists all available Docker containers.
        
        Returns:
            List[Dict[str, str]]: A list of dictionaries with container details.
        """
        return self.docker_manager.list_running_containers()
    
    def reload_codebase_context(self) -> bool:
        """Reloads the codebase context to reflect any changes.
        
        Returns:
            bool: True if the reload was successful, False otherwise.
        """
        try:
            self.codebase_context.load_code_context()
            return True
        except Exception as e:
            logger.error(f"Error reloading codebase context: {str(e)}")
            return False
            
    def get_codebase_components(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all code components (classes and functions) from the codebase.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with classes and functions.
        """
        return self.codebase_context.get_code_components()
    
    def search_codebase(self, query: str, max_results: int = 5, semantic: bool = True) -> Dict[str, Any]:
        """Search the codebase for files matching the query.
        
        Args:
            query (str): The search query string.
            max_results (int): Maximum number of results to return.
            semantic (bool): Whether to use semantic search with embeddings.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and metadata.
        """
        return self.codebase_context.search_context(query, max_results, semantic)
