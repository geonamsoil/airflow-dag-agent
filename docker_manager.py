# Handles Docker operations, specifically for Airflow DAG deployment and testing.import warnings
import warnings
warnings.filterwarnings('ignore')
import subprocess
import logging
import tempfile
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DockerManager:
    """Handles Docker operations, specifically for Airflow DAG deployment and testing.
    
    This class provides methods to interact with Docker containers, particularly
    for deploying and testing Airflow DAGs. It handles container status checking,
    file copying, and command execution within containers.
    
    Attributes:
        container_name (str): Name of the Airflow container to interact with.
        docker_available (bool): Whether Docker is available on the system.
    """
    def __init__(self, container_name: str = "airflow-webserver-1", 
                 compose_file: Optional[str] = None,
                 compose_project: str = "airflow"):
        """Initialize the DockerManager.
        
        Args:
            container_name (str): Name of the Airflow container to interact with.
            compose_file (Optional[str]): Path to docker-compose.yml for Airflow.
                If None, will look for standard locations or use docker-compose.yml in the current directory.
            compose_project (str): Docker Compose project name.
        """
        self.container_name = container_name
        self.compose_file = compose_file
        self.compose_project = compose_project
        self.docker_available = self._check_docker_available()
        logger.info(f"DockerManager initialized for container: {self.container_name}")

    def _check_docker_available(self) -> bool:
        """Verify if the Docker daemon is accessible.
        
        Returns:
            bool: True if Docker is available, False otherwise.
        """
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=10)
            logger.info("Docker is available")
            return True
        except Exception as e:
            logger.error(f"Docker not available: {e}")
            return False

    def _run_docker_command(self, command: List[str], check: bool = True, timeout: int = 45) -> Optional[str]:
        """Helper to run Docker commands with logging and error handling.
        
        Args:
            command (List[str]): The Docker command to run as a list of strings.
            check (bool): Whether to check the return code of the command.
            timeout (int): Timeout in seconds for the command.
            
        Returns:
            Optional[str]: Command output if successful, None otherwise.
        """
        logger.debug(f"Running Docker command: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=check, capture_output=True, timeout=timeout)
            return result.stdout.decode('utf-8')
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker command failed with return code {e.returncode}: {e.stderr.decode('utf-8')}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"Docker command timed out after {timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Docker command failed with unexpected error: {str(e)}")
            return None

    def is_container_running(self) -> bool:
        """Check if the target container is running.
        
        Returns:
            bool: True if the container is running, False otherwise.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return False
            
        result = self._run_docker_command(["docker", "ps", "-q", "-f", f"name={self.container_name}"])
        running = bool(result and result.strip())
        logger.info(f"Container '{self.container_name}' running: {running}")
        return running
        
    def is_airflow_running(self) -> bool:
        """Check if the Airflow webserver is running and healthy.

        Returns:
            bool: True if the Airflow webserver is running and healthy, False otherwise.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return False

        if not self.is_container_running():
            logger.error(f"Container '{self.container_name}' is not running")
            return False

        # Check the health of the Airflow webserver
        health_cmd = [
            "docker", "exec", self.container_name,
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "http://localhost:8080/health"
        ]
        health_result = self._run_docker_command(health_cmd, check=False)

        if health_result and health_result.strip() == "200":
            logger.info("Airflow webserver is running and healthy")
            return True
        else:
            logger.warning("Airflow webserver is not healthy or not responding")
            return False

    def start_container(self, wait_for_webserver: bool = True, max_wait_seconds: int = 120) -> Dict[str, Any]:
        """Start container using Docker Compose if it's not already running.
        
        Args:
            wait_for_webserver (bool): Whether to wait for the webserver to be ready.
            max_wait_seconds (int): Maximum time to wait for the webserver to be ready.
            
        Returns:
            Dict[str, Any]: A dictionary with the start status and details.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return {"started": False, "error": "Docker is not available"}
            
        # Check if container is already running
        if self.is_container_running():
            logger.info(f"Container '{self.container_name}' was already running.")
            return {"started": True, "status": "Container was already running"}
            
        # Try to find docker-compose file if not provided
        compose_file = self.compose_file
        if not compose_file:
            # Check common locations
            potential_locations = [
                "./docker-compose.yml",
                "./docker-compose.yaml",
                "../docker-compose.yml",
                "../docker-compose.yaml",
                "/opt/airflow/docker-compose.yml",
                os.path.expanduser("~/airflow/docker-compose.yml")
            ]
            
            for location in potential_locations:
                if os.path.isfile(location):
                    compose_file = location
                    break
                    
        if not compose_file:
            # If no compose file found, try to use docker-compose directly (might be configured in .env)
            logger.warning("No docker-compose file specified or found, trying with default settings")
            
        try:
            # Build the docker-compose up command
            cmd = ["docker-compose"]
            
            if compose_file and os.path.isfile(compose_file):
                cmd.extend(["-f", compose_file])
                
            cmd.extend(["-p", self.compose_project, "up", "-d"])
            
            # Start container
            logger.info(f"Starting container with command: {' '.join(cmd)}")
            start_result = self._run_docker_command(cmd, timeout=90)
            
            if start_result is None:
                logger.error("Failed to start container")
                return {"started": False, "error": "Failed to start container"}
                
            # Wait for the webserver to be ready if requested
            if wait_for_webserver:
                logger.info(f"Waiting for container webserver to be ready (max {max_wait_seconds} seconds)")
                start_time = time.time()
                ready = False
                
                while time.time() - start_time < max_wait_seconds:
                    if self.is_container_running():
                        # Check if the webserver is responding
                        health_cmd = [
                            "docker", "exec", self.container_name,
                            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                            "http://localhost:8080/health"
                        ]
                        health_result = self._run_docker_command(health_cmd, check=False)
                        
                        if health_result and health_result.strip() == "200":
                            ready = True
                            break
                            
                    time.sleep(5)
                    
                if ready:
                    logger.info("Container started and webserver is ready")
                    return {"started": True, "status": "Container started and webserver is ready"}
                else:
                    logger.warning("Container started but webserver might not be ready yet")
                    return {"started": True, "status": "Container started but webserver might not be ready yet"}
            
            logger.info("Container started")
            return {"started": True, "status": "Container started"}
            
        except Exception as e:
            logger.error(f"Failed to start container: {str(e)}")
            return {"started": False, "error": f"Failed to start container: {str(e)}"}

    def start_airflow(self, wait_for_webserver: bool = True, max_wait_seconds: int = 120) -> dict:
        """Alias for start_container for compatibility with client code."""
        return self.start_container(wait_for_webserver=wait_for_webserver, max_wait_seconds=max_wait_seconds)

    def deploy_dag(self, dag_code: str, dag_id: str) -> Dict[str, Any]:
        """Copy the DAG file into the running container's DAGs folder and save a local copy.
        
        Args:
            dag_code (str): The Python code for the DAG as a string.
            dag_id (str): The ID of the DAG, used for the filename.
            
        Returns:
            Dict[str, Any]: A dictionary with the deployment status and details.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return {"deployed": False, "error": "Docker is not available"}
            
        if not self.is_container_running():
            logger.error(f"Container '{self.container_name}' is not running")
            # Try to start container if it's not running
            start_result = self.start_container()
            if not start_result.get("started", False):
                return {"deployed": False, "error": f"Container '{self.container_name}' is not running and could not be started: {start_result.get('error', 'Unknown error')}"}
            
        try:
            # Create a temporary file with the DAG code
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
                temp_file.write(dag_code)
                temp_file_path = temp_file.name
                
            # Copy the file to the container's DAGs folder
            dest_path = f"/opt/airflow/dags/{dag_id}.py"
            copy_cmd = ["docker", "cp", temp_file_path, f"{self.container_name}:{dest_path}"]
            # Run docker cp and capture both stdout and stderr for diagnostics
            try:
                result = subprocess.run(copy_cmd, capture_output=True, timeout=45)
                copy_stdout = result.stdout.decode('utf-8', errors='replace')
                copy_stderr = result.stderr.decode('utf-8', errors='replace')
                copy_returncode = result.returncode
            except Exception as e:
                logger.error(f"Exception during docker cp: {str(e)}")
                copy_stdout = ''
                copy_stderr = str(e)
                copy_returncode = 1

            # Save a local copy of the DAG file
            local_dags_dir = os.path.join(os.getcwd(), "dags")
            os.makedirs(local_dags_dir, exist_ok=True)
            local_dag_path = os.path.join(local_dags_dir, f"{dag_id}.py")
            with open(local_dag_path, "w", encoding="utf-8") as local_file:
                local_file.write(dag_code)
            logger.info(f"Saved local copy of DAG to {local_dag_path}")

            # Clean up the temporary file
            os.unlink(temp_file_path)

            if copy_returncode != 0:
                logger.error(f"Failed to copy DAG file to container.\nSTDOUT: {copy_stdout}\nSTDERR: {copy_stderr}")
                return {"deployed": False, "error": f"Failed to copy DAG file to container. STDOUT: {copy_stdout} STDERR: {copy_stderr}"}

            logger.info(f"DAG '{dag_id}' deployed to container '{self.container_name}' at '{dest_path}'")
            return {
                "deployed": True,
                "dag_id": dag_id,
                "container": self.container_name,
                "destination_path": dest_path,
                "local_path": local_dag_path
            }
        except Exception as e:
            logger.error(f"Failed to deploy DAG: {str(e)}")
            return {"deployed": False, "error": f"Failed to deploy DAG: {str(e)}"}

    def test_dag_syntax(self, dag_id: str) -> Dict[str, Any]:
        """Run 'airflow dags test' or check for DAG bag errors for basic syntax/import checks.
        
        Args:
            dag_id (str): The ID of the DAG to test.
            
        Returns:
            Dict[str, Any]: A dictionary with the test results and details.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return {"success": False, "error": "Docker is not available"}
            
        if not self.is_container_running():
            logger.error(f"Container '{self.container_name}' is not running")
            return {"success": False, "error": f"Container '{self.container_name}' is not running"}
            
        try:
            # First check if the DAG is in the DAG bag (syntax check)
            check_cmd = [
                "docker", "exec", self.container_name,
                "airflow", "dags", "list", "-o", "json"
            ]
            check_result = self._run_docker_command(check_cmd)
            
            if check_result is None:
                logger.error("Failed to list DAGs in container")
                return {"success": False, "error": "Failed to list DAGs in container"}
                
            try:
                dags_list = json.loads(check_result)
                dag_exists = any(dag.get("dag_id") == dag_id for dag in dags_list)
                
                if not dag_exists:
                    logger.error(f"DAG '{dag_id}' not found in container DAG bag. Check for syntax errors.")
                    return {
                        "success": False, 
                        "error": f"DAG '{dag_id}' not found in container DAG bag. Check for syntax errors."
                    }
            except json.JSONDecodeError:
                # If JSON parsing fails, try a different approach
                pass
                
            # Run a test execution of the DAG
            from datetime import datetime
            exec_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            test_cmd = [
                "docker", "exec", self.container_name,
                "airflow", "dags", "test", dag_id, exec_date
            ]
            test_result = self._run_docker_command(test_cmd, check=False)
            
            if test_result and "success" in test_result.lower():
                logger.info(f"DAG '{dag_id}' test successful")
                return {
                    "success": True,
                    "dag_id": dag_id,
                    "result": "DAG test successful",
                    "details": test_result
                }
            else:
                logger.error(f"DAG '{dag_id}' test failed: {test_result or 'No output from test command'}")
                return {
                    "success": False,
                    "dag_id": dag_id,
                    "error": "DAG test failed",
                    "details": test_result or "No output from test command"
                }
                
        except Exception as e:
            logger.error(f"Failed to test DAG: {str(e)}")
            return {"success": False, "error": f"Failed to test DAG: {str(e)}"}
    
    def list_running_containers(self) -> List[Dict[str, str]]:
        """List all running Docker containers.
        
        Returns:
            List[Dict[str, str]]: A list of dictionaries with container details.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return []
            
        cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"]
        result = self._run_docker_command(cmd)
        
        if not result:
            logger.warning("No running containers found")
            return []
            
        containers = []
        for line in result.strip().split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 4:
                    containers.append({
                        "id": parts[0],
                        "name": parts[1],
                        "image": parts[2],
                        "status": parts[3]
                    })
                    
        return containers
    
    def execute_in_container(self, command: List[str]) -> Tuple[bool, str]:
        """Execute a command in a running container.
        
        Args:
            command (List[str]): The command to execute as a list of strings.
            
        Returns:
            Tuple[bool, str]: A tuple with success status and command output/error.
        """
        if not self.docker_available:
            logger.error("Docker is not available")
            return False, "Docker is not available"
            
        # Check if the container is running
        check_cmd = ["docker", "ps", "-q", "-f", f"name={self.container_name}"]
        check_result = self._run_docker_command(check_cmd)
        
        if not check_result or not check_result.strip():
            logger.error(f"Container '{self.container_name}' is not running")
            return False, f"Container '{self.container_name}' is not running"
            
        # Execute the command in the container
        exec_cmd = ["docker", "exec", self.container_name] + command
        result = self._run_docker_command(exec_cmd, check=False)
        
        if result is not None:
            logger.info(f"Command '{' '.join(command)}' executed successfully in container '{self.container_name}'")
            return True, result
        else:
            logger.error(f"Command execution failed in container '{self.container_name}'")
            return False, "Command execution failed"
