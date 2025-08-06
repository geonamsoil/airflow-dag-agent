# Creates and configures the CrewAI crew for DAG generation.
import warnings
warnings.filterwarnings('ignore')
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Turn off CrewAI telemetry as early as possible
Crew.telemetry_enabled = False

# Local imports
from tools import CodeSearchTool, FileReadTool
from context import CodebaseContext

from docker_manager import DockerManager


logger = logging.getLogger(__name__)

def get_llm_config(api_key: str, llm_provider: str = "google", temperature: float = 0.2):
    """Create an LLM instance based on the provider.
    
    Args:
        api_key (str): API key for the LLM provider.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        temperature (float): Temperature for the LLM.
        
    Returns:
        The configured LLM instance.
    """
    if llm_provider.lower() == "openai":
      
        
        return LLM(
            model="azure/gpt-4.1",
            api_version="2023-05-15",
            # http_client=client
        )
    else:  # Default to Google
        # Use CrewAI's LLM class with the correct model format for Google
        # Using a more capable Gemini model for complex tasks like DAG generation
        
        import json
        file_path = ''

        # Load the JSON file
        with open(file_path, 'r') as file:
            vertex_credentials = json.load(file)

        # Convert to JSON string
        vertex_credentials_json = json.dumps(vertex_credentials)
        return LLM(
            model="vertex_ai/gemini-2.5-pro-exp-03-25",  # Upgrade to more capable model
            api_key=api_key,
            temperature=temperature,
            vertex_credentials=vertex_credentials_json
        )

def create_analyst_agent(api_key: str, tools: List[Any], llm_provider: str = "google") -> Agent:
    """Create an Analyst agent for analyzing requirements and codebase.
    
    Args:
        api_key (str): API key for the LLM provider.
        tools (List[Any]): List of tools for the agent to use.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        
    Returns:
        Agent: The configured Analyst agent.
    """
    model = get_llm_config(api_key, llm_provider, temperature=0.1)  # Lower temperature for analysis
    
    return Agent(
        role="Data Pipeline Analyst",
        goal="Analyze requirements to understand data flow needs and identify existing patterns in the codebase if not found leave the existing codebase context",
        backstory="""You are an expert data engineer specializing in Apache Airflow. 
        You have years of experience analyzing business requirements and translating them 
        into effective data pipeline architectures. You excel at understanding existing 
        codebases and identifying the right components to leverage for new pipelines.
        You're known for your meticulous attention to detail and ability to create comprehensive
        analyses that serve as a solid foundation for pipeline development.""",
        verbose=True,
        allow_delegation=False,
        llm=model,
        tools=tools,  # Added tools parameter
        memory=True,
        respect_context_window=True,
        max_iter=25,
        max_retry_limit=3
    )

def create_architect_agent(api_key: str, tools: List[Any], llm_provider: str = "google") -> Agent:
    """Create an Architect agent for designing DAG architecture.
    
    Args:
        api_key (str): API key for the LLM provider.
        tools (List[Any]): List of tools for the agent to use.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        
    Returns:
        Agent: The configured Architect agent.
    """
    model = get_llm_config(api_key, llm_provider, temperature=0.2)
    
    return Agent(
        role="DAG Architect",
        goal="Design an optimal Airflow DAG structure based on requirements",
        backstory="""You are a senior data pipeline architect with deep expertise in Apache Airflow. 
        You design elegant, efficient, and maintainable DAG structures that follow best practices. 
        You have a talent for creating modular, reusable components and ensuring proper error handling 
        and monitoring in your designs. Your DAG designs are known to be clear, scalable, and 
        resilient to failures, always incorporating appropriate retry mechanisms and 
        monitoring points.""",
        verbose=True,
        allow_delegation=True,
        llm=model,
        tools=tools,  # Added tools parameter
        memory=True,
        respect_context_window=True,
        max_iter=25,
        max_retry_limit=3
    )

def create_developer_agent(api_key: str, tools: List[Any], llm_provider: str = "google") -> Agent:
    """Create a Developer agent for implementing the DAG code.
    
    Args:
        api_key (str): API key for the LLM provider.
        tools (List[Any]): List of tools for the agent to use.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        
    Returns:
        Agent: The configured Developer agent.
    """
    model = get_llm_config(api_key, llm_provider, temperature=0.2)  # Lower temperature for more reliable code
    
    return Agent(
        role="Airflow DAG Developer",
        goal="Implement robust, well-documented DAG code based on the architecture design",
        backstory="""You are an expert Python developer specialized in building Apache Airflow DAGs. 
        You write clean, efficient, and well-documented code that adheres to best practices. 
        Your code is known for being reliable, maintainable, and easy to understand. You excel at 
        implementing complex data pipelines with proper error handling and logging. You always 
        ensure your code follows PEP 8 standards and includes comprehensive docstrings.""",
        verbose=True,
        allow_delegation=False,
        llm=model,
        tools=tools,  # Added tools parameter
        memory=True,
        respect_context_window=True,
        max_iter=30,
        max_retry_limit=3,
        allow_code_execution=False  # For safety, keeping this off by default
    )

def create_reviewer_agent(api_key: str, tools: List[Any], llm_provider: str = "google") -> Agent:
    """Create a Reviewer agent for testing and validating the DAG.
    
    Args:
        api_key (str): API key for the LLM provider.
        tools (List[Any]): List of tools for the agent to use.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        
    Returns:
        Agent: The configured Reviewer agent.
    """
    model = get_llm_config(api_key, llm_provider, temperature=0.1)  # Lower temperature for review
    
    return Agent(
        role="Data Pipeline Quality Assurance Engineer",
        goal="Thoroughly review and validate DAG code for quality, correctness, and best practices",
        backstory="""You are a meticulous QA engineer specializing in data pipelines and Apache Airflow. 
        Your keen eye for detail helps catch bugs, inefficiencies, and anti-patterns before they reach 
        production. You have extensive experience in reviewing complex data workflows and ensuring they 
        meet high standards of reliability, maintainability, and performance. You're known for providing
        actionable feedback that significantly improves code quality.""",
        verbose=True,
        allow_delegation=False,
        llm=model,
        tools=tools,  # Added tools parameter
        memory=True,
        respect_context_window=True,
        max_iter=20,
        max_retry_limit=3
    )

def create_analysis_task(analyst_agent: Agent, requirements: str) -> Task:
    """Create a task for analyzing requirements and codebase, injecting user requirements."""
    return Task(
        description=f"""
        Analyze the following user requirements for a new Airflow DAG:
        ---
        {requirements}
        ---
        First, use the CodeSearchTool to search the codebase for existing patterns, utilities, and components that can be leveraged.
        Use the FileReadTool to examine specific files mentioned in the requirements or found through search.

        Your analysis must include:
        1. Key requirements and constraints extracted from the requirements document
        2. Relevant existing code components that can be reused (with file paths and snippets)
        3. Data sources and sinks identified in the requirements
        4. Required transformations with clear specification of input/output formats
        5. Potential challenges or considerations for implementation
        6. Airflow-specific considerations (scheduling, connections, variables needed)

        Be extremely thorough in your analysis, as it will form the foundation for the entire DAG development process.
        Search for similar DAGs in the codebase to understand the organization's patterns and conventions.

        Format your output as a structured report with clear sections and code examples where relevant.
        """,
        agent=analyst_agent,
        expected_output="A comprehensive analysis report of the requirements and codebase findings"
    )

def create_architecture_task(architect_agent: Agent, analysis_task: Task) -> Task:
    """Create a task for designing the DAG architecture.
    
    Args:
        architect_agent (Agent): The agent assigned to this task.
        analysis_task (Task): The prerequisite analysis task.
        
    Returns:
        Task: The configured architecture task.
    """
    return Task(
        description="""Based on the analysis report, design the architecture for the Airflow DAG.
        Use the CodeSearchTool to identify existing architectural patterns in the codebase that can be followed.
        
        Create a detailed design document that includes:
        1. DAG structure with a clear visual representation of task dependencies (ASCII diagram)
        2. Operators to be used for each task, with justification for each choice
        3. Configuration parameters and their default values
        4. Error handling and retry strategies for each task
        5. Resource considerations (memory, CPU) for each task
        6. Scheduling recommendations based on requirements
        7. Idempotency considerations to ensure the DAG can safely be rerun
        8. Monitoring and alerting recommendations
        
        Your design must follow Airflow best practices and be optimized for reliability and maintainability.
        Reference specific patterns from the codebase where appropriate.
        
        For each component in your design, explain:
        - Why this approach was chosen
        - How it addresses the requirements
        - How it integrates with existing codebase patterns
        
        Your output should be suitable for review by both technical and non-technical stakeholders.
        """,
        agent=architect_agent,
        context=[analysis_task],
        expected_output="A detailed DAG architecture design document with clear task dependencies and operator selections."
    )

def create_implementation_task(developer_agent: Agent, architecture_task: Task) -> Task:
    """Create a task for implementing the DAG code.
    
    Args:
        developer_agent (Agent): The agent assigned to this task.
        architecture_task (Task): The prerequisite architecture task.
        
    Returns:
        Task: The configured implementation task.
    """
    return Task(
        description="""Implement the Airflow DAG based on the provided architecture design.
        Use the CodeSearchTool to find examples and patterns in the existing codebase.
        
        Write production-ready Python code that:
        1. Follows Airflow best practices and design patterns
        2. Includes proper docstrings following Google style and inline comments for complex logic
        3. Implements all required operators and tasks exactly as specified in the architecture
        4. Sets up appropriate dependencies between tasks using the recommended methods
        5. Implements proper error handling, retries, and failure callbacks
        6. Uses appropriate scheduling parameters
        7. Includes appropriate tagging for monitoring and organization
        8. Sets sensible default values for all parameters
        
        Ensure your code handles all edge cases identified in the analysis and architecture phases.
        Include thorough logging at appropriate levels to facilitate troubleshooting.
        
        Your code should be complete and ready for production deployment with no TODOs or placeholders.
        Format your implementation as a complete DAG file with all necessary imports and components.
        """,
        agent=developer_agent,
        context=[architecture_task],
        expected_output="Complete, production-ready Python code for the Airflow DAG with all required components and error handling."
    )

def create_review_task(reviewer_agent: Agent, implementation_task: Task) -> Task:
    """Create a task for reviewing and validating the DAG code.
    
    Args:
        reviewer_agent (Agent): The agent assigned to this task.
        implementation_task (Task): The prerequisite implementation task.
        
    Returns:
        Task: The configured review task.
    """
    return Task(
        description="""Review the implemented DAG code for quality, correctness, and adherence to best practices.
        Use the CodeSearchTool to compare against existing DAGs in the codebase for consistency.
        
        Provide a detailed review report that includes:
        1. Code quality assessment (clarity, style, documentation)
        2. Identification of potential bugs, edge cases, or race conditions
        3. Performance analysis and optimization suggestions
        4. Error handling and recovery assessment
        5. Validation that the implementation meets all requirements from the analysis
        6. Comparison with codebase conventions and patterns
        7. Security considerations and potential vulnerabilities
        8. Testing recommendations
        
        For each issue identified, provide:
        - Specific location in the code
        - Explanation of the issue
        - Recommended fix with code example
        - Priority level (critical, important, minor)
        
        If there are critical issues, provide specific code snippets that can be used to fix them.
        Include a final assessment: "PASS", "PASS WITH MINOR ISSUES", or "NEEDS REVISION"
        """,
        agent=reviewer_agent,
        context=[implementation_task],
        expected_output="A comprehensive review report with validation results, specific issues, and actionable improvement suggestions."
    )

def create_dag_generator_crew(
    api_key: str, 
    codebase_path: str, 
    airflow_container_name: str = "airflow-webserver-1",
    llm_provider: str = "google",
    airflow_compose_file: Optional[str] = None,
    use_embeddings: bool = True,
    embedding_provider: str = "local",
    requirements: str = ""
) -> Tuple[Crew, DockerManager, CodebaseContext]:
    """Creates and configures the CrewAI crew for DAG generation, now with user requirements."""
    logger.info(f"Creating DAG Generator Crew with LLM provider: {llm_provider}")
    
    # Initialize Docker manager
    docker_manager = DockerManager(
        container_name=airflow_container_name,
        compose_file=airflow_compose_file
    )
    
    # Initialize codebase context
    codebase_context = CodebaseContext(
        codebase_path=codebase_path,
        use_embeddings=use_embeddings,
        embedding_provider=embedding_provider,
        api_key=api_key
    )
    
    # Create tools
    code_search_tool = CodeSearchTool(codebase_context)
    file_read_tool = FileReadTool(codebase_context)
    tools = [code_search_tool, file_read_tool]
    
    # Create agents with tools
    analyst = create_analyst_agent(api_key, tools, llm_provider)
    architect = create_architect_agent(api_key, tools, llm_provider)
    developer = create_developer_agent(api_key, tools, llm_provider)
    reviewer = create_reviewer_agent(api_key, tools, llm_provider)
    
    # Create tasks, passing requirements to analysis_task
    analysis_task = create_analysis_task(analyst, requirements)
    architecture_task = create_architecture_task(architect, analysis_task)
    implementation_task = create_implementation_task(developer, architecture_task)
    review_task = create_review_task(reviewer, implementation_task)
    
    # Create crew with Sequential process
    crew = Crew(
        agents=[analyst, architect, developer, reviewer],
        tasks=[analysis_task, architecture_task, implementation_task, review_task],
        verbose=True,
        memory=True,
        process=Process.sequential
    )
    
    return crew, docker_manager, codebase_context