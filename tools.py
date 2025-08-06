# Tools for searching and interacting with the codebase.import warnings
import warnings
warnings.filterwarnings('ignore')
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from context import CodebaseContext

class CodeSearchToolInput(BaseModel):
    """Input model for CodeSearchTool."""
    query: str = Field(description="The natural language query describing the functionality needed, e.g., 'find email sending utility', 'class for processing CSV data'.")
    max_results: int = Field(default=5, description="Maximum number of results to return.")

class CodeSearchTool(BaseTool):
    """Tool for searching the codebase for relevant Python code snippets.
    
    This tool allows agents to search through the codebase to find relevant code
    based on natural language queries. It returns both file paths and code snippets
    to help agents understand the existing codebase structure and functionality.
    """
    name: str = "codebase_search"
    description: str = (
        "Searches the existing Python codebase for files or code snippets "
        "relevant to a given functional query. Returns a dictionary where keys are "
        "relative file paths and values are code snippets from those files."
    )
    args_schema: type = CodeSearchToolInput
    
    def __init__(self, codebase_context: CodebaseContext):
        """Initialize the CodeSearchTool with a CodebaseContext.
        
        Args:
            codebase_context (CodebaseContext): The context containing loaded code files.
        """
        super().__init__()
        self._codebase_context = codebase_context

    def _run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the codebase for relevant code snippets.
        
        Args:
            query (str): The natural language query describing the functionality needed.
            max_results (int, optional): Maximum number of results to return. Defaults to 5.
            
        Returns:
            Dict[str, Any]: Dictionary with search results and snippets.
        """
        try:
            # Search the codebase for relevant code
            results = self._codebase_context.search_code(query, max_results=max_results)
            
            if not results:
                return {
                    "success": False,
                    "error": "No relevant code found for the query.",
                    "results": {}
                }
            
            # Format the results
            formatted_results = {}
            for file_path, snippets in results.items():
                # Join snippets with line breaks if there are multiple
                if isinstance(snippets, list):
                    formatted_results[file_path] = "\n\n".join(snippets)
                else:
                    formatted_results[file_path] = snippets
            
            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }

    async def _arun(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the codebase for relevant code snippets asynchronously.
        
        Args:
            query (str): The natural language query describing the functionality needed.
            max_results (int, optional): Maximum number of results to return. Defaults to 5.
            
        Returns:
            Dict[str, Any]: Dictionary with search results and snippets.
        """
        # For now, just call the synchronous version
        return self._run(query, max_results)


class FileReadToolInput(BaseModel):
    """Input model for FileReadTool."""
    file_path: str = Field(description="The path to the file to read.")

class FileReadTool(BaseTool):
    """Tool for reading the contents of a file.
    
    This tool allows agents to read the full contents of a specific file,
    which is useful for understanding code in detail after finding relevant
    files using the CodeSearchTool.
    """
    name: str = "file_read"
    description: str = "Reads the contents of a specified file."
    args_schema: type = FileReadToolInput
    
    def __init__(self, codebase_context: CodebaseContext):
        """Initialize the FileReadTool with a CodebaseContext.
        
        Args:
            codebase_context (CodebaseContext): The context containing loaded code files.
        """
        super().__init__()
        self._codebase_context = codebase_context

    def _run(self, file_path: str) -> Dict[str, Any]:
        """Read the contents of a file.
        
        Args:
            file_path (str): The path to the file to read.
            
        Returns:
            Dict[str, Any]: The file contents and metadata.
        """
        try:
            # Read the file contents
            content = self._codebase_context.read_file(file_path)
            
            if content is None:
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "content": ""
                }
            
            # Get file extension for language detection
            extension = file_path.split('.')[-1] if '.' in file_path else ''
            language = self._get_language_from_extension(extension)
            
            return {
                "success": True,
                "file_path": file_path,
                "content": content,
                "language": language,
                "size": len(content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }

    async def _arun(self, file_path: str) -> Dict[str, Any]:
        """Read the contents of a file asynchronously.
        
        Args:
            file_path (str): The path to the file to read.
            
        Returns:
            Dict[str, Any]: The file contents and metadata.
        """
        # For now, just call the synchronous version
        return self._run(file_path)
    
    def _get_language_from_extension(self, extension: str) -> str:
        """Get the programming language based on file extension.
        
        Args:
            extension (str): The file extension.
            
        Returns:
            str: The programming language name.
        """
        extension_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'c': 'c',
            'cpp': 'cpp',
            'h': 'cpp',
            'hpp': 'cpp',
            'cs': 'csharp',
            'go': 'go',
            'rb': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kt': 'kotlin',
            'rs': 'rust',
            'scala': 'scala',
            'sh': 'bash',
            'bat': 'batch',
            'ps1': 'powershell',
            'sql': 'sql',
            'r': 'r',
            'md': 'markdown',
            'json': 'json',
            'xml': 'xml',
            'yaml': 'yaml',
            'yml': 'yaml',
            'html': 'html',
            'css': 'css',
            'scss': 'scss',
            'less': 'less',
            'txt': 'text'
        }
        
        return extension_map.get(extension.lower(), 'text')
