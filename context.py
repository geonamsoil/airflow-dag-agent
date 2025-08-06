# Handles loading and providing access to codebase context.import warnings
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import logging
import re
import ast
from typing import Dict, Optional, List, Any, Tuple, Union

# For embeddings
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

class CodebaseContext:
    """Handles loading and providing access to codebase context.
    
    This class is responsible for scanning a codebase directory structure,
    loading Python files, and providing a searchable context for AI agents
    to understand and work with the existing codebase.
    
    Attributes:
        codebase_path (str): Path to the root directory of the codebase.
        _context (Optional[Dict[str, Dict[str, str]]]): Internal storage for code context.
        _embeddings (Optional[Dict[str, np.ndarray]]): Embeddings for code files.
    """

    def __init__(self, codebase_path: str, util_dirs: Optional[List[str]] = None, 
                 use_embeddings: bool = True, embedding_provider: str = "google",
                 api_key: Optional[str] = None):
        """Initialize the CodebaseContext with a path to the codebase.
        
        Args:
            codebase_path (str): Path to the root directory of the codebase.
            util_dirs (Optional[List[str]], optional): Utility directories to include. 
                Defaults to None.
            use_embeddings (bool, optional): Whether to use embeddings for search. 
                Defaults to True.
            embedding_provider (str, optional): Provider for embeddings ('openai', 'google', or 'local'). 
                Defaults to "google".
            api_key (Optional[str], optional): API key for the embedding provider. 
                Defaults to None.
        """
        self.codebase_path = os.path.abspath(codebase_path)
        self.util_dirs = util_dirs or []
        self.use_embeddings = use_embeddings
        self.embedding_provider = embedding_provider
        self.api_key = api_key
        
        # Initialize context and embeddings
        self._context = None
        self._embeddings = None
        self._embedding_model = None
        
        # Initialize embedding model if embeddings are enabled
        if self.use_embeddings:
            self._init_embedding_model()
        
        # Load the codebase context
        self.load_context()
        
    def _init_embedding_model(self):
        """Initialize the embedding model based on the provider."""
        if not self.use_embeddings:
            return
            
        if self.embedding_provider.lower() == "openai":
            if not self.api_key:
                logger.warning("OpenAI API key not provided, embeddings will not be available")
                self.use_embeddings = False
                return
                
            self._embedding_model = OpenAIEmbeddings(
                openai_api_key=self.api_key
            )
            logger.info("Initialized OpenAI embeddings model")
            
        elif self.embedding_provider.lower() == "google":
            if not self.api_key:
                logger.warning("Google API key not provided, embeddings will not be available")
                self.use_embeddings = False
                return
                
            self._embedding_model = GoogleGenerativeAIEmbeddings(
                google_api_key=self.api_key,
                model="embedding-001"
            )
            logger.info("Initialized Google embeddings model")
            
        elif self.embedding_provider.lower() == "local":
            # Use HuggingFace embeddings with a local model that doesn't require an API key
            self._embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Initialized local HuggingFace embeddings model")
            
        else:
            logger.warning(f"Unknown embedding provider: {self.embedding_provider}, embeddings will not be available")
            self.use_embeddings = False

    def load_context(self) -> None:
        """Load or reload code structure from the codebase path.
        
        Scans the codebase directory and all subdirectories for Python files,
        excluding environment and cache directories, reads their content, and stores them in the internal context dictionary.
        If embeddings are enabled, also computes embeddings for each file.
        """
        logger.info(f"Loading code context from: {self.codebase_path}")
        context: Dict[str, Dict[str, Any]] = {}
        loaded_files = set()

        # Directories to ignore
        ignore_dirs = {'.venv', 'venv', 'env', '__pycache__'}

        # Recursively scan all subdirectories for Python files (excluding __init__.py and ignored dirs)
        py_files = []
        for root, dirs, files in os.walk(self.codebase_path):
            # Remove ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    py_files.append(os.path.join(root, file))
        for py_file in py_files:
            if py_file not in loaded_files:
                self._process_file(py_file, context, loaded_files)

        self._context = context
        logger.info(f"Loaded {len(context)} Python files into context")
        
        # Compute embeddings if enabled
        if self.use_embeddings and self._embedding_model:
            self._compute_embeddings()
            
    def _process_file(self, file_path: str, context: Dict[str, Dict[str, Any]], loaded_files: set):
        """Process a single file and add it to the context.
        
        Args:
            file_path (str): Path to the file to process.
            context (Dict[str, Dict[str, Any]]): Context dictionary to update.
            loaded_files (set): Set of already loaded files.
        """
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                file_content = f.read()
                # Store both absolute and relative paths for easier reference
                rel_path = os.path.relpath(file_path, self.codebase_path)
                
                # Extract classes and functions for better understanding
                classes, functions = self._extract_code_components(file_content)
                
                context[file_path] = {
                    "content": file_content,
                    "relative_path": rel_path,
                    "file_size": len(file_content),
                    "classes": classes,
                    "functions": functions
                }
            loaded_files.add(file_path)
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {str(e)}")
            
    def _extract_code_components(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract classes and functions from Python code using AST for accuracy.
        Args:
            content (str): Python code content.
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Extracted classes and functions.
        """
        classes = []
        functions = []
        try:
            tree = ast.parse(content)
        except Exception:
            # fallback to regex if AST fails
            return self._extract_code_components_regex(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                decorators = [ast.unparse(d) if hasattr(ast, 'unparse') else getattr(d, 'id', None) for d in node.decorator_list]
                classes.append({
                    "name": node.name,
                    "docstring": docstring,
                    "lineno": node.lineno,
                    "decorators": decorators
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                decorators = [ast.unparse(d) if hasattr(ast, 'unparse') else getattr(d, 'id', None) for d in node.decorator_list]
                args = [a.arg for a in node.args.args]
                if hasattr(node.args, 'posonlyargs'):
                    args = [a.arg for a in node.args.posonlyargs] + args
                if node.args.vararg:
                    args.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")
                return_type = ast.unparse(node.returns) if getattr(node, 'returns', None) is not None and hasattr(ast, 'unparse') else None
                functions.append({
                    "name": node.name,
                    "params": args,
                    "return_type": return_type,
                    "docstring": docstring,
                    "lineno": node.lineno,
                    "decorators": decorators,
                    "async": isinstance(node, ast.AsyncFunctionDef)
                })
        return classes, functions

    def _extract_code_components_regex(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fallback method to extract classes and functions using regex."""
        classes = []
        functions = []
        class_pattern = r'class\s+([\w_]+)\s*(?:\([^)]*\))?\s*:'
        function_pattern = r'def\s+([\w_]+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_pos = match.start()
            docstring = self._extract_docstring(content, start_pos)
            classes.append({
                "name": class_name,
                "docstring": docstring
            })
        for match in re.finditer(function_pattern, content):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3) if match.group(3) else None
            start_pos = match.start()
            docstring = self._extract_docstring(content, start_pos)
            functions.append({
                "name": func_name,
                "params": params,
                "return_type": return_type,
                "docstring": docstring
            })
        return classes, functions

    def _extract_docstring(self, content: str, start_pos: int) -> Optional[str]:
        """Extract docstring after a class or function definition.
        
        Args:
            content (str): Python code content.
            start_pos (int): Start position of the class or function.
            
        Returns:
            Optional[str]: Extracted docstring or None if not found.
        """
        # Find the colon that ends the definition
        colon_pos = content.find(':', start_pos)
        if colon_pos == -1:
            return None
            
        # Find the next non-whitespace character
        pos = colon_pos + 1
        while pos < len(content) and content[pos].isspace():
            pos += 1
            
        if pos >= len(content):
            return None
            
        # Check for triple quotes (docstring)
        if content[pos:pos+3] in ['"""', "'''"]:
            quote_type = content[pos:pos+3]
            start = pos + 3
            end = content.find(quote_type, start)
            
            if end != -1:
                return content[start:end].strip()
                
        return None

    def _compute_embeddings(self):
        """Compute embeddings for all files in the context."""
        if not self._context or not self._embedding_model:
            return
            
        logger.info("Computing embeddings for codebase files")
        self._embeddings = {}
        
        for file_path, data in self._context.items():
            try:
                # Create a summary of the file for embedding
                summary = f"File: {data['relative_path']}\n"
                
                # Add classes
                if data.get("classes"):
                    summary += "\nClasses:\n"
                    for cls in data["classes"]:
                        summary += f"- {cls['name']}: {cls.get('docstring', '')}\n"
                        
                # Add functions
                if data.get("functions"):
                    summary += "\nFunctions:\n"
                    for func in data["functions"]:
                        summary += f"- {func['name']}({func.get('params', '')}): {func.get('docstring', '')}\n"
                        
                # Get embedding
                embedding = self._embedding_model.embed_query(summary)
                self._embeddings[file_path] = np.array(embedding)
                
            except Exception as e:
                logger.warning(f"Failed to compute embedding for {file_path}: {str(e)}")
                
        logger.info(f"Computed embeddings for {len(self._embeddings)} files")

    def get_context(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get the loaded code context.
        
        Returns:
            Optional[Dict[str, Dict[str, Any]]]: Dictionary mapping file paths to their content and metadata.
        """
        return self._context
    
    def search_context(self, query: str, max_results: int = 5, semantic: bool = True) -> Dict[str, Any]:
        """Search the code context for files matching the query.
        
        Args:
            query (str): The search query string.
            max_results (int): Maximum number of results to return.
            semantic (bool): Whether to use semantic search with embeddings.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and metadata.
        """
        if not self._context:
            return {"error": "Code context not loaded", "results": []}
            
        if semantic and self.use_embeddings and self._embedding_model and self._embeddings:
            return self._semantic_search(query, max_results)
        else:
            return self._keyword_search(query, max_results)
            
    def _keyword_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform keyword-based search in the code context.
        
        Args:
            query (str): The search query string.
            max_results (int): Maximum number of results to return.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and metadata.
        """
        results = []
        for file_path, data in self._context.items():
            if query.lower() in data["content"].lower():
                results.append({
                    "file_path": file_path,
                    "relative_path": data["relative_path"],
                    "file_size": data["file_size"],
                    "relevance": "high" if query.lower() in data["relative_path"].lower() else "medium"
                })
                
        # Sort by relevance and limit results
        results.sort(key=lambda x: 0 if x["relevance"] == "high" else 1)
        limited_results = results[:max_results]
        
        return {
            "query": query,
            "search_type": "keyword",
            "total_matches": len(results),
            "returned_matches": len(limited_results),
            "results": limited_results
        }
        
    def _semantic_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform semantic search using embeddings.
        
        Args:
            query (str): The search query string.
            max_results (int): Maximum number of results to return.
            
        Returns:
            Dict[str, Any]: Dictionary containing search results and metadata.
        """
        try:
            # Get query embedding
            query_embedding = np.array(self._embedding_model.embed_query(query))
            
            # Calculate similarity with all file embeddings
            similarities = {}
            for file_path, embedding in self._embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities[file_path] = similarity
                
            # Sort by similarity (descending)
            sorted_files = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Prepare results
            results = []
            for file_path, similarity in sorted_files[:max_results]:
                data = self._context[file_path]
                results.append({
                    "file_path": file_path,
                    "relative_path": data["relative_path"],
                    "file_size": data["file_size"],
                    "similarity": float(similarity),  # Convert numpy float to Python float
                    "relevance": "high" if similarity > 0.8 else "medium" if similarity > 0.5 else "low"
                })
                
            return {
                "query": query,
                "search_type": "semantic",
                "total_matches": len(sorted_files),
                "returned_matches": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            # Fall back to keyword search
            logger.info("Falling back to keyword search")
            return self._keyword_search(query, max_results)
            
    def get_code_components(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all code components (classes and functions) from the codebase.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with classes and functions.
        """
        if not self._context:
            return {"classes": [], "functions": []}
            
        all_classes = []
        all_functions = []
        
        for file_path, data in self._context.items():
            rel_path = data["relative_path"]
            
            for cls in data.get("classes", []):
                all_classes.append({
                    **cls,
                    "file": rel_path
                })
                
            for func in data.get("functions", []):
                all_functions.append({
                    **func,
                    "file": rel_path
                })
                
        return {
            "classes": all_classes,
            "functions": all_functions
        }
        
    def search_code(self, query: str, max_results: int = 5) -> Dict[str, str]:
        """Search the codebase for relevant code snippets.
        
        This method is used by the CodeSearchTool to find relevant code in the codebase.
        
        Args:
            query (str): The search query string.
            max_results (int): Maximum number of results to return.
            
        Returns:
            Dict[str, str]: Dictionary mapping file paths to code snippets.
        """
        search_results = self.search_context(query, max_results)
        
        # Extract the relevant code snippets from the search results
        code_snippets = {}
        for result in search_results.get("results", []):
            file_path = result.get("file_path")
            if file_path and file_path in self._context:
                # Use the relative path as the key for better readability
                rel_path = self._context[file_path]["relative_path"]
                code_snippets[rel_path] = self._context[file_path]["content"]
                
        return code_snippets
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read the contents of a file.
        
        This method is used by the FileReadTool to read the contents of a file.
        
        Args:
            file_path (str): The path to the file to read.
            
        Returns:
            Optional[str]: The file contents, or None if the file doesn't exist.
        """
        # Check if the file is in the context
        if file_path in self._context:
            return self._context[file_path]["content"]
            
        # If not, try to find it by relative path
        for path, data in self._context.items():
            if data["relative_path"] == file_path:
                return data["content"]
                
        # If still not found, try to read it directly
        try:
            full_path = os.path.join(self.codebase_path, file_path)
            if not os.path.exists(full_path):
                full_path = file_path  # Try using the path as is
                
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            
        return None
