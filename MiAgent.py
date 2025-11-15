#!/usr/bin/env python3
"""
MiAgent - A CLI agent that uses LLM API calls to write code and create projects.
Similar to Cursor/Windsurf functionality.
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import openai
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProvider:
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate(self, prompt: str, system_prompt: str = "", conversation_history: List[Dict] = None) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = "", conversation_history: List[Dict] = None) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key)
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = "", conversation_history: List[Dict] = None) -> str:
        messages = []
        
        if conversation_history:
            # Convert OpenAI format to Anthropic format
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt if system_prompt else "",
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"


class GroqProvider(LLMProvider):
    """Groq API provider"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        super().__init__(api_key)
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("groq package is required. Install it with: pip install groq")
    
    def generate(self, prompt: str, system_prompt: str = "", conversation_history: List[Dict] = None) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, api_key: str = None, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        # Ollama doesn't require an API key, but we keep it for consistency
        super().__init__(api_key or "")
        self.model = model
        self.base_url = base_url
        try:
            from ollama import Client
            self.client = Client(host=base_url)
        except ImportError:
            raise ImportError("ollama package is required. Install it with: pip install ollama")
    
    def generate(self, prompt: str, system_prompt: str = "", conversation_history: List[Dict] = None) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}. Make sure Ollama is running and the model '{self.model}' is available."


class ProjectManager:
    """Manages project structure and file operations"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_file(self, file_path: str, content: str) -> bool:
        """Create a file with the given content"""
        try:
            full_path = self.base_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error creating file {file_path}: {e}")
            return False
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read a file's content"""
        try:
            full_path = self.base_path / file_path
            if full_path.exists():
                return full_path.read_text(encoding='utf-8')
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def list_files(self, directory: str = ".") -> List[str]:
        """List all files in a directory"""
        try:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                return []
            files = []
            for item in dir_path.rglob("*"):
                if item.is_file():
                    files.append(str(item.relative_to(self.base_path)))
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def create_directory(self, dir_path: str) -> bool:
        """Create a directory"""
        try:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
            return False
    
    def grep(self, pattern: str, case_sensitive: bool = False, file_extensions: List[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for a pattern in all files (grep-like functionality)"""
        results = []
        files_searched = 0
        
        try:
            # Compile regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            
            # Get all files
            all_files = self.list_files()
            
            # Filter by extension if specified
            if file_extensions:
                all_files = [f for f in all_files if any(f.endswith(ext) for ext in file_extensions)]
            
            for file_path in all_files:
                if files_searched >= max_results:
                    break
                    
                try:
                    content = self.read_file(file_path)
                    if content is None:
                        continue
                    
                    lines = content.split('\n')
                    matches = []
                    
                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            matches.append({
                                'line_number': line_num,
                                'line_content': line.strip()
                            })
                    
                    if matches:
                        results.append({
                            'file': file_path,
                            'matches': matches,
                            'match_count': len(matches)
                        })
                        files_searched += 1
                        
                except Exception as e:
                    # Skip files that can't be read (binary files, etc.)
                    continue
            
            return results
            
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return []
        except Exception as e:
            print(f"Error during grep: {e}")
            return []
    
    def get_project_structure(self) -> Dict[str, Any]:
        """Get comprehensive project structure information"""
        structure = {
            'base_path': str(self.base_path),
            'files': [],
            'directories': [],
            'file_types': {},
            'total_files': 0,
            'total_size': 0
        }
        
        try:
            # Get all files and directories
            for item in self.base_path.rglob("*"):
                if item.is_file():
                    rel_path = str(item.relative_to(self.base_path))
                    file_ext = item.suffix.lower()
                    
                    structure['files'].append({
                        'path': rel_path,
                        'size': item.stat().st_size,
                        'extension': file_ext
                    })
                    
                    # Count file types
                    if file_ext:
                        structure['file_types'][file_ext] = structure['file_types'].get(file_ext, 0) + 1
                    else:
                        structure['file_types']['no_extension'] = structure['file_types'].get('no_extension', 0) + 1
                    
                    structure['total_files'] += 1
                    structure['total_size'] += item.stat().st_size
                    
                elif item.is_dir():
                    rel_path = str(item.relative_to(self.base_path))
                    structure['directories'].append(rel_path)
            
            return structure
            
        except Exception as e:
            print(f"Error getting project structure: {e}")
            return structure
    
    def get_file_contents_summary(self, max_files: int = 50, max_lines_per_file: int = 100) -> str:
        """Get a summary of file contents for analysis"""
        files = self.list_files()
        summary_parts = []
        
        # Filter out common non-code files
        exclude_patterns = ['.git', '__pycache__', '.pyc', '.png', '.jpg', '.jpeg', '.gif', 
                           '.pdf', '.zip', '.tar', '.gz', '.node_modules']
        
        code_files = [f for f in files if not any(exc in f for exc in exclude_patterns)]
        code_files = code_files[:max_files]
        
        for file_path in code_files:
            try:
                content = self.read_file(file_path)
                if content:
                    lines = content.split('\n')
                    # Take first and last few lines
                    preview_lines = lines[:max_lines_per_file]
                    if len(lines) > max_lines_per_file:
                        preview_lines.append(f"... ({len(lines) - max_lines_per_file} more lines)")
                    
                    summary_parts.append(f"\n{'='*80}")
                    summary_parts.append(f"File: {file_path}")
                    summary_parts.append(f"Lines: {len(lines)}")
                    summary_parts.append(f"{'='*80}")
                    summary_parts.append('\n'.join(preview_lines))
            except Exception:
                continue
        
        return '\n'.join(summary_parts)
    
    def analyze_code_patterns(self, max_files: int = 20) -> Dict[str, Any]:
        """Analyze code patterns from existing files"""
        patterns = {
            'language': None,
            'naming_conventions': {},
            'import_style': {},
            'docstring_style': None,
            'indentation': None,
            'line_length': None,
            'function_style': {},
            'class_style': {},
            'error_handling': {},
            'code_structure': {},
            'example_code': []
        }
        
        files = self.list_files()
        
        # Filter code files
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php']
        code_files = [f for f in files if any(f.endswith(ext) for ext in code_extensions)]
        code_files = code_files[:max_files]
        
        if not code_files:
            return patterns
        
        # Determine primary language
        ext_counts = {}
        for f in code_files:
            ext = Path(f).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        if ext_counts:
            primary_ext = max(ext_counts, key=ext_counts.get)
            patterns['language'] = primary_ext
        
        # Analyze sample files
        sample_code = []
        for file_path in code_files[:10]:  # Analyze first 10 files in detail
            try:
                content = self.read_file(file_path)
                if not content:
                    continue
                
                lines = content.split('\n')
                sample_code.append({
                    'file': file_path,
                    'content': content[:2000]  # First 2000 chars
                })
                
                # Analyze patterns
                self._extract_patterns_from_code(content, file_path, patterns)
                
            except Exception:
                continue
        
        patterns['example_code'] = sample_code
        return patterns
    
    def _extract_patterns_from_code(self, content: str, file_path: str, patterns: Dict[str, Any]) -> None:
        """Extract code patterns from a single file"""
        lines = content.split('\n')
        
        # Detect indentation (tabs vs spaces)
        for line in lines[:50]:
            if line.strip():
                if line.startswith('\t'):
                    patterns['indentation'] = 'tabs'
                    break
                elif line.startswith(' '):
                    # Count spaces
                    indent = len(line) - len(line.lstrip())
                    if indent > 0:
                        patterns['indentation'] = f'{indent} spaces'
                        break
        
        # Analyze imports
        import_patterns = []
        for line in lines[:100]:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_patterns.append(stripped)
        
        if import_patterns:
            patterns['import_style']['examples'] = import_patterns[:5]
        
        # Analyze function definitions
        func_patterns = []
        for line in lines:
            if re.match(r'^\s*(def|function|fn|func)\s+\w+', line):
                func_patterns.append(line.strip())
        
        if func_patterns:
            patterns['function_style']['examples'] = func_patterns[:5]
        
        # Analyze class definitions
        class_patterns = []
        for line in lines:
            if re.match(r'^\s*class\s+\w+', line):
                class_patterns.append(line.strip())
        
        if class_patterns:
            patterns['class_style']['examples'] = class_patterns[:5]
        
        # Analyze docstrings
        docstring_patterns = []
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                # Get docstring block
                doc_start = i
                if i + 1 < len(lines):
                    docstring_patterns.append('\n'.join(lines[doc_start:doc_start+5]))
        
        if docstring_patterns:
            patterns['docstring_style'] = 'triple_quotes'
        
        # Analyze naming conventions
        # Check variable naming (snake_case vs camelCase)
        var_patterns = {'snake_case': 0, 'camelCase': 0, 'PascalCase': 0}
        for line in lines[:200]:
            # Find variable assignments
            matches = re.findall(r'\b([a-z_][a-z0-9_]*|[A-Z][a-zA-Z0-9]*)\s*=', line)
            for match in matches:
                if '_' in match and match.islower():
                    var_patterns['snake_case'] += 1
                elif match[0].isupper():
                    var_patterns['PascalCase'] += 1
                elif match[0].islower() and not '_' in match:
                    var_patterns['camelCase'] += 1
        
        if any(var_patterns.values()):
            patterns['naming_conventions']['variables'] = max(var_patterns, key=var_patterns.get)
        
        # Analyze error handling
        error_patterns = []
        for line in lines:
            if 'try:' in line or 'except' in line or 'catch' in line or 'error' in line.lower():
                error_patterns.append(line.strip())
        
        if error_patterns:
            patterns['error_handling']['examples'] = error_patterns[:5]


class MiAgent:
    """Main CLI agent class"""
    
    def __init__(self, provider: str = "openai", api_key: str = None, model: str = None, base_path: str = ".", ollama_base_url: str = None):
        self.provider_name = provider.lower()
        
        # Get API key from parameter, environment variable, or .env file
        if not api_key:
            env_key_name = f"{self.provider_name.upper()}_API_KEY"
            self.api_key = os.getenv(env_key_name)
        else:
            self.api_key = api_key
        
        # Initialize provider
        if self.provider_name == "openai":
            if not self.api_key:
                raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable or pass --api-key")
            self.llm = OpenAIProvider(self.api_key, model or "gpt-4")
        elif self.provider_name == "anthropic":
            if not self.api_key:
                raise ValueError("API key not provided. Set ANTHROPIC_API_KEY environment variable or pass --api-key")
            self.llm = AnthropicProvider(self.api_key, model or "claude-3-5-sonnet-20241022")
        elif self.provider_name == "groq":
            if not self.api_key:
                raise ValueError("API key not provided. Set GROQ_API_KEY environment variable or pass --api-key")
            self.llm = GroqProvider(self.api_key, model or "llama-3.1-70b-versatile")
        elif self.provider_name == "ollama":
            # Ollama doesn't require API key, but we allow it for consistency
            ollama_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm = OllamaProvider(api_key=self.api_key, model=model or "llama3.2", base_url=ollama_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', 'groq', or 'ollama'")
        
        self.project_manager = ProjectManager(base_path)
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()
        self.code_patterns = None  # Will store analyzed code patterns
        self.follow_patterns = True  # Default to following patterns
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for code generation"""
        return """You are an expert software development assistant. Your task is to help users create projects, write code, and implement features.

When generating code:
1. Write clean, well-documented, production-ready code
2. Follow best practices and conventions for the language/framework
3. Include necessary imports and dependencies
4. Add comments where appropriate
5. Consider error handling and edge cases
6. When creating projects, include proper project structure, README files, and configuration files

When the user asks you to create files or projects, respond in a structured format:
- For single file: Just provide the code
- For multiple files: Use JSON format with file paths as keys and content as values
- For project structure: Provide a comprehensive project layout with all necessary files

Always be thorough and consider the full context of what the user wants to build."""
    
    def _parse_code_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response to extract file paths and content"""
        files = {}
        
        # Try to parse as JSON first (for multiple files)
        try:
            # Look for JSON code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
        except:
            pass
        
        # Try to parse as regular JSON
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass
        
        # If not JSON, treat as single file content
        # Try to extract code blocks
        if "```" in response:
            # Extract code from markdown code blocks
            lines = response.split("\n")
            current_code = []
            in_code_block = False
            current_lang = ""
            
            for line in lines:
                if line.strip().startswith("```"):
                    if in_code_block:
                        # End of code block
                        if current_code:
                            code_content = "\n".join(current_code)
                            # Try to infer filename from language or context
                            filename = f"code.{current_lang}" if current_lang else "code.txt"
                            files[filename] = code_content
                        current_code = []
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                        current_lang = line.strip()[3:].strip()
                elif in_code_block:
                    current_code.append(line)
            
            # Handle case where code block doesn't close
            if current_code:
                code_content = "\n".join(current_code)
                filename = f"code.{current_lang}" if current_lang else "code.txt"
                files[filename] = code_content
        else:
            # No code blocks, treat entire response as code
            files["output.txt"] = response
        
        return files
    
    def analyze_code_patterns(self) -> Dict[str, Any]:
        """Analyze code patterns from the project"""
        print("\n🔍 Analyzing code patterns in project...\n")
        patterns = self.project_manager.analyze_code_patterns()
        self.code_patterns = patterns
        return patterns
    
    def get_code_patterns_summary(self) -> str:
        """Get a formatted summary of code patterns"""
        if not self.code_patterns:
            self.code_patterns = self.project_manager.analyze_code_patterns()
        
        patterns = self.code_patterns
        summary = []
        
        summary.append("📋 CODE PATTERNS SUMMARY")
        summary.append("=" * 80)
        
        if patterns['language']:
            summary.append(f"\nPrimary Language: {patterns['language']}")
        
        if patterns['indentation']:
            summary.append(f"Indentation: {patterns['indentation']}")
        
        if patterns['naming_conventions'].get('variables'):
            summary.append(f"Variable Naming: {patterns['naming_conventions']['variables']}")
        
        if patterns['docstring_style']:
            summary.append(f"Docstring Style: {patterns['docstring_style']}")
        
        if patterns['import_style'].get('examples'):
            summary.append(f"\nImport Style Examples:")
            for imp in patterns['import_style']['examples'][:3]:
                summary.append(f"  {imp}")
        
        if patterns['function_style'].get('examples'):
            summary.append(f"\nFunction Style Examples:")
            for func in patterns['function_style']['examples'][:3]:
                summary.append(f"  {func}")
        
        if patterns['class_style'].get('examples'):
            summary.append(f"\nClass Style Examples:")
            for cls in patterns['class_style']['examples'][:3]:
                summary.append(f"  {cls}")
        
        if patterns['example_code']:
            summary.append(f"\nExample Code Files Analyzed:")
            for ex in patterns['example_code'][:5]:
                summary.append(f"  - {ex['file']}")
        
        summary.append("=" * 80)
        
        return '\n'.join(summary)
    
    def _build_pattern_guidance(self) -> str:
        """Build guidance text from code patterns for LLM"""
        if not self.code_patterns:
            self.code_patterns = self.project_manager.analyze_code_patterns()
        
        patterns = self.code_patterns
        guidance = []
        
        guidance.append("IMPORTANT: Follow the existing code patterns in this project:")
        guidance.append("")
        
        if patterns['language']:
            guidance.append(f"- Primary language: {patterns['language']}")
        
        if patterns['indentation']:
            guidance.append(f"- Use {patterns['indentation']} for indentation")
        
        if patterns['naming_conventions'].get('variables'):
            guidance.append(f"- Use {patterns['naming_conventions']['variables']} for variable naming")
        
        if patterns['docstring_style']:
            guidance.append(f"- Use {patterns['docstring_style']} for documentation")
        
        if patterns['import_style'].get('examples'):
            guidance.append(f"- Import style examples:")
            for imp in patterns['import_style']['examples'][:3]:
                guidance.append(f"  {imp}")
        
        if patterns['function_style'].get('examples'):
            guidance.append(f"- Function definition style:")
            for func in patterns['function_style']['examples'][:2]:
                guidance.append(f"  {func}")
        
        if patterns['class_style'].get('examples'):
            guidance.append(f"- Class definition style:")
            for cls in patterns['class_style']['examples'][:2]:
                guidance.append(f"  {cls}")
        
        if patterns['example_code']:
            guidance.append(f"\nExample code from project:")
            for ex in patterns['example_code'][:2]:
                guidance.append(f"\nFile: {ex['file']}")
                guidance.append(ex['content'][:500])  # First 500 chars
                guidance.append("")
        
        guidance.append("\nCRITICAL: Match the exact style, naming conventions, and patterns shown above.")
        
        return '\n'.join(guidance)
    
    def execute_task(self, task: str, interactive: bool = False, follow_patterns: bool = True) -> None:
        """Execute a task using the LLM"""
        print(f"\n🤖 Processing task: {task}\n")
        
        # Build context about current project
        project_context = self._get_project_context()
        
        # Analyze and include code patterns if requested
        pattern_guidance = ""
        if follow_patterns:
            try:
                if not self.code_patterns:
                    print("🔍 Analyzing code patterns...")
                    self.analyze_code_patterns()
                pattern_guidance = self._build_pattern_guidance()
            except Exception as e:
                print(f"⚠️  Warning: Could not analyze patterns: {e}\n")
        
        # Create prompt
        prompt = f"""Task: {task}

Current project context:
{project_context}
"""
        
        if pattern_guidance:
            prompt += f"\n{pattern_guidance}\n"
        
        prompt += """
Please help me implement this task. If you need to create files, provide the code in a structured format.
For multiple files, use JSON format with file paths as keys and content as values.
For single file, provide the code directly or in a code block."""
        
        # Get LLM response
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            conversation_history=self.conversation_history if interactive else None
        )
        
        print("📝 LLM Response:\n")
        print(response)
        print("\n" + "="*80 + "\n")
        
        # Parse and create files
        files = self._parse_code_response(response)
        
        if files:
            print("📁 Creating files...\n")
            for file_path, content in files.items():
                if self.project_manager.create_file(file_path, content):
                    print(f"✅ Created: {file_path}")
                else:
                    print(f"❌ Failed to create: {file_path}")
            print()
        
        # Update conversation history for interactive mode
        if interactive:
            self.conversation_history.append({"role": "user", "content": task})
            self.conversation_history.append({"role": "assistant", "content": response})
    
    def _get_project_context(self) -> str:
        """Get context about the current project"""
        files = self.project_manager.list_files()
        if not files:
            return "Empty project directory"
        
        context = f"Project has {len(files)} file(s):\n"
        for file in files[:20]:  # Limit to first 20 files
            content = self.project_manager.read_file(file)
            if content:
                # Include first few lines for context
                lines = content.split("\n")[:5]
                context += f"\n{file}:\n" + "\n".join(lines) + "\n"
        
        return context
    
    def grep_project(self, pattern: str, case_sensitive: bool = False, file_extensions: List[str] = None, max_results: int = 100) -> None:
        """Search for a pattern across the project"""
        print(f"\n🔍 Searching for pattern: '{pattern}'\n")
        
        results = self.project_manager.grep(pattern, case_sensitive, file_extensions, max_results)
        
        if not results:
            print("❌ No matches found.\n")
            return
        
        print(f"✅ Found {len(results)} file(s) with matches:\n")
        
        for result in results:
            print(f"📄 {result['file']} ({result['match_count']} match(es))")
            print("-" * 80)
            for match in result['matches'][:10]:  # Show first 10 matches per file
                print(f"  Line {match['line_number']}: {match['line_content']}")
            if result['match_count'] > 10:
                print(f"  ... and {result['match_count'] - 10} more match(es)")
            print()
    
    def analyze_project(self, analysis_type: str = "comprehensive") -> None:
        """Analyze the complete project using LLM"""
        print(f"\n🔬 Analyzing project...\n")
        
        # Get project structure
        structure = self.project_manager.get_project_structure()
        
        # Get file contents summary
        file_summary = self.project_manager.get_file_contents_summary(max_files=30, max_lines_per_file=50)
        
        # Build analysis prompt
        analysis_prompt = f"""Please analyze this project comprehensively and provide insights.

Project Structure:
- Base Path: {structure['base_path']}
- Total Files: {structure['total_files']}
- Total Size: {structure['total_size']} bytes
- File Types: {json.dumps(structure['file_types'], indent=2)}

Directories:
{chr(10).join(structure['directories'][:50])}

File Contents Summary:
{file_summary}

Please provide:
1. Project Overview: What type of project is this? What is its purpose?
2. Technology Stack: What technologies, frameworks, and languages are used?
3. Architecture: What is the overall architecture and structure?
4. Code Quality: Any observations about code quality, patterns, or best practices?
5. Dependencies: What dependencies and external libraries are used?
6. Potential Issues: Any potential bugs, security concerns, or improvements?
7. Recommendations: Suggestions for improvements, refactoring, or enhancements.

Be thorough and provide actionable insights."""
        
        print("📊 Gathering project information...")
        print(f"   - Found {structure['total_files']} files")
        print(f"   - Analyzing file contents...\n")
        
        # Get LLM analysis
        response = self.llm.generate(
            prompt=analysis_prompt,
            system_prompt="You are an expert software architect and code reviewer. Analyze projects thoroughly and provide detailed, actionable insights.",
            conversation_history=None
        )
        
        print("="*80)
        print("📋 PROJECT ANALYSIS REPORT")
        print("="*80)
        print(response)
        print("="*80 + "\n")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*80)
        print("🚀 MiAgent Interactive Mode")
        print("="*80)
        print("Commands:")
        print("  - Type your tasks or questions to execute them")
        print("  - 'exit', 'quit', or 'q' - Exit the program")
        print("  - 'clear' - Clear conversation history")
        print("  - 'context' - See current project context")
        print("  - 'grep <pattern>' - Search for pattern in project files")
        print("  - 'analyze' - Analyze the complete project")
        print("  - 'patterns' - Check code patterns in the project")
        print("  - 'patterns off' - Disable pattern matching for next task")
        print("  - 'patterns on' - Enable pattern matching (default)")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✅ Conversation history cleared.\n")
                    continue
                
                if user_input.lower() == 'context':
                    context = self._get_project_context()
                    print(f"\n📋 Project Context:\n{context}\n")
                    continue
                
                # Handle grep command
                if user_input.lower().startswith('grep '):
                    pattern = user_input[5:].strip()
                    if pattern:
                        self.grep_project(pattern)
                    else:
                        print("❌ Please provide a search pattern. Usage: grep <pattern>\n")
                    continue
                
                # Handle analyze command
                if user_input.lower() == 'analyze':
                    self.analyze_project()
                    continue
                
                # Handle patterns command
                if user_input.lower() == 'patterns':
                    summary = self.get_code_patterns_summary()
                    print(f"\n{summary}\n")
                    continue
                
                if user_input.lower() == 'patterns off':
                    self.follow_patterns = False
                    print("✅ Pattern matching disabled for next task\n")
                    continue
                
                if user_input.lower() == 'patterns on':
                    self.follow_patterns = True
                    print("✅ Pattern matching enabled\n")
                    continue
                
                # Execute task with pattern matching (if enabled)
                follow_patterns = getattr(self, 'follow_patterns', True)
                self.execute_task(user_input, interactive=True, follow_patterns=follow_patterns)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MiAgent - CLI agent for code generation and project creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python MiAgent.py --interactive
  
  # Single task
  python MiAgent.py "Create a Python web scraper"
  
  # Use Groq (reads GROQ_API_KEY from .env)
  python MiAgent.py --provider groq "Create a REST API"
  
  # Use Ollama local LLM
  python MiAgent.py --provider ollama --model llama3.2 "Create a web scraper"
  
  # Specify provider and model
  python MiAgent.py --provider anthropic --model claude-3-5-sonnet-20241022 "Create a REST API"
  
  # Work in specific directory
  python MiAgent.py --path ./myproject "Initialize a React project"
  
  # Search for pattern in project
  python MiAgent.py --grep "def main"
  
  # Analyze complete project
  python MiAgent.py --analyze
  
  # Check code patterns
  python MiAgent.py --patterns
  
  # Generate code without following patterns
  python MiAgent.py --no-patterns "Create a new module"
        """
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to execute (optional if using --interactive)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "groq", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the LLM provider (or set PROVIDER_API_KEY env var, e.g., GROQ_API_KEY)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to use (default: gpt-4 for OpenAI, claude-3-5-sonnet-20241022 for Anthropic, llama-3.1-70b-versatile for Groq, llama3.2 for Ollama)"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        help="Base URL for Ollama server (default: http://localhost:11434, or set OLLAMA_BASE_URL env var)"
    )
    
    parser.add_argument(
        "--path", "-d",
        type=str,
        default=".",
        help="Base path for project files (default: current directory)"
    )
    
    parser.add_argument(
        "--grep",
        type=str,
        help="Search for a pattern in project files (grep functionality)"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze the complete project"
    )
    
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="Check and display code patterns in the project"
    )
    
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Disable pattern matching when generating code"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.task and not args.grep and not args.analyze and not args.patterns:
        parser.error("Either provide a task, use --interactive mode, --grep, --analyze, or --patterns")
    
    try:
        agent = MiAgent(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            base_path=args.path,
            ollama_base_url=args.ollama_url
        )
        
        # Set pattern matching preference
        if args.no_patterns:
            agent.follow_patterns = False
        
        if args.interactive:
            agent.interactive_mode()
        elif args.grep:
            agent.grep_project(args.grep)
        elif args.analyze:
            agent.analyze_project()
        elif args.patterns:
            summary = agent.get_code_patterns_summary()
            print(f"\n{summary}\n")
        else:
            agent.execute_task(args.task, follow_patterns=not args.no_patterns)
    
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

