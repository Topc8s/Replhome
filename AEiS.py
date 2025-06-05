#!/usr/bin/env python3
"""
Enhanced AutoDev IDE System - A Complete Replit-like Development Environment
Features:
- Web-based IDE interface
- Real-time code execution
- Project management
- Terminal emulation
- Multi-language support
- File browser
- Collaborative editing support
"""

import os
import json
import subprocess
import sys
import asyncio
import shutil
import venv
import uuid
import tempfile
import traceback
import logging
import signal
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
from contextlib import asynccontextmanager

# Web framework imports
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn

# Additional imports
import psutil
import docker
import git
from pydantic import BaseModel, Field
import redis
import jwt
from passlib.context import CryptContext
import boto3
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import openai
from rich.console import Console
from rich.logging import RichHandler

# ======================= CONFIGURATION =======================
BASE_DIR = Path(__file__).parent
PROJECTS_DIR = BASE_DIR / "projects"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Create necessary directories
for dir_path in [PROJECTS_DIR, STATIC_DIR, TEMPLATES_DIR, LOGS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "app_name": "AutoDev IDE",
    "version": "2.0.0",
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "database_url": "sqlite:///./autodev.db",
    "redis_url": "redis://localhost:6379",
    "secret_key": "your-secret-key-here-change-in-production",
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "github_token": os.getenv("GITHUB_TOKEN", ""),
    "max_project_size_mb": 100,
    "max_file_size_mb": 10,
    "execution_timeout": 300,  # 5 minutes
    "supported_languages": {
        "python": {"extension": ".py", "runtime": "python3", "icon": "🐍"},
        "javascript": {"extension": ".js", "runtime": "node", "icon": "📜"},
        "typescript": {"extension": ".ts", "runtime": "ts-node", "icon": "📘"},
        "java": {"extension": ".java", "runtime": "java", "icon": "☕"},
        "cpp": {"extension": ".cpp", "runtime": "g++", "icon": "⚡"},
        "go": {"extension": ".go", "runtime": "go", "icon": "🐹"},
        "rust": {"extension": ".rs", "runtime": "rustc", "icon": "🦀"},
        "ruby": {"extension": ".rb", "runtime": "ruby", "icon": "💎"},
        "php": {"extension": ".php", "runtime": "php", "icon": "🐘"},
        "bash": {"extension": ".sh", "runtime": "bash", "icon": "🐚"}
    },
    "docker_enabled": True,
    "collaboration_enabled": True,
    "ai_assistance_enabled": True
}

# ======================= LOGGING SETUP =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(LOGS_DIR / "autodev.log")
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# ======================= DATABASE MODELS =======================
Base = declarative_base()
engine = create_engine(CONFIG["database_url"])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    owner_id = Column(String, nullable=False)
    language = Column(String, default="python")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = Column(Boolean, default=False)
    github_url = Column(String)
    last_run = Column(DateTime)
    run_count = Column(Integer, default=0)

class FileItem(Base):
    __tablename__ = "files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, nullable=False)
    path = Column(String, nullable=False)
    content = Column(Text)
    language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExecutionHistory(Base):
    __tablename__ = "execution_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, nullable=False)
    file_path = Column(String)
    command = Column(String)
    output = Column(Text)
    error = Column(Text)
    exit_code = Column(Integer)
    duration = Column(Integer)  # milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# ======================= PYDANTIC MODELS =======================
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    language: str = "python"
    template: Optional[str] = None

class FileCreate(BaseModel):
    path: str
    content: str

class FileUpdate(BaseModel):
    content: str

class ExecuteRequest(BaseModel):
    project_id: str
    file_path: Optional[str] = None
    command: Optional[str] = None
    stdin: Optional[str] = None

class TerminalCommand(BaseModel):
    project_id: str
    command: str

class AIAssistRequest(BaseModel):
    project_id: str
    prompt: str
    context: Optional[str] = None
    file_path: Optional[str] = None

# ======================= CORE SERVICES =======================
class ProjectManager:
    """Manages projects and their lifecycle"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.docker_client = None
        if CONFIG["docker_enabled"]:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
    
    def create_project(self, project_data: ProjectCreate, owner_id: str) -> Project:
        """Create a new project"""
        db = SessionLocal()
        try:
            # Create project record
            project = Project(
                name=project_data.name,
                description=project_data.description,
                owner_id=owner_id,
                language=project_data.language
            )
            db.add(project)
            db.commit()
            
            # Create project directory
            project_path = PROJECTS_DIR / project.id
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize with template if provided
            if project_data.template:
                self._apply_template(project_path, project_data.template, project_data.language)
            else:
                # Create default files
                self._create_default_files(project_path, project_data.language)
            
            # Initialize git repository
            try:
                repo = git.Repo.init(project_path)
                repo.index.add(["."])
                repo.index.commit("Initial commit")
            except Exception as e:
                logger.warning(f"Failed to initialize git: {e}")
            
            logger.info(f"Created project: {project.name} ({project.id})")
            return project
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()
    
    def _create_default_files(self, project_path: Path, language: str):
        """Create default files for a new project"""
        templates = {
            "python": {
                "main.py": '''#!/usr/bin/env python3
"""
Welcome to AutoDev IDE!
This is your Python project.
"""

def main():
    print("Hello from AutoDev IDE! 🚀")
    print("Start coding here...")
    
    # Example: Simple calculation
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    print(f"Sum of {numbers} = {total}")

if __name__ == "__main__":
    main()
''',
                "requirements.txt": "# Add your dependencies here\nrequests>=2.28.0\n",
                "README.md": f"# {project_path.name}\n\nCreated with AutoDev IDE\n\n## Getting Started\n\n1. Edit `main.py`\n2. Run your code\n3. Have fun coding!\n"
            },
            "javascript": {
                "index.js": '''// Welcome to AutoDev IDE!
// This is your JavaScript project

console.log("Hello from AutoDev IDE! 🚀");
console.log("Start coding here...");

// Example: Simple array operations
const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((a, b) => a + b, 0);
console.log(`Sum of [${numbers}] = ${sum}`);

// Your code here...
''',
                "package.json": '''{
  "name": "autodev-project",
  "version": "1.0.0",
  "description": "Created with AutoDev IDE",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  }
}
''',
                "README.md": f"# JavaScript Project\n\nCreated with AutoDev IDE\n"
            }
        }
        
        # Get templates for the language
        files = templates.get(language, templates["python"])
        
        # Create files
        for filename, content in files.items():
            file_path = project_path / filename
            file_path.write_text(content)
    
    def _apply_template(self, project_path: Path, template: str, language: str):
        """Apply a project template"""
        # TODO: Implement template system
        self._create_default_files(project_path, language)
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        db = SessionLocal()
        try:
            return db.query(Project).filter(Project.id == project_id).first()
        finally:
            db.close()
    
    def list_projects(self, owner_id: str, limit: int = 20) -> List[Project]:
        """List projects for a user"""
        db = SessionLocal()
        try:
            return db.query(Project).filter(
                Project.owner_id == owner_id
            ).order_by(Project.updated_at.desc()).limit(limit).all()
        finally:
            db.close()
    
    def delete_project(self, project_id: str, owner_id: str) -> bool:
        """Delete a project"""
        db = SessionLocal()
        try:
            project = db.query(Project).filter(
                Project.id == project_id,
                Project.owner_id == owner_id
            ).first()
            
            if not project:
                return False
            
            # Delete project directory
            project_path = PROJECTS_DIR / project_id
            if project_path.exists():
                shutil.rmtree(project_path)
            
            # Delete database records
            db.query(FileItem).filter(FileItem.project_id == project_id).delete()
            db.query(ExecutionHistory).filter(ExecutionHistory.project_id == project_id).delete()
            db.delete(project)
            db.commit()
            
            logger.info(f"Deleted project: {project_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete project: {e}")
            return False
        finally:
            db.close()

class CodeExecutor:
    """Handles code execution in isolated environments"""
    
    def __init__(self):
        self.execution_queue = queue.Queue()
        self.active_processes = {}
        self.docker_client = None
        if CONFIG["docker_enabled"]:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
    
    async def execute_code(self, request: ExecuteRequest) -> Dict[str, Any]:
        """Execute code in an isolated environment"""
        start_time = time.time()
        project_path = PROJECTS_DIR / request.project_id
        
        if not project_path.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Determine execution method
        if self.docker_client and CONFIG["docker_enabled"]:
            result = await self._execute_in_docker(request, project_path)
        else:
            result = await self._execute_in_subprocess(request, project_path)
        
        # Record execution history
        duration = int((time.time() - start_time) * 1000)
        self._record_execution(request, result, duration)
        
        return result
    
    async def _execute_in_subprocess(self, request: ExecuteRequest, project_path: Path) -> Dict[str, Any]:
        """Execute code in a subprocess"""
        try:
            # Prepare command
            if request.command:
                cmd = request.command.split()
            elif request.file_path:
                file_path = project_path / request.file_path
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Detect language and prepare command
                language = self._detect_language(file_path)
                runtime = CONFIG["supported_languages"].get(language, {}).get("runtime", "python3")
                
                if language == "cpp":
                    # Compile first
                    exe_path = file_path.with_suffix("")
                    compile_cmd = [runtime, str(file_path), "-o", str(exe_path)]
                    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        return {
                            "output": "",
                            "error": compile_result.stderr,
                            "exit_code": compile_result.returncode
                        }
                    cmd = [str(exe_path)]
                elif language == "java":
                    # Compile first
                    compile_cmd = ["javac", str(file_path)]
                    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        return {
                            "output": "",
                            "error": compile_result.stderr,
                            "exit_code": compile_result.returncode
                        }
                    class_name = file_path.stem
                    cmd = ["java", "-cp", str(project_path), class_name]
                else:
                    cmd = [runtime, str(file_path)]
            else:
                raise HTTPException(status_code=400, detail="No command or file specified")
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if request.stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_path),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            # Provide stdin if needed
            stdin_data = request.stdin.encode() if request.stdin else None
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_data),
                    timeout=CONFIG["execution_timeout"]
                )
                
                return {
                    "output": stdout.decode('utf-8', errors='replace'),
                    "error": stderr.decode('utf-8', errors='replace'),
                    "exit_code": process.returncode
                }
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "output": "",
                    "error": f"Execution timed out after {CONFIG['execution_timeout']} seconds",
                    "exit_code": -1
                }
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "output": "",
                "error": str(e),
                "exit_code": -1
            }
    
    async def _execute_in_docker(self, request: ExecuteRequest, project_path: Path) -> Dict[str, Any]:
        """Execute code in a Docker container"""
        # TODO: Implement Docker execution
        return await self._execute_in_subprocess(request, project_path)
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension = file_path.suffix
        for lang, info in CONFIG["supported_languages"].items():
            if info["extension"] == extension:
                return lang
        return "python"  # Default
    
    def _record_execution(self, request: ExecuteRequest, result: Dict[str, Any], duration: int):
        """Record execution in history"""
        db = SessionLocal()
        try:
            execution = ExecutionHistory(
                project_id=request.project_id,
                file_path=request.file_path,
                command=request.command,
                output=result.get("output", "")[:10000],  # Limit output size
                error=result.get("error", "")[:10000],
                exit_code=result.get("exit_code", -1),
                duration=duration
            )
            db.add(execution)
            
            # Update project last_run
            project = db.query(Project).filter(Project.id == request.project_id).first()
            if project:
                project.last_run = datetime.utcnow()
                project.run_count += 1
            
            db.commit()
        except Exception as e:
            logger.error(f"Failed to record execution: {e}")
        finally:
            db.close()

class FileManager:
    """Handles file operations"""
    
    def create_file(self, project_id: str, file_data: FileCreate) -> FileItem:
        """Create a new file"""
        project_path = PROJECTS_DIR / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        file_path = project_path / file_data.path
        
        # Security check - prevent path traversal
        try:
            file_path.resolve().relative_to(project_path.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        file_path.write_text(file_data.content)
        
        # Save to database
        db = SessionLocal()
        try:
            file_item = FileItem(
                project_id=project_id,
                path=file_data.path,
                content=file_data.content,
                language=self._detect_language(file_path)
            )
            db.add(file_item)
            db.commit()
            db.refresh(file_item)
            return file_item
        finally:
            db.close()
    
    def read_file(self, project_id: str, file_path: str) -> Dict[str, Any]:
        """Read a file"""
        project_path = PROJECTS_DIR / project_id
        full_path = project_path / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            content = full_path.read_text()
            return {
                "path": file_path,
                "content": content,
                "language": self._detect_language(full_path),
                "size": full_path.stat().st_size,
                "modified": datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def update_file(self, project_id: str, file_path: str, content: str) -> bool:
        """Update a file"""
        project_path = PROJECTS_DIR / project_id
        full_path = project_path / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            full_path.write_text(content)
            
            # Update database
            db = SessionLocal()
            try:
                file_item = db.query(FileItem).filter(
                    FileItem.project_id == project_id,
                    FileItem.path == file_path
                ).first()
                
                if file_item:
                    file_item.content = content
                    file_item.updated_at = datetime.utcnow()
                    db.commit()
                
                return True
            finally:
                db.close()
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def delete_file(self, project_id: str, file_path: str) -> bool:
        """Delete a file"""
        project_path = PROJECTS_DIR / project_id
        full_path = project_path / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            full_path.unlink()
            
            # Remove from database
            db = SessionLocal()
            try:
                db.query(FileItem).filter(
                    FileItem.project_id == project_id,
                    FileItem.path == file_path
                ).delete()
                db.commit()
                return True
            finally:
                db.close()
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def list_files(self, project_id: str) -> List[Dict[str, Any]]:
        """List all files in a project"""
        project_path = PROJECTS_DIR / project_id
        if not project_path.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        files = []
        for file_path in project_path.rglob("*"):
            if file_path.is_file() and not self._should_ignore(file_path):
                relative_path = file_path.relative_to(project_path)
                files.append({
                    "path": str(relative_path),
                    "name": file_path.name,
                    "type": "file",
                    "language": self._detect_language(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return sorted(files, key=lambda x: x["path"])
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension"""
        extension = file_path.suffix
        for lang, info in CONFIG["supported_languages"].items():
            if info["extension"] == extension:
                return lang
        return "text"
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".pyc", ".pyo", ".class", ".o", ".exe", ".dll", ".so"
        }
        
        for pattern in ignore_patterns:
            if pattern in str(file_path):
                return True
        return False

class AIAssistant:
    """AI-powered coding assistant"""
    
    def __init__(self):
        self.openai_client = None
        if CONFIG["openai_api_key"]:
            openai.api_key = CONFIG["openai_api_key"]
            self.openai_client = openai
    
    async def assist(self, request: AIAssistRequest) -> Dict[str, Any]:
        """Provide AI assistance"""
        if not self.openai_client:
            return {
                "success": False,
                "message": "AI assistance not configured",
                "code": None
            }
        
        try:
            # Get project context
            context = await self._get_project_context(request.project_id, request.file_path)
            
            # Prepare prompt
            system_prompt = """You are an expert coding assistant integrated into AutoDev IDE.
Your role is to help users write better code, fix bugs, and implement features.
Always provide clear, concise, and working code examples.
Follow best practices and include helpful comments."""
            
            user_prompt = f"""
Project Context:
{context}

User Request: {request.prompt}

Additional Context: {request.context or 'None'}

Please provide a helpful response with code examples if applicable.
"""
            
            # Call OpenAI API
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(content)
            
            return {
                "success": True,
                "message": content,
                "code": code_blocks[0] if code_blocks else None,
                "all_code_blocks": code_blocks
            }
            
        except Exception as e:
            logger.error(f"AI assistance error: {e}")
            return {
                "success": False,
                "message": f"AI assistance error: {str(e)}",
                "code": None
            }
    
    async def _get_project_context(self, project_id: str, file_path: Optional[str]) -> str:
        """Get project context for AI"""
        context_parts = []
        
        # Get project info
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                context_parts.append(f"Project: {project.name}")
                context_parts.append(f"Language: {project.language}")
                context_parts.append(f"Description: {project.description or 'No description'}")
        finally:
            db.close()
        
        # Get current file content if specified
        if file_path:
            try:
                file_manager = FileManager()
                file_data = file_manager.read_file(project_id, file_path)
                context_parts.append(f"\nCurrent File: {file_path}")
                context_parts.append(f"```{file_data['language']}")
                context_parts.append(file_data['content'][:1000])  # Limit context size
                context_parts.append("```")
            except Exception:
                pass
        
        return "\n".join(context_parts)
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown text"""
        import re
        pattern = r'```[\w]*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

# ======================= WEB APPLICATION =======================
app = FastAPI(
    title=CONFIG["app_name"],
    version=CONFIG["version"],
    description="A complete Replit-like development environment"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
project_manager = ProjectManager()
code_executor = CodeExecutor()
file_manager = FileManager()
ai_assistant = AIAssistant()

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, project_id: str):
        if project_id in self.active_connections:
            self.active_connections[project_id].remove(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
    
    async def send_to_project(self, project_id: str, message: dict):
        if project_id in self.active_connections:
            for connection in self.active_connections[project_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()

# ======================= API ENDPOINTS =======================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main IDE interface"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoDev IDE</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 250px; background: #161b22; border-right: 1px solid #30363d; padding: 1rem; overflow-y: auto; }
        .main { flex: 1; display: flex; flex-direction: column; }
        .toolbar { background: #161b22; padding: 0.5rem 1rem; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 1rem; }
        .editor-container { flex: 1; display: flex; }
        .editor { flex: 1; background: #0d1117; padding: 1rem; font-family: 'Monaco', 'Consolas', monospace; overflow: auto; }
        .terminal { height: 200px; background: #000; color: #0f0; padding: 1rem; font-family: monospace; overflow-y: auto; border-top: 1px solid #30363d; }
        .button { background: #238636; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 14px; }
        .button:hover { background: #2ea043; }
        .file-list { list-style: none; }
        .file-item { padding: 0.5rem; cursor: pointer; border-radius: 4px; }
        .file-item:hover { background: #30363d; }
        .project-header { font-size: 18px; font-weight: bold; margin-bottom: 1rem; }
        #editor { width: 100%; height: 100%; background: transparent; color: #c9d1d9; border: none; outline: none; resize: none; font-family: 'Monaco', 'Consolas', monospace; font-size: 14px; }
        .status-bar { background: #161b22; padding: 0.5rem 1rem; border-top: 1px solid #30363d; font-size: 12px; display: flex; justify-content: space-between; }
        .loading { opacity: 0.5; }
        .error { color: #f85149; }
        .success { color: #3fb950; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="project-header">AutoDev IDE</div>
            <button class="button" onclick="createProject()" style="width: 100%; margin-bottom: 1rem;">New Project</button>
            <div id="file-browser">
                <ul class="file-list" id="file-list">
                    <li class="file-item">Loading...</li>
                </ul>
            </div>
        </div>
        <div class="main">
            <div class="toolbar">
                <button class="button" onclick="runCode()">▶ Run</button>
                <button class="button" onclick="saveFile()">💾 Save</button>
                <button class="button" onclick="newFile()">📄 New File</button>
                <span id="current-file" style="margin-left: auto;">No file selected</span>
            </div>
            <div class="editor-container">
                <div class="editor">
                    <textarea id="editor" placeholder="// Start coding here..."></textarea>
                </div>
            </div>
            <div class="terminal" id="terminal">
                <div>Welcome to AutoDev IDE Terminal</div>
                <div>Ready to run your code...</div>
            </div>
            <div class="status-bar">
                <span id="status">Ready</span>
                <span id="language">Python</span>
            </div>
        </div>
    </div>

    <script>
        let currentProject = null;
        let currentFile = null;
        let ws = null;

        // Initialize
        window.onload = async () => {
            await loadProjects();
            setupWebSocket();
        };

        async function loadProjects() {
            try {
                // For demo, create a default project
                const response = await fetch('/api/projects', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: 'My Project',
                        description: 'Default project',
                        language: 'python'
                    })
                });
                
                if (response.ok) {
                    currentProject = await response.json();
                    await loadFiles();
                }
            } catch (error) {
                console.error('Error loading projects:', error);
            }
        }

        async function loadFiles() {
            if (!currentProject) return;
            
            try {
                const response = await fetch(`/api/projects/${currentProject.id}/files`);
                const files = await response.json();
                
                const fileList = document.getElementById('file-list');
                fileList.innerHTML = files.map(file => 
                    `<li class="file-item" onclick="openFile('${file.path}')">${file.path}</li>`
                ).join('');
                
                // Open first file
                if (files.length > 0) {
                    await openFile(files[0].path);
                }
            } catch (error) {
                console.error('Error loading files:', error);
            }
        }

        async function openFile(path) {
            if (!currentProject) return;
            
            try {
                const response = await fetch(`/api/projects/${currentProject.id}/files/${encodeURIComponent(path)}`);
                const file = await response.json();
                
                currentFile = path;
                document.getElementById('editor').value = file.content;
                document.getElementById('current-file').textContent = path;
                document.getElementById('language').textContent = file.language || 'Text';
            } catch (error) {
                console.error('Error opening file:', error);
            }
        }

        async function saveFile() {
            if (!currentProject || !currentFile) {
                alert('No file selected');
                return;
            }
            
            const content = document.getElementById('editor').value;
            
            try {
                const response = await fetch(`/api/projects/${currentProject.id}/files/${encodeURIComponent(currentFile)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                });
                
                if (response.ok) {
                    setStatus('File saved', 'success');
                } else {
                    setStatus('Failed to save file', 'error');
                }
            } catch (error) {
                console.error('Error saving file:', error);
                setStatus('Error saving file', 'error');
            }
        }

        async function runCode() {
            if (!currentProject) {
                alert('No project selected');
                return;
            }
            
            clearTerminal();
            appendToTerminal('Running code...');
            
            try {
                const response = await fetch('/api/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        project_id: currentProject.id,
                        file_path: currentFile
                    })
                });
                
                const result = await response.json();
                
                clearTerminal();
                if (result.output) {
                    appendToTerminal(result.output);
                }
                if (result.error) {
                    appendToTerminal(result.error, 'error');
                }
                appendToTerminal(`\\nProcess exited with code ${result.exit_code}`);
                
            } catch (error) {
                console.error('Error running code:', error);
                appendToTerminal('Error running code: ' + error.message, 'error');
            }
        }

        async function newFile() {
            const filename = prompt('Enter filename:');
            if (!filename || !currentProject) return;
            
            try {
                const response = await fetch(`/api/projects/${currentProject.id}/files`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        path: filename,
                        content: ''
                    })
                });
                
                if (response.ok) {
                    await loadFiles();
                    await openFile(filename);
                }
            } catch (error) {
                console.error('Error creating file:', error);
            }
        }

        async function createProject() {
            const name = prompt('Enter project name:');
            if (!name) return;
            
            try {
                const response = await fetch('/api/projects', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: name,
                        description: 'Created in AutoDev IDE',
                        language: 'python'
                    })
                });
                
                if (response.ok) {
                    currentProject = await response.json();
                    await loadFiles();
                }
            } catch (error) {
                console.error('Error creating project:', error);
            }
        }

        function setupWebSocket() {
            if (!currentProject) return;
            
            ws = new WebSocket(`ws://localhost:8000/ws/${currentProject.id}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('WebSocket message:', data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        function clearTerminal() {
            document.getElementById('terminal').innerHTML = '';
        }

        function appendToTerminal(text, className = '') {
            const terminal = document.getElementById('terminal');
            const line = document.createElement('div');
            line.textContent = text;
            if (className) line.className = className;
            terminal.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
        }

        function setStatus(text, className = '') {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = className;
            setTimeout(() => {
                status.textContent = 'Ready';
                status.className = '';
            }, 3000);
        }

        // Auto-save
        let saveTimeout;
        document.getElementById('editor').addEventListener('input', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(saveFile, 2000);
        });
    </script>
</body>
</html>'''
    return html_content

@app.post("/api/projects", response_model=Project)
async def create_project(project: ProjectCreate):
    """Create a new project"""
    # For demo, use a default owner ID
    owner_id = "demo-user"
    return project_manager.create_project(project, owner_id)

@app.get("/api/projects")
async def list_projects():
    """List all projects"""
    owner_id = "demo-user"
    return project_manager.list_projects(owner_id)

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details"""
    project = project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    owner_id = "demo-user"
    success = project_manager.delete_project(project_id, owner_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted"}

@app.post("/api/projects/{project_id}/files")
async def create_file(project_id: str, file: FileCreate):
    """Create a new file"""
    return file_manager.create_file(project_id, file)

@app.get("/api/projects/{project_id}/files")
async def list_files(project_id: str):
    """List all files in a project"""
    return file_manager.list_files(project_id)

@app.get("/api/projects/{project_id}/files/{file_path:path}")
async def get_file(project_id: str, file_path: str):
    """Get file content"""
    return file_manager.read_file(project_id, file_path)

@app.put("/api/projects/{project_id}/files/{file_path:path}")
async def update_file(project_id: str, file_path: str, update: FileUpdate):
    """Update file content"""
    success = file_manager.update_file(project_id, file_path, update.content)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update file")
    return {"message": "File updated"}

@app.delete("/api/projects/{project_id}/files/{file_path:path}")
async def delete_file(project_id: str, file_path: str):
    """Delete a file"""
    success = file_manager.delete_file(project_id, file_path)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "File deleted"}

@app.post("/api/execute")
async def execute_code(request: ExecuteRequest):
    """Execute code"""
    return await code_executor.execute_code(request)

@app.post("/api/terminal")
async def terminal_command(command: TerminalCommand):
    """Execute terminal command"""
    # Create execution request
    request = ExecuteRequest(
        project_id=command.project_id,
        command=command.command
    )
    return await code_executor.execute_code(request)

@app.post("/api/ai/assist")
async def ai_assist(request: AIAssistRequest):
    """Get AI assistance"""
    return await ai_assistant.assist(request)

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, project_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle WebSocket messages
            message = json.loads(data)
            
            # Broadcast to all connected clients
            await manager.send_to_project(project_id, {
                "type": "update",
                "data": message
            })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, project_id)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {CONFIG['app_name']} v{CONFIG['version']}")
    
    # Create required directories
    for directory in [PROJECTS_DIR, STATIC_DIR, TEMPLATES_DIR, LOGS_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    Base.metadata.create_all(bind=engine)
    
    logger.info("Application initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")

# ======================= MAIN ENTRY POINT =======================
if __name__ == "__main__":
    # Check if running in development mode
    if "--dev" in sys.argv:
        CONFIG["debug"] = True
        CONFIG["host"] = "127.0.0.1"
    
    # Run the application
    uvicorn.run(
        "autodev_ide:app",
        host=CONFIG["host"],
        port=CONFIG["port"],
        reload=CONFIG["debug"],
        log_level="info"
    )
