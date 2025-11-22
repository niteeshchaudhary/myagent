# agent/utils/config_loader.py
"""
Central configuration loader for the agent system.

Loads and merges configuration from:
- configs/config.yaml (main agent config)
- configs/models.yaml (LLM model configs)
- configs/rag.yaml (RAG config)
- configs/tools.yaml (tool configs)

Provides a unified interface to access all configuration.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    from agent.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class AgentConfig:
    """Main agent configuration from config.yaml"""
    max_steps: int = 20
    enable_memory: bool = True
    verbose: bool = True
    allow_risky_commands: bool = False
    ci_auto_apply: bool = False
    ci_paths: str = "."
    review_mode: bool = False
    max_error_retries: int = 5
    enable_error_recovery: bool = True
    auto_index: bool = True  # automatically index codebase when agent starts


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: str = "logs/agent.log"
    max_size_mb: int = 5
    backups: int = 3


@dataclass
class PathsConfig:
    """Path configuration"""
    workspace: str = "./workspace"
    temp: str = "./temp"
    models: str = "./models"


@dataclass
class SystemConfig:
    """System configuration"""
    allow_subprocess: bool = True
    allow_internet: bool = True
    allow_file_access: bool = True


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration"""
    enabled: bool = True
    servers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AppConfig:
    """Application metadata"""
    name: str = "CodingAgent"
    version: str = "1.0.0"
    mode: str = "production"


@dataclass
class UnifiedConfig:
    """Unified configuration container"""
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    models: Dict[str, Any] = field(default_factory=dict)
    rag: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load(cls, config_dir: Optional[str] = None) -> "UnifiedConfig":
        """
        Load all configuration files from the configs directory.
        
        Args:
            config_dir: Path to configs directory. Defaults to ./configs relative to project root.
        """
        if config_dir is None:
            # Try to find configs directory relative to this file
            current_file = os.path.abspath(__file__)
            # Go up from agent/utils/config_loader.py to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            config_dir = os.path.join(project_root, "configs")
        
        config_dir = os.path.abspath(config_dir)
        logger.info("Loading configuration from: %s", config_dir)
        
        cfg = cls()
        
        if yaml is None:
            logger.warning("PyYAML not installed. Configuration files will not be loaded.")
            return cfg
        
        # Load config.yaml
        config_file = os.path.join(config_dir, "config.yaml")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    cfg._load_main_config(data)
                    logger.info("Loaded config.yaml")
            except Exception as e:
                logger.exception("Failed to load config.yaml: %s", e)
        else:
            logger.debug("config.yaml not found at %s", config_file)
        
        # Load models.yaml
        models_file = os.path.join(config_dir, "models.yaml")
        if os.path.exists(models_file):
            try:
                with open(models_file, "r", encoding="utf-8") as f:
                    cfg.models = yaml.safe_load(f) or {}
                    logger.info("Loaded models.yaml")
            except Exception as e:
                logger.exception("Failed to load models.yaml: %s", e)
        else:
            logger.debug("models.yaml not found at %s", models_file)
        
        # Load rag.yaml
        rag_file = os.path.join(config_dir, "rag.yaml")
        if os.path.exists(rag_file):
            try:
                with open(rag_file, "r", encoding="utf-8") as f:
                    cfg.rag = yaml.safe_load(f) or {}
                    logger.info("Loaded rag.yaml")
            except Exception as e:
                logger.exception("Failed to load rag.yaml: %s", e)
        else:
            logger.debug("rag.yaml not found at %s", rag_file)
        
        # Load tools.yaml
        tools_file = os.path.join(config_dir, "tools.yaml")
        if os.path.exists(tools_file):
            try:
                with open(tools_file, "r", encoding="utf-8") as f:
                    cfg.tools = yaml.safe_load(f) or {}
                    logger.info("Loaded tools.yaml")
            except Exception as e:
                logger.exception("Failed to load tools.yaml: %s", e)
        else:
            logger.debug("tools.yaml not found at %s", tools_file)
        
        return cfg
    
    def _load_main_config(self, data: Dict[str, Any]) -> None:
        """Load main configuration from config.yaml data"""
        # App config
        if "app" in data:
            app_data = data["app"]
            if isinstance(app_data, dict):
                self.app = AppConfig(
                    name=app_data.get("name", self.app.name),
                    version=app_data.get("version", self.app.version),
                    mode=app_data.get("mode", self.app.mode)
                )
        
        # Logging config
        if "logging" in data:
            log_data = data["logging"]
            if isinstance(log_data, dict):
                self.logging = LoggingConfig(
                    level=log_data.get("level", self.logging.level),
                    file=log_data.get("file", self.logging.file),
                    max_size_mb=log_data.get("max_size_mb", self.logging.max_size_mb),
                    backups=log_data.get("backups", self.logging.backups)
                )
        
        # Agent config
        if "agent" in data:
            agent_data = data["agent"]
            if isinstance(agent_data, dict):
                self.agent = AgentConfig(
                    max_steps=agent_data.get("max_steps", self.agent.max_steps),
                    enable_memory=agent_data.get("enable_memory", self.agent.enable_memory),
                    verbose=agent_data.get("verbose", self.agent.verbose),
                    allow_risky_commands=agent_data.get("allow_risky_commands", self.agent.allow_risky_commands),
                    ci_auto_apply=agent_data.get("ci_auto_apply", self.agent.ci_auto_apply),
                    ci_paths=agent_data.get("ci_paths", self.agent.ci_paths),
                    review_mode=agent_data.get("review_mode", self.agent.review_mode),
                    max_error_retries=agent_data.get("max_error_retries", self.agent.max_error_retries),
                    enable_error_recovery=agent_data.get("enable_error_recovery", self.agent.enable_error_recovery),
                    auto_index=agent_data.get("auto_index", self.agent.auto_index)
                )
        
        # Paths config
        if "paths" in data:
            paths_data = data["paths"]
            if isinstance(paths_data, dict):
                self.paths = PathsConfig(
                    workspace=paths_data.get("workspace", self.paths.workspace),
                    temp=paths_data.get("temp", self.paths.temp),
                    models=paths_data.get("models", self.paths.models)
                )
        
        # System config
        if "system" in data:
            sys_data = data["system"]
            if isinstance(sys_data, dict):
                self.system = SystemConfig(
                    allow_subprocess=sys_data.get("allow_subprocess", self.system.allow_subprocess),
                    allow_internet=sys_data.get("allow_internet", self.system.allow_internet),
                    allow_file_access=sys_data.get("allow_file_access", self.system.allow_file_access)
                )
        
        # MCP config
        if "mcp" in data:
            mcp_data = data["mcp"]
            if isinstance(mcp_data, dict):
                servers = mcp_data.get("servers")
                if servers is None:
                    servers = []
                elif not isinstance(servers, list):
                    servers = []
                self.mcp = MCPConfig(
                    enabled=mcp_data.get("enabled", self.mcp.enabled),
                    servers=servers
                )


# Global config instance (lazy-loaded)
_global_config: Optional[UnifiedConfig] = None


def get_config(reload: bool = False) -> UnifiedConfig:
    """
    Get the global configuration instance.
    
    Args:
        reload: If True, reload configuration from files.
    
    Returns:
        UnifiedConfig instance
    """
    global _global_config
    if _global_config is None or reload:
        _global_config = UnifiedConfig.load()
    return _global_config

