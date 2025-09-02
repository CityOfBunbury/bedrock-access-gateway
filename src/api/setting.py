import os
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"
ENABLE_APPLICATION_INFERENCE_PROFILES = os.environ.get("ENABLE_APPLICATION_INFERENCE_PROFILES", "true").lower() != "false"

# Load agent configuration from YAML file in the src directory
# Path relative to this setting.py file
AGENTS_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents.yaml"))
AGENTS = {}
DEFAULT_AGENT = None

if os.path.exists(AGENTS_CONFIG_PATH):
    try:
        with open(AGENTS_CONFIG_PATH, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data and "agents" in config_data:
                for agent_conf in config_data["agents"]:
                    if "name" in agent_conf and "agent_id" in agent_conf and "alias_id" in agent_conf:
                        AGENTS[agent_conf["name"]] = {
                            "agent_id": agent_conf["agent_id"],
                            "alias_id": agent_conf["alias_id"]
                        }
                        if DEFAULT_AGENT is None:
                            DEFAULT_AGENT = agent_conf["name"]
                        logger.info(f"Loaded agent configuration for {agent_conf['name']}")
                    else:
                        logger.warning(f"Skipping invalid agent configuration: {agent_conf}")
            else:
                logger.warning(f"No 'agents' key found or empty configuration in {AGENTS_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error loading agent configuration from {AGENTS_CONFIG_PATH}: {e}")
else:
    logger.warning(f"Agent configuration file not found at {AGENTS_CONFIG_PATH}. No agents will be available.")

# Allow overriding the default agent via environment variable
DEFAULT_AGENT = os.environ.get("DEFAULT_AGENT", DEFAULT_AGENT)

if DEFAULT_AGENT and DEFAULT_AGENT not in AGENTS:
    logger.warning(f"DEFAULT_AGENT '{DEFAULT_AGENT}' not found in loaded configurations. Check agents.yaml or environment variable.")
    # Optionally, set DEFAULT_AGENT back to None or the first available one if the override is invalid
    DEFAULT_AGENT = next(iter(AGENTS)) if AGENTS else None

logger.info(f"Available agents: {list(AGENTS.keys())}")
logger.info(f"Default agent: {DEFAULT_AGENT}")
