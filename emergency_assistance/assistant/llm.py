from __future__ import annotations

import logging
import shutil
import httpx

LOGGER = logging.getLogger(__name__)


class PromptBuilder:
    """Constructs prompts for CARE Assistant, integrating system prompts, safety guidelines, and retrieved context."""

    SYSTEM_PROMPT = (
        "You are CARE, a calm, expert emergency assistant for electric-vehicle crashes.\n"
        "This is safety information, not a substitute for local emergency responders."
    )

    SAFETY_GUIDELINES = (
        "1. Prioritize life safety above all else.\n"
        "2. For fire, smoke, trapped people, unconsciousness, severe bleeding, or immediate danger:\n"
        "   Advise calling local emergency services (e.g. 911) immediately before doing anything else.\n"
        "3. Never diagnose injuries or vehicle technical faults.\n"
        "4. Refuse to provide technical instructions or vehicle-specific procedures if no manuals or context are available.\n"
        "5. Advise contacting emergency services for all dangerous situations."
    )

    @classmethod
    def build(cls, query: str, context: str | None) -> str:
        """Combines all elements into a structured LLM prompt."""
        formatted_context = context.strip() if context else "No matching local manual is indexed."

        return f"""System Prompt:
{cls.SYSTEM_PROMPT}

Safety Guidelines (Must be followed strictly):
{cls.SAFETY_GUIDELINES}

Knowledge Base Context (Use this ONLY to answer technical vehicle procedures. Do not invent details):
{formatted_context}

User Query:
{query}

Calm, safe, numbered emergency response steps:
Answer:"""


class ModelManager:
    """Manages Ollama service status, model availability, and text generation."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def check_ollama_installed(self) -> bool:
        """Checks if the ollama executable is present in PATH."""
        return shutil.which("ollama") is not None

    def check_server_running(self) -> bool:
        """Checks if the Ollama server is responsive at base_url."""
        try:
            response = httpx.get(self.base_url, timeout=3.0)
            return response.status_code == 200 or "Ollama is running" in response.text
        except httpx.RequestError:
            return False

    def verify_model_exists(self) -> bool:
        """Verifies if the configured model is pulled in Ollama."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            if response.status_code != 200:
                return False
            data = response.json()
            models = data.get("models", [])
            installed_names = [m.get("name") for m in models]
            installed_models = [m.get("model") for m in models]

            target = self.model.strip()
            # Match exact target name or with standard tags
            for name in installed_names + installed_models:
                if name == target or name == f"{target}:latest" or (":" not in target and name.startswith(f"{target}:")):
                    return True
            return False
        except Exception as error:
            LOGGER.error(f"Error checking model presence in Ollama: {error}")
            return False

    def get_status(self) -> dict[str, object]:
        """Returns the detailed status of Ollama connection and model presence."""
        installed = self.check_ollama_installed()
        running = self.check_server_running()
        model_exists = self.verify_model_exists() if running else False

        if not running:
            status = "unreachable"
            if installed:
                message = f"Ollama CLI is installed, but the server is not running on {self.base_url}."
            else:
                message = f"Ollama is not installed (or not in PATH), and server is unreachable on {self.base_url}."
        elif not model_exists:
            status = "missing_model"
            message = f"Ollama server is running, but model '{self.model}' is not pulled. Run 'ollama pull {self.model}'."
        else:
            status = "healthy"
            message = "Ollama is running and model is available."

        return {
            "status": status,
            "installed": installed,
            "running": running,
            "model_exists": model_exists,
            "message": message,
        }

    def generate(self, prompt: str) -> str:
        """Sends prompt to Ollama for text generation. Handles failures gracefully."""
        LOGGER.info("Sending request to Ollama model '%s'", self.model)
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=45.0,
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            if not result:
                raise RuntimeError("Ollama returned an empty response.")
            LOGGER.info("Ollama generation successful.")
            return result
        except httpx.HTTPStatusError as error:
            err_msg = f"Ollama HTTP error {error.response.status_code}: {error.response.text}"
            LOGGER.error(err_msg)
            raise RuntimeError(err_msg) from error
        except httpx.RequestError as error:
            status = self.get_status()
            err_msg = f"Ollama server is unreachable: {status['message']} ({error})"
            LOGGER.error(err_msg)
            raise RuntimeError(err_msg) from error
