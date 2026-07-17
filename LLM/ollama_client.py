#!/usr/bin/env python3
"""
Ollama REST API wrapper for generating answers using context chunks retrieved from RAG.
"""

import logging
import requests
from typing import Any, Dict, List

# Set up simple logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OllamaRAGAssistant:
    """Assistant utilizing a local Ollama instance for context-grounded generation."""

    # CHANGED: Default model is now set to "qwen2.5:3b"
    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"

    def _build_prompt(self, question: str, context_chunks: List[Dict[str, str]]) -> str:
        """Grounds the LLM by framing system rules and listing available context chunks."""
        context_str = ""
        for idx, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "").strip()
            source = chunk.get("source", "").strip()
            section = chunk.get("section", "").strip() or "General"
            # Using clean XML-like tags helps Qwen isolate context perfectly
            context_str += f"<chunk id=\"{idx}\" source=\"{source}\" section=\"{section}\">\n{text}\n</chunk>\n\n"

        prompt = (
            "You are the intelligence engine for CARE_DOLL, an emergency assistance and response system for electric vehicles.\n"
            "Your task is to answer the user's question directly, clearly, and concisely, using ONLY the provided vehicle documentation context.\n\n"
            "Rules:\n"
            "1. Ground your answer strictly in the provided vehicle documentation context chunks. Do not assume, extrapolate, or bring in external knowledge.\n"
            "2. If the answer is not explicitly found in the provided context, you must state exactly: \"I'm sorry, I cannot find that information in the vehicle documentation.\"\n"
            "3. Do not mention \"based on the context\" or \"according to the documents\" in your answer. Just answer the question directly.\n\n"
            f"Context Documentation:\n{context_str}\n"
            f"User Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def generate_answer(self, question: str, context_chunks: List[Dict[str, str]], timeout_seconds: int = 30) -> str:
        """Retrieves generated response from the local Ollama instance's generate REST endpoint."""
        prompt = self._build_prompt(question, context_chunks)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=timeout_seconds)
            if response.status_code == 200:
                result_json = response.json()
                return result_json.get("response", "").strip()
            else:
                logger.error(f"Ollama returned status code {response.status_code}: {response.text}")
                return "I'm sorry, I encountered an error communicating with the generation server."

        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Is the server running?")
            return "I'm sorry, the generation service is currently unavailable. Please verify that Ollama is running."

        except requests.exceptions.Timeout:
            logger.error("Ollama generation timed out.")
            return "I'm sorry, the request to the generation service timed out."

        except Exception as exc:
            logger.error(f"Unexpected error in OllamaRAGAssistant: {exc}")
            return "I'm sorry, an unexpected error occurred during answer generation."


# This block handles execution when running the file directly
if __name__ == "__main__":
    # 1. Initialize the assistant with your Qwen model
    assistant = OllamaRAGAssistant(model="qwen2.5:3b")
    
    # 2. Mock some sample RAG context data that a retriever would pass in
    mock_retrieved_chunks = [
        {
            "text": "In the event of a high-voltage battery pack overheating, the CARE_DOLL system will automatically isolate the cell modules and activate the localized coolant flush. The driver will see a red flashing 'HV BATTERY HOT' warning on the main instrument cluster.",
            "source": "Emergency_Procedures_Manual.pdf",
            "section": "Battery Safety"
        },
        {
            "text": "Tire pressure for the EV model standard 19-inch wheels should always be maintained at 42 PSI cold for optimal range and safety.",
            "source": "Owners_Manual.pdf",
            "section": "Maintenance"
        }
    ]
    
    # 3. Ask a question that is inside the context
    test_question = "What warning will the driver see if the high-voltage battery overheats?"
    
    print(f"Asking Qwen: '{test_question}'...\n")
    answer = assistant.generate_answer(question=test_question, context_chunks=mock_retrieved_chunks)
    print("--- Qwen Response ---")
    print(answer)
    print("---------------------\n")

    # 4. Ask a question that is NOT in the context to test your grounding rules
    out_of_bounds_question = "What is the top speed of the car?"
    print(f"Asking Qwen an out-of-context question: '{out_of_bounds_question}'...\n")
    fallback_answer = assistant.generate_answer(question=out_of_bounds_question, context_chunks=mock_retrieved_chunks)
    print("--- Qwen Response ---")
    print(fallback_answer)
    print("---------------------")