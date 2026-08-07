"""Utilities for downloading and running a local Qwen2.5 causal language model.

This module provides a helper to download a Hugging Face model snapshot to a
local directory, and a wrapper class (`LLMModel`) that loads the model onto a
CUDA GPU and generates text using a chat template.
"""

import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import bfloat16, no_grad
from torch import device
from torch.cuda import is_available
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def download_model(
        path: Path = "LLM/model",
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
):
    """Download model from Hugging Face to local directory.

    Args:
        path: Local directory where the model snapshot will be saved.
            Created if it does not already exist.
        model_name: Hugging Face repo ID of the model to download.

    Returns:
        A confirmation message including the model name and the local
        path where the snapshot was saved.

    Raises:
        ValueError: If the download fails for any reason (e.g. network
            error, invalid repo ID, insufficient disk space).
    """

    local_dir = str(path)
    try:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        local_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            repo_type="model"  # or "dataset" for datasets
        )
        return f"✓ Model '{model_name}' downloaded to '{local_path}'"
    except Exception:
        raise ValueError(f"Error while downloading the model.")


class LLMModel:
    """Wrapper around a Hugging Face causal LM for chat-style text generation.

    Loads a tokenizer and model from a local directory onto a CUDA GPU in
    bfloat16 precision, and exposes a simple `generate`/`__call__` interface
    that applies the model's chat template to a single user prompt and
    returns the decoded completion.

    Attributes:
        model_name: The path the model was loaded from.
        device: The torch device the model is loaded on (always "cuda";
            CUDA availability is enforced in `__init__`).
        tokenizer: The loaded `AutoTokenizer` instance.
        model: The loaded `AutoModelForCausalLM` instance, in eval mode.
    """

    def __init__(self, model_path: str = "LLM/model"):
        """Load the tokenizer and model from a local directory onto the GPU.

        Args:
            model_path: Path to a local directory containing a Hugging Face
                model and tokenizer (e.g. as produced by `download_model`).

        Raises:
            RuntimeError: If no CUDA-capable GPU is available.
            ValueError: If loading the tokenizer or model fails for any
                other reason (e.g. missing or corrupted files).
        """

        if not is_available():
            raise RuntimeError(
                "CUDA is not available.\n"
                "This model is intended to run on an NVIDIA GPU.\n"
                "Please run the code on a machine with CUDA support."
            )
        self.model_name = model_path
        self.device = device("cuda" if is_available() else "cpu")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            logger.info(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=self.quantization_config,
                dtype=bfloat16
            ).to(self.device)
            self.model.eval()
            logger.info("✓ Model loaded successfully!")
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Generate a text completion for a single user prompt.

        Wraps the prompt in a one-turn chat message, applies the model's
        chat template, and greedily decodes a response (no sampling).

        Args:
            prompt: The user's input text.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            The generated response text, with special tokens stripped.
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)
        with no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            response,
            skip_special_tokens=True
        )

    def __call__(self, prompt: str, max_new_tokens: int = 200):
        """Shorthand for `generate`.

        Args:
            prompt: The user's input text.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            The generated response text.
        """
        return self.generate(prompt, max_new_tokens)
