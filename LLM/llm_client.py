from pathlib import Path

from huggingface_hub import snapshot_download
from torch import bfloat16, no_grad
from torch import device
from torch.cuda import is_available
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def download_model(
        path: Path = "LLM/model",
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
):
    """Download model from Hugging Face to local directory."""

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
    def __init__(self, model_path: str):
        if not is_available():
            raise RuntimeError(
                "CUDA is not available.\n"
                "This model is intended to run on an NVIDIA GPU.\n"
                "Please run the code on a machine with CUDA support."
            )
        self.model_name = model_path
        self.device = device("cuda" if is_available() else "cpu")
        try:
            print(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=bfloat16
            ).to(self.device)
            self.model.eval()
            print("✓ Model loaded successfully!")
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
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
        return self.generate(prompt, max_new_tokens)
