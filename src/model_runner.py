import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class HuggingFaceRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading tokenizer for {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Some models don't have a pad token, we can set it to eos token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logging.info(f"Loading model {model_id} to {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
            
        generated_tokens = outputs[0][input_token_count:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text, len(generated_tokens)

    def cleanup(self):
        """Releases the model and frees up GPU/RAM memory."""
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
