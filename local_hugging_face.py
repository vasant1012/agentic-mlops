from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalHFAdapter:
    """
    Local HuggingFace pipeline adapter for Qwen (or any HF model).
    Works fully offline on CPU.
    """

    def __init__(self, model_id="Qwen/Qwen2-1.5B", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu")
        print(f"Loading local model {model_id} on CPU...")
        # HuggingFace pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.2,
        )

    @property
    def _llm_type(self) -> str:
        """Required by LangChain"""
        return "local_hf"

    @property
    def _identifying_params(self):
        return {"model_id": self.model_id}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.2
        )
        return out[0]["generated_text"]
