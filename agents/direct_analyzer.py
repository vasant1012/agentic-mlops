import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.logger import logger


class DirectTransformerMLflowAnalyzer:
    """
    A direct transformer-based analyzer for MLflow experiments
     without LangChain agents.
    This approach gives you more control over the generation process.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 4096,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # NOQA E501
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, max_new_tokens: int = 2048) -> str:  # NOQA E501
        """Generate response using the transformer model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        )

        if self.device == "cuda":
            inputs = inputs.to("cuda")

        logger.info('Getting response from model!!')
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        logger.info('Decoding response from model!!')
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True  # NOQA E501
        )
        return response.strip()

    def analyze_json_file(self, json_file_path: str) -> str:
        """Analyze MLflow experiments from JSON file."""
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)

            # Create analysis prompt
            prompt = f"""<|im_start|>system
                You are an expert MLflow experiment analyst. Analyze the
                provided experiment data and give actionable insights.
                <|im_end|> <|im_start|>user
                Analyze the following MLflow experiment runs:
                {data}
                Please provide output in json with following keys:
                1. Performance summary
                2. Best performing runs
                3. Parameter trends and recommendations
                4. Key insights and next steps
                <|im_end|>
                <|im_start|>assistant
                """

            return self.generate_response(prompt)

        except Exception as e:
            return f"Error analyzing JSON file: {str(e)}"
