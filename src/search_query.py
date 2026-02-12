from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SubQueryGenerator:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

    def decompose_query(self, query: str):
        """Breaks a user query into simplified search terms."""
        prompt = f"Decompose the following complex query into 2-3 simple web search queries. Query: {query}\nSub-queries:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Simple parsing logic for bulleted output
        return [q.strip("- ") for q in result.split("\n") if q.strip()]