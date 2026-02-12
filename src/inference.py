import torch
from PIL import Image
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template

class VisionInferenceEngine:
    def __init__(self, model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"):
        """Initializes the MLLM with Unsloth optimizations."""
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama_3_1")

    def generate_response(self, prompt: str, image: Image.Image, max_new_tokens=128):
        """Generates a response based on an image and a text prompt."""
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Greedily decode for factual consistency
            temperature=0.0,
        )
        
        # Decode only the new tokens (the assistant's response)
        generated_ids = output[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()