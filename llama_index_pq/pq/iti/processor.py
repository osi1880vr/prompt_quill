import os
import shutil
from datetime import datetime
from PIL import Image
from interrogate.molmo import molmo
from generators.automatics.client import automa_client
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Scoring class (from earlier)
class ImageScorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def score(self, original_image, generated_image):
        orig_np = np.array(original_image.convert("RGB"))
        gen_np = np.array(generated_image.resize(original_image.size).convert("RGB"))
        ssim_score = ssim(orig_np, gen_np, multichannel=True, channel_axis=2)
        inputs = self.clip_processor(images=[original_image, generated_image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            clip_score = torch.cosine_similarity(image_features[0], image_features[1], dim=0).item()
        return int((clip_score * 70) + (ssim_score * 30))

class ITIProcessor:
    def __init__(self):
        self.vision = molmo()  # Your 7B 4-bit Molmo
        self.generator = automa_client()
        self.scorer = ImageScorer()
        self.llm = None  # Placeholder—replace with your LLM class
        self.out_dir_good = "./good"
        self.out_dir_drafts = "./drafts"
        os.makedirs(self.out_dir_good, exist_ok=True)
        os.makedirs(self.out_dir_drafts, exist_ok=True)

    def clean_prompt(self, prompt):
        return prompt.replace('\n', ' ').strip()  # Borrowed from your i2i

    def get_image_prompt(self, img):
        # Flexible Molmo prompt—could be a setting later
        molmo_prompt = "Describe this image in a detailed, vivid way for generating a similar image."
        prompt = self.vision.process_image(img, molmo_prompt)
        return self.clean_prompt(prompt)

    def process(self, input_image_path, max_iterations=5, target_score=80):
        # Load input image
        input_image = Image.open(input_image_path).convert("RGB")
        prompt = self.get_image_prompt(input_image)
        negative_prompt = ""  # Add if needed
        best_score = -1
        best_image = None
        best_path = None

        for i in range(max_iterations):
            # Generate image
            response = self.generator.request_generation(prompt, negative_prompt, settings_data={})
            generated_image = response[0]  # First image from array
            print(f"Iteration {i+1}: Prompt = {prompt}")

            # Score it
            score = self.scorer.score(input_image, generated_image)
            print(f"Score = {score}")

            # Save temporarily
            temp_path = f"temp_image_{i}.png"
            generated_image.save(temp_path)
            if score > best_score:
                best_score = score
                best_image = generated_image
                if best_path:
                    os.remove(best_path)  # Clean up old best
                best_path = temp_path
            else:
                shutil.move(temp_path, f"{self.out_dir_drafts}/draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{score}.png")

            # Check if good enough
            if score >= target_score:
                break

            # Refine prompt (LLM placeholder)
            if self.llm:
                feedback = f"Score was {score} (best so far: {best_score}). Improve the prompt."
                prompt = self.llm.generate_response(f"Original prompt: {prompt}\nFeedback: {feedback}")
            else:
                prompt = f"{prompt} with more detail and vibrancy"  # Dummy refinement
                print("LLM not set, using dummy refinement.")

        # Save final image
        output_folder = self.out_dir_good if best_score >= target_score else self.out_dir_drafts
        final_path = f"{output_folder}/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{best_score}.png"
        best_image.save(final_path)
        if best_path != final_path:
            os.remove(best_path)  # Clean up temp
        print(f"Saved to {final_path} with score {best_score}")
        return final_path, best_score, best_image

# Test it
if __name__ == "__main__":
    processor = ITIProcessor()
    final_path, final_score, final_image = processor.process("input_image.png")