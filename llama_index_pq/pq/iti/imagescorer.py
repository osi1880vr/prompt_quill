import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imagehash
import numpy as np
from scipy.ndimage import gaussian_laplace
from PIL import ImageStat
import math
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import snapshot_download
from urllib.request import urlretrieve
import globals
from settings.io import settings_io

class ImageScorer:
    def __init__(self):
        self.g = globals.get_globals()
        self.g.settings_data = settings_io().load_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if 'iti' not in self.g.settings_data:
            self.g.settings_data['iti'] = {
                'scorers': {
                    'clip': {'enabled': True, 'weight': 0.4},
                    'ssim': {'enabled': True, 'weight': 0.2},
                    'color': {'enabled': True, 'weight': 0.2},
                    'phash': {'enabled': True, 'weight': 0.1},
                    'aesthetic': {'enabled': True, 'weight': 0.1}
                }
            }
        self.clip_path = snapshot_download(repo_id="openai/clip-vit-base-patch32")
        self.aesthetic_weights = self.download_aesthetic_weights()
        self.clip_model = None
        self.clip_processor = None
        self.aesthetic_model = None

    def download_aesthetic_weights(self):
        home = os.path.expanduser("~")
        cache_dir = os.path.join(home, ".cache", "prompt_quill")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "sa_0_4_vit_b_32_linear.pth")
        if not os.path.exists(weights_path):
            url = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
            urlretrieve(url, weights_path)
        return weights_path

    def compute_clip(self, original, generated):
        if not self.clip_model:
            self.clip_model = CLIPModel.from_pretrained(self.clip_path).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_path)
        inputs = self.clip_processor(images=[original, generated], return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            score = torch.cosine_similarity(features[0], features[1], dim=0).item()
        return int(score * 100)


    def compute_clip_coherence(self, image):
        """Compute standalone quality of an image using CLIP."""
        text_inputs = self.clip_processor(["a high-quality image", "a distorted image"], return_tensors="pt", padding=True).to(self.device)
        image_input = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_input)
            text_features = self.clip_model.get_text_features(**text_inputs)
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        return similarity[0].item()  # Similarity to "high-quality image"

    def compute_ssim(self, original, generated):
        orig_np = np.array(original.convert("RGB"))
        gen_np = np.array(generated.resize(original.size).convert("RGB"))
        score = ssim(orig_np, gen_np, win_size=11, gaussian_weights=True, sigma=2.0,
                     multichannel=True, channel_axis=2)
        return int(score * 100)

    def compute_color_histogram(self, original, generated):
        orig_np = np.array(original.convert("RGB"))
        gen_np = np.array(generated.resize(original.size).convert("RGB"))
        hist_orig = [cv2.calcHist([orig_np], [i], None, [256], [0, 256]) for i in range(3)]
        hist_gen = [cv2.calcHist([gen_np], [i], None, [256], [0, 256]) for i in range(3)]
        score = np.mean([cv2.compareHist(hist_orig[i], hist_gen[i], cv2.HISTCMP_CORREL)
                         for i in range(3)])
        return int(max(0, score) * 100)

    def compute_perceptual_hash(self, original, generated):
        hash_orig = imagehash.phash(original, hash_size=8)
        hash_gen = imagehash.phash(generated.resize(original.size), hash_size=8)
        score = 1 - (hash_orig - hash_gen) / 64.0
        return int(score * 100)

    def aesthetic_logic(self, generated):
        if not self.clip_model:
            self.clip_model = CLIPModel.from_pretrained(self.clip_path).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_path)
        if not self.aesthetic_model:
            self.aesthetic_model = nn.Linear(512, 1).to(self.device)
            state_dict = torch.load(self.aesthetic_weights, map_location=self.device)
            self.aesthetic_model.load_state_dict(state_dict)
            self.aesthetic_model.eval()
        inputs = self.clip_processor(images=generated, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            score = self.aesthetic_model(features).item()  # 0-10 scale
        return int(min(max(score, 0), 10) * 10)  # Scale to 0-100

    def score_(self, original_path, generated_path):
        original = self.load_image_safe(original_path)
        generated = self.load_image_safe(generated_path)
        scores = {}
        settings = self.g.settings_data['iti']['scorers']

        if settings['clip']['enabled']:
            scores['clip'] = self.compute_clip(original, generated)
        if settings['ssim']['enabled']:
            scores['ssim'] = self.compute_ssim(original, generated)
        if settings['color']['enabled']:
            scores['color'] = self.compute_color_histogram(original, generated)
        if settings['phash']['enabled']:
            scores['phash'] = self.compute_perceptual_hash(original, generated)
        if settings['aesthetic']['enabled']:
            scores['aesthetic'] = self.compute_aesthetic(generated)
        if settings.get('detail', {'enabled': False})['enabled']:  # Add to settings if needed
            scores['detail'] = self.compute_detail_density(generated)

        self.clip_model = None
        self.clip_processor = None
        self.aesthetic_model = None
        torch.cuda.empty_cache()

        total = sum(scores[metric] * settings[metric]['weight']
                    for metric in scores) / sum(settings[metric]['weight']
                                                for metric in scores)
        return {"scores": scores, "total": int(total)}

    def cleanup_clip(self):
        self.clip_model = None
        self.clip_processor = None
        self.aesthetic_model = None
        torch.cuda.empty_cache()

    def score__(self, original_path, generated_path, compare=True):
        """Score an image, penalizing trashy outputs."""
        generated = self.load_image_safe(generated_path)
        scores = {}
        penalties = {}
        settings = self.g.settings_data['iti']['scorers']

        # Comparison-based scores (require original)
        if compare and original and generated:
            if settings['clip']['enabled']:
                self.load_clip()
                scores['clip'] = self.compute_clip(original, generated)
            if settings['ssim']['enabled']:
                scores['ssim'] = self.compute_ssim(original, generated)
            if settings['color']['enabled']:
                scores['color'] = self.compute_color_histogram(original, generated)
            if settings['phash']['enabled']:
                scores['phash'] = self.compute_perceptual_hash(original, generated)

        if generated:
            if settings['aesthetic']['enabled']:
                scores['aesthetic'] = self.compute_aesthetic(generated)
            if settings.get('detail', {'enabled': False})['enabled']:
                scores['detail'] = self.compute_detail_density(generated)

            # Trash detection
            gray = generated.convert('L')
            img_array = np.array(gray)

            # Blur check: Low variance = blurry
            variance = ImageStat.Stat(gray).var[0]
            if variance < 20:  # Tune this threshold
                penalties['blur'] = 0.5  # 50% penalty
                print(f"Trash detected: blurry (variance={variance:.1f})")

            # Artifact check: High Laplacian variance = noise/artifacts
            laplacian = gaussian_laplace(img_array, sigma=1.0)
            lap_var = np.var(laplacian)
            if lap_var > 1000:  # Early detection for subtle noise
                # Penalty: ~5% at 1000, ~15% at 4000, 40% at 5000, 80%+ at 10000+
                base_penalty = 0.05  # Light base at 1000
                scale_factor = 0.35  # Ramp from 5000
                if lap_var < 5000:
                    penalty = base_penalty * (lap_var / 5000)  # Linear light touch
                else:
                    penalty = min(0.9, 0.4 + scale_factor * math.log10(lap_var / 5000))
                penalties['artifacts'] = penalty

                print(f"Trash detected: artifacts (laplacian_var={lap_var:.1f}, penalty={penalty:.2f})")

            # Coherence check: CLIP zero-shot classification
            if not self.clip_model:
                self.clip_model = CLIPModel.from_pretrained(self.clip_path).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(self.clip_path)
            text_inputs = self.clip_processor(["a coherent image", "random noise"], return_tensors="pt", padding=True)
            image_input = self.clip_processor(images=generated, return_tensors="pt")
            # Move to CUDA if model is on GPU
            if next(self.clip_model.parameters()).is_cuda:
                image_input = {k: v.to('cuda') for k, v in image_input.items()}
                text_inputs = {k: v.to('cuda') for k, v in text_inputs.items()}
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_input)
                text_features = self.clip_model.get_text_features(**text_inputs)
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
            if similarity[0] < 0.2:  # Low coherence to "coherent image"
                penalties['incoherence'] = 0.6  # 60% penalty
                print(f"Trash detected: incoherent (similarity={similarity[0]:.2f})")

        self.cleanup_clip()  # Assuming you have this for cleanup

        if not scores:
            print(f"No scores computed for {generated_path}")
            return {"scores": {}, "total": 0}

        total = sum(scores[metric] * settings[metric]['weight'] for metric in scores)
        total_weight = sum(settings[metric]['weight'] for metric in scores)
        base_score = total / total_weight if total_weight else 0

        # Apply penalties
        penalty_factor = 1.0
        for reason, factor in penalties.items():
            penalty_factor *= (1 - factor)  # Stack penalties (e.g., 0.5 * 0.6 = 0.3)
        final_score = base_score * penalty_factor

        print(f"Base score: {base_score:.1f}, Penalties: {penalties}, Final score: {final_score:.1f}")
        return {"scores": scores, "total": int(final_score)}

    def load_clip(self):
        if not self.clip_model:
            self.clip_model = CLIPModel.from_pretrained(self.clip_path).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_path)

    def compute_blur_score(self, img_array):
        variance = cv2.Laplacian(img_array, cv2.CV_64F).var()
        return min(variance, 200) / 200  # Normalize to 0-1, cap at 200

    def compute_artifacts_score(self, img_array):
        lap_var = np.var(gaussian_laplace(img_array, sigma=1.0))
        return (5000 - min(lap_var, 5000)) / 5000  # Invert: lower variance = higher score

    def compute_aesthetic(self, image):
        # Assuming this returns 0-100, normalize it
        return self.aesthetic_logic(image) / 100

    def compute_detail_density(self, image):
        # Assuming 0-100, normalize it
        return self.detail_density_logic(image) / 100


    # Fixed penalty methods
    def compute_blur_penalty(self, variance):  # Takes normalized score (0-1)
        raw_variance = variance * 200  # Scale back to original range
        return max(0, 0.5 * (20 - raw_variance) / 20) if raw_variance < 20 else 0

    def compute_artifacts_penalty(self, score):  # Takes normalized score (0-1)
        lap_var = (1 - score) * 5000  # Invert back to raw variance
        return min(0.9, 0.05 + 0.35 * math.log1p(lap_var / 5000)) if lap_var > 1000 else 0


    def score___(self, original_path, generated_path, compare=True):
        """Score an image, penalizing trashy outputs with modular scoring."""
        generated = self.load_image_safe(generated_path)
        original = self.load_image_safe(original_path) if compare and original_path else None
        scores = {}
        penalties = {}
        settings = self.g.settings_data['iti']['scorers']

        # Comparison-based scores (require original)
        if compare and original and generated:
            if settings['clip']['enabled']:
                self.load_clip()
                scores['clip'] = self.compute_clip(original, generated)
            if settings['ssim']['enabled']:
                scores['ssim'] = self.compute_ssim(original, generated)
            if settings['color']['enabled']:
                scores['color'] = self.compute_color_histogram(original, generated)
            if settings['phash']['enabled']:
                scores['phash'] = self.compute_perceptual_hash(original, generated)

        # Generated-only scores
        if generated:
            if settings['aesthetic']['enabled']:
                scores['aesthetic'] = self.compute_aesthetic(generated)
            if settings.get('detail', {'enabled': False})['enabled']:
                scores['detail'] = self.compute_detail_density(generated)
            if settings['gen_clip']['enabled'] and not (compare and original):
                self.load_clip()
                scores['gen_clip'] = self.compute_clip_coherence(generated)  # Standalone quality

            gray = generated.convert('L')
            gray = np.array(gray)

            if settings['blur']['enabled']:
                blur_score = self.compute_blur_score(gray)
                scores['blur'] = blur_score
                blur_penalty = self.compute_blur_penalty(blur_score)
                if blur_penalty > 0:
                    penalties['blur'] = blur_penalty
                    print(f"Trash detected: blurry (score={blur_score:.1f}, penalty={blur_penalty:.2f})")

            if settings['artifacts']['enabled']:
                art_score = self.compute_artifacts_score(gray)
                scores['artifacts'] = art_score
                art_penalty = self.compute_artifacts_penalty(art_score)
                if art_penalty > 0:
                    penalties['artifacts'] = art_penalty
                    print(f"Trash detected: artifacts (score={art_score:.1f}, penalty={art_penalty:.2f})")

        self.cleanup_clip()

        if not scores:
            print(f"No scores computed for {generated_path}")
            return {"scores": {}, "total": 0}

        # Weighted base score
        total = sum(scores[k] * settings[k]['weight'] for k in scores)
        total_weight = sum(settings[k]['weight'] for k in scores)
        base_score = total / total_weight if total_weight else 0

        # Apply penalties with exponential decay
        penalty_factor = math.exp(-sum(penalties.values()))
        final_score = base_score * penalty_factor

        print(f"Base score: {base_score:.2f}, Penalties: {penalties}, Final score: {final_score:.2f}")
        return {"scores": scores, "penalties": penalties, "total": round(final_score)}


    def score(self, original_path, generated_path, compare=True):
        """Score an image, penalizing trashy outputs with modular scoring."""
        generated = self.load_image_safe(generated_path)
        original = self.load_image_safe(original_path) if compare and original_path else None
        scores = {}
        penalties = {}
        settings = self.g.settings_data['iti']
        scorers_key = 'scorers' if compare else 'scorers_single'
        active_settings = settings[scorers_key]

        # Comparison-based scores
        if compare and original and generated:
            if active_settings['clip']['enabled']:
                self.load_clip()
                scores['clip'] = self.compute_clip(original, generated) / 100
            if active_settings['ssim']['enabled']:
                scores['ssim'] = self.compute_ssim(original, generated) / 100
            if active_settings['color']['enabled']:
                scores['color'] = self.compute_color_histogram(original, generated) / 100
            if active_settings['phash']['enabled']:
                scores['phash'] = self.compute_perceptual_hash(original, generated) / 100

        # Generated-only scores (available in both modes if enabled)
        if generated:
            if active_settings['aesthetic']['enabled']:
                scores['aesthetic'] = self.compute_aesthetic(generated)
            if active_settings.get('detail', {'enabled': False})['enabled']:
                scores['detail'] = self.compute_detail_density(generated)
            if active_settings.get('gen_clip', {'enabled': False})['enabled']:
                self.load_clip()
                scores['gen_clip'] = self.compute_clip_coherence(generated)  # Now runs regardless of compare

            gray = generated.convert('L')
            img_array = np.array(gray)

            if active_settings['blur']['enabled']:
                blur_score = self.compute_blur_score(img_array)
                scores['blur'] = blur_score
                blur_penalty = self.compute_blur_penalty(blur_score)
                if blur_penalty > 0:
                    penalties['blur'] = blur_penalty
                    print(f"Trash detected: blurry (score={blur_score:.2f}, penalty={blur_penalty:.2f})")

            if active_settings['artifacts']['enabled']:
                art_score = self.compute_artifacts_score(img_array)
                scores['artifacts'] = art_score
                art_penalty = self.compute_artifacts_penalty(art_score)
                if art_penalty > 0:
                    penalties['artifacts'] = art_penalty
                    print(f"Trash detected: artifacts (score={art_score:.2f}, penalty={art_penalty:.2f})")

        self.cleanup_clip()

        if not scores:
            print(f"No scores computed for {generated_path}")
            return {"scores": {}, "penalties": {}, "total": 0}

        active_weights = {k: active_settings[k]['weight'] for k in scores if k in active_settings and active_settings[k]['enabled']}
        base_score = sum(scores[k] * active_weights[k] for k in scores)
        penalty_factor = math.exp(-sum(penalties.values()))
        final_score = base_score * penalty_factor * 100

        print(f"Base score: {base_score:.2f}, Penalties: {penalties}, Final score: {final_score:.2f}")
        return {"scores": scores, "penalties": penalties, "total": round(final_score)}


    def load_image_safe(self, filepath):
        try:
            img = Image.open(filepath).convert("RGB")
            return img
        except Exception as e:
            raise ValueError(f"Failed to load image {filepath}: {e}")


    def compute_color_vibrancy(self, image):
        """Score color richness based on HSV variance."""
        img_np = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hue_var = np.var(hsv[:, :, 0])  # Hue variance
        sat_mean = np.mean(hsv[:, :, 1])  # Saturation mean
        score = (hue_var / 180) * 50 + (sat_mean / 255) * 50  # Normalize to 0-100
        return int(min(max(score, 0), 100))

    def detail_density_logic(self, image):
        """Score detail via edge detection."""
        img_np = np.array(image.convert("L"))  # Grayscale
        edges = cv2.Canny(img_np, 100, 200)
        edge_count = np.sum(edges > 0)
        max_edges = img_np.size * 0.1  # Rough cap for "max detail"
        score = (edge_count / max_edges) * 100
        return int(min(max(score, 0), 100))

    def score_single(self, image_path):
        """Score a single image without comparison."""
        image = self.load_image_safe(image_path)
        scores = {}
        settings = self.g.settings_data['iti']['scorers']
        if settings['clip']['enabled']:
            scores['clip'] = 0
        if settings['ssim']['enabled']:
            scores['ssim'] = 0
        if settings['color']['enabled']:
            scores['color'] = self.compute_color_vibrancy(image)
        if settings['aesthetic']['enabled']:
            scores['aesthetic'] = self.compute_aesthetic(image)
        if settings['phash']['enabled']:
            scores['phash'] = 0
        if settings.get('detail', {'enabled': False})['enabled']:  # Add to settings if needed
            scores['detail'] = self.compute_detail_density(image)

        torch.cuda.empty_cache()
        return {"scores": scores, "total": int(sum(scores.values()) / len(scores)) if scores else 0}


# Test it
if __name__ == "__main__":
    scorer = ImageScorer()
    result = scorer.score("tanned_busty_model_in_pink.jpg", "tanned_natural_bathing_beauty.jpg")
    print(result)