# iti/iti_manager.py
import os
import shutil
from PIL import Image
import urllib.request
import json
from interrogate.molmo import molmo
from generators.automatics.client import automa_client
from llm_fw import llm_interface_qdrant
from iti.imagescorer import ImageScorer
import globals
from settings.io import settings_io
import random
from datetime import datetime
import base64
from io import BytesIO
import torch
import math
import gc
from collections import deque
from .sailing_manager import SailingManager
import sqlite3
import spacy
from collections import Counter
import subprocess
import sys
import itertools

from enhancer.prompts import PromptEnhance
import shared

class ITIManager:
    def __init__(self):
        self.last_prompt = None
        self.last_file = None
        self.g = globals.get_globals()
        self.vision = None
        self.generator = automa_client()
        self.scorer = None
        self.llm = llm_interface_qdrant.get_interface()
        self.log = ""
        self.last6_good = deque(maxlen=6)  # Fresh deque
        self.last6_running = deque(maxlen=6)
        self.startup_temperature = self.g.settings_data.get("Temperature", 1.0)
        self.sailing_manager = SailingManager()
        self.original_image = None
        self.actual_score = None
        self.original_scores = None
        self.create_image_scores_db()
        self.prompt_enhancer = PromptEnhance()


    def update_weights(self,
            # Comparison inputs
            clip_enabled, clip_weight, ssim_enabled, ssim_weight,
            color_enabled, color_weight, phash_enabled, phash_weight,
            aesthetic_enabled_comp, aesthetic_weight_comp, gen_clip_enabled_comp, gen_clip_weight_comp, detail_enabled_comp, detail_weight_comp,
            blur_enabled_comp, blur_weight_comp, artifacts_enabled_comp, artifacts_weight_comp,
            # Single inputs
            aesthetic_enabled_single, aesthetic_weight_single, detail_enabled_single, detail_weight_single,
            gen_clip_enabled, gen_clip_weight, blur_enabled_single, blur_weight_single,
            artifacts_enabled_single, artifacts_weight_single
        ):
        # Comparison weights
        comp_weights = {
            "clip": {"enabled": clip_enabled, "weight": clip_weight},
            "ssim": {"enabled": ssim_enabled, "weight": ssim_weight},
            "color": {"enabled": color_enabled, "weight": color_weight},
            "phash": {"enabled": phash_enabled, "weight": phash_weight},
            "aesthetic": {"enabled": aesthetic_enabled_comp, "weight": aesthetic_weight_comp},
            "gen_clip": {"enabled": gen_clip_enabled_comp, "weight": gen_clip_weight_comp},
            "detail": {"enabled": detail_enabled_comp, "weight": detail_weight_comp},
            "blur": {"enabled": blur_enabled_comp, "weight": blur_weight_comp},
            "artifacts": {"enabled": artifacts_enabled_comp, "weight": artifacts_weight_comp}
        }
        comp_total = sum(data["weight"] for data in comp_weights.values() if data["enabled"])
        if comp_total > 0:
            for scorer in comp_weights:
                if comp_weights[scorer]["enabled"]:
                    comp_weights[scorer]["weight"] /= comp_total
        self.g.settings_data["iti"]["scorers"].update(comp_weights)

        # Single weights
        single_weights = {
            "aesthetic": {"enabled": aesthetic_enabled_single, "weight": aesthetic_weight_single},
            "detail": {"enabled": detail_enabled_single, "weight": detail_weight_single},
            "gen_clip": {"enabled": gen_clip_enabled, "weight": gen_clip_weight},
            "blur": {"enabled": blur_enabled_single, "weight": blur_weight_single},
            "artifacts": {"enabled": artifacts_enabled_single, "weight": artifacts_weight_single}
        }
        single_total = sum(data["weight"] for data in single_weights.values() if data["enabled"])
        if single_total > 0:
            for scorer in single_weights:
                if single_weights[scorer]["enabled"]:
                    single_weights[scorer]["weight"] /= single_total
        self.g.settings_data["iti"]["scorers_single"].update(single_weights)

        # Save settings
        settings_io().write_settings(self.g.settings_data)
        # Prepare return values: weights for sliders + status
        comp_outputs = [
            comp_weights["clip"]["weight"] if comp_weights["clip"]["enabled"] else 0,
            comp_weights["ssim"]["weight"] if comp_weights["ssim"]["enabled"] else 0,
            comp_weights["color"]["weight"] if comp_weights["color"]["enabled"] else 0,
            comp_weights["phash"]["weight"] if comp_weights["phash"]["enabled"] else 0,
            comp_weights["aesthetic"]["weight"] if comp_weights["aesthetic"]["enabled"] else 0,
            comp_weights["detail"]["weight"] if comp_weights["detail"]["enabled"] else 0,
            comp_weights["gen_clip"]["weight"] if comp_weights["gen_clip"]["enabled"] else 0,
            comp_weights["blur"]["weight"] if comp_weights["blur"]["enabled"] else 0,
            comp_weights["artifacts"]["weight"] if comp_weights["artifacts"]["enabled"] else 0
        ]
        single_outputs = [
            single_weights["aesthetic"]["weight"] if single_weights["aesthetic"]["enabled"] else 0,
            single_weights["detail"]["weight"] if single_weights["detail"]["enabled"] else 0,
            single_weights["gen_clip"]["weight"] if single_weights["gen_clip"]["enabled"] else 0,
            single_weights["blur"]["weight"] if single_weights["blur"]["enabled"] else 0,
            single_weights["artifacts"]["weight"] if single_weights["artifacts"]["enabled"] else 0
        ]
        return comp_outputs + single_outputs + ["Weights updated and normalized successfully!"]


    def set_iti_settings(self,
                         # Work tab
                         input_folder,

                         # Settings > Folders tab
                         good_folder, drafts_folder, log_file,
                         improvement_threshold,

                         max_iter, target_score,

                         # Settings > Molmo tab
                         molmo_prompt, molmo_tokens, refine_prompt,
                         refine_max_tokens,

                         # Miscellaneous settings
                         automa_count, feedback_style, revert_drop, early_stop):
        """
        Update the settings data based on UI inputs and write to storage.
        """
        # Update only if a job is running for specific fields (similar to sail_depth in the example)

        # Update settings_data with all relevant values
        self.g.settings_data["iti"]["input_folder"] = input_folder

        self.g.settings_data["iti"]["output_dirs"].update({
            "good": good_folder,
            "drafts": drafts_folder
        })

        self.g.settings_data["iti"]["output"].update({
            "log_file": log_file,
            "save_metadata": save_metadata
        })

        self.g.settings_data["iti"]["refinement"].update({
            "improvement_threshold": improvement_threshold,
            "max_iterations": max_iter,
            "target_score": target_score,
            "feedback_style": feedback_style,
            "revert_on_drop": revert_drop,
            "early_stop_gain": early_stop
        })

        self.g.settings_data["iti"]["molmo"].update({
            "custom_prompt": molmo_prompt,
            "max_tokens": molmo_tokens,
            "refine_prompt": refine_prompt,
            "refine_max_tokens": refine_max_tokens
        })

        self.g.settings_data["iti"]["automa"]["image_count"] = automa_count

        # Write the updated settings to storage
        settings_io().write_settings(self.g.settings_data)


    def set_gen_iti_settings(self, # Settings > Generating tab
                             # Prompting tab

                             iti_sail_sampler, iti_sail_scheduler, iti_sail_checkpoint,
                             iti_sail_vae, iti_pos_sail_lora, iti_neg_sail_lora,
                             iti_pos_sail_embedding, iti_neg_sail_embedding,
                             pos_style, pos_lora, automa_neg, neg_style, neg_lora, molmo_style,

                             # Settings > Sailing tab
                             enable_sailing_step, enable_sailing, sail_steps):
        print("set_gen_iti_settings called with:")
        print(f"iti_pos_sail_lora: {iti_pos_sail_lora}")
        print(f"iti_neg_sail_lora: {iti_neg_sail_lora}")
        print(f"iti_pos_sail_embedding: {iti_pos_sail_embedding}")
        print(f"iti_neg_sail_embedding: {iti_neg_sail_embedding}")


        if self.g.job_running:
            self.sail_depth_start = sail_steps  # Using sail_steps as an analog to sail_depth

        self.g.settings_data["iti"]["prompt"].update({
            "pos_style": pos_style,
            "pos_lora": pos_lora,
            "neg_style": neg_style,
            "neg_lora": neg_lora
        })

        self.g.settings_data["iti"]["molmo"].update({
            "prompt_style": molmo_style
        })

        self.g.settings_data["iti"]["automa"]["negative_prompt"] = automa_neg
        self.g.settings_data["iti"]["sailing"].update({
            "sail_sampler": iti_sail_sampler,
            "sail_scheduler": iti_sail_scheduler,
            "sail_checkpoint": iti_sail_checkpoint,
            "sail_vae": iti_sail_vae,
            "pos_sail_lora": iti_pos_sail_lora,
            "pos_sail_embedding": iti_pos_sail_embedding,
            "neg_sail_lora": iti_neg_sail_lora,
            "neg_sail_embedding": iti_neg_sail_embedding,
            "enable_sailing_step": enable_sailing_step,
            "enable_sailing": enable_sailing,
            "sail_steps": sail_steps
        })
        settings_io().write_settings(self.g.settings_data)


    def stop_job(self):
        self.g.job_running = False

    def log_step(self, message):
        self.log += f"{message}\n"
        print(message)
        if self.g.settings_data["iti"]["output"]["log_file"]:
            with open(self.g.settings_data["iti"]["output"]["log_file"], "a", encoding="utf-8") as f:
                f.write(f"{message}\n")

    def cleanup_vram(self, stage=""):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.memory_allocated()
                torch.cuda.synchronize()
        self.log_step(f"VRAM after cleanup ({stage}): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def create_image_scores_db(self):
        """Create image_scores.db with the latest schema if it doesn’t exist."""

        with sqlite3.connect('image_scores.db') as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT,
                    image_path TEXT,
                    basename TEXT,
                    model TEXT,
                    pos_embedding TEXT,
                    neg_embedding TEXT,
                    pos_lora TEXT,
                    neg_lora TEXT,
                    sampler TEXT,
                    scheduler TEXT,
                    vae TEXT,
                    score INTEGER,
                    prompt TEXT,
                    iteration INTEGER,
                    topic TEXT,
                    filename TEXT,
                    mode TEXT,
                    scores_json TEXT,
                    penalties_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_basename ON image_scores(basename)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON image_scores(topic)")
            conn.commit()
        self.log_step("Ensured image_scores.db exists with updated schema")

    def delete_files(self, files):
        if not files:
            return "No files dropped."
        deleted = []
        for file in files:
            filename = file.name  # Full path from dropped file
            base_name = os.path.basename(filename)  # Just the filename
            with sqlite3.connect("image_scores.db") as db:
                cursor = db.execute("SELECT filename FROM image_scores WHERE filename LIKE ?", (f"%{base_name}",))
                row = cursor.fetchone()
                if row:
                    db_path = row[0]
                    db.execute("DELETE FROM image_scores WHERE filename = ?", (db_path,))
                    db.commit()
                    if os.path.exists(db_path):
                        os.remove(db_path)
                        deleted.append(f"Deleted {base_name} from DB and filesystem"), []
                    else:
                        deleted.append(f"Deleted {base_name} from DB; file not found"), []
                else:
                    if os.path.exists(db_path):
                        os.remove(db_path)
                        deleted.append(f"Deleted {base_name} from filesystem not found in DB"), []
                    else:
                        deleted.append(f"{base_name} not found in DB nor Filesystem"), []
        return "\n".join(deleted)

    def down_score_files(self, files):
        conn = sqlite3.connect("image_scores.db")  # Adjust DB path
        cursor = conn.cursor()
        updated = 0
        for file in files:
            filename = os.path.basename(file.name if hasattr(file, "name") else file)
            cursor.execute("SELECT filename FROM image_scores WHERE filename LIKE ?", (f"%{filename}",))
            row = cursor.fetchone()

            cursor.execute(
                "UPDATE image_scores SET score = 1 WHERE filename LIKE ?",
                (f"%{filename}",)
            )
            if cursor.rowcount > 0:  # Check if any rows were affected
                updated += cursor.rowcount
                if os.path.exists(row[0]):
                    os.remove(row[0])
            conn.commit()
        conn.close()
        return f"Updated {updated} image(s) to score 10!", []

    def update_file_list(self):
        with sqlite3.connect("image_scores.db") as db:
            rows = db.execute("SELECT filename, score FROM image_scores ORDER BY score DESC LIMIT 10").fetchall()
            if rows:
                return "## Top Files\n" + "\n".join(f"- {row[0]} (Score: {row[1]})" for row in rows)
            return "No files in DB yet."

    def get_research_topics(self):
        for n in range(3):
            self.g.settings_data["iti"]["research"]["prompts"][n][1] = self.get_prompt_topic(self.g.settings_data["iti"]["research"]["prompts"][n][0])
        settings_io().write_settings(self.g.settings_data)

        self.vision.unload_model()
        self.vision = None

    def learn_model_performance(self, input_folder=None, num_prompts=3):
        self.g.job_running = True
        if self.llm:
            self.llm.del_llm_model()
            self.llm = None
        """Run a learning phase to test model/LoRA/embedding/sampler/scheduler/vae combos."""
        self.log_step(f"Starting model performance learning with {num_prompts} prompts")

        # Load sailing arrays
        sailing_settings = self.g.settings_data["iti"]["sailing"]
        models = sailing_settings.get("sail_checkpoint", ["artcore_v10"])
        pos_loras = sailing_settings.get("pos_sail_lora", ["None"])
        neg_loras = sailing_settings.get("neg_sail_lora", ["None"])
        pos_embeddings = sailing_settings.get("pos_sail_embedding", ["None"])
        neg_embeddings = sailing_settings.get("neg_sail_embedding", ["None"])
        samplers = sailing_settings.get("sail_sampler", ["Euler a"])
        schedulers = sailing_settings.get("sail_scheduler", ["DPM++ 2M"])
        vaes = sailing_settings.get("sail_vae", ["default"])

        # Generate combos
        combos = list(itertools.product(models, pos_loras,neg_loras, pos_embeddings,neg_embeddings, samplers, schedulers, vaes))
        self.log_step(f"Generated {len(combos)} combos: {combos[:5]}...")

        # Get prompts from UI settings
        prompts = self.g.settings_data["iti"]["research"].get("prompts", [
            ["A majestic castle atop a misty hill at dawn.", "castle"],
            ["A bustling cyberpunk city under neon rain.", "cyberpunk"],
            ["A serene beach with golden sand at sunset.", "beach"]
        ])[:num_prompts]
        self.log_step(f"Using {len(prompts)} prompts from UI settings")

        total_images = len(combos) * len(prompts)
        self.log_step(f"Will generate {total_images} images across {len(models)} models")

        status_message = "learned from {image_counter} images how the models perform"
        image_counter = 0
        self.g.settings_data["automa"]["automa_seed"] = 1282321159
        for i, (prompt, topic) in enumerate(prompts):
            self.log_step(f"Prompt {i+1}: '{prompt}' (Topic: {topic})")

            for combo in combos:
                model, pos_lora, neg_lora, pos_embedding, neg_embedding, sampler, scheduler, vae = combo
                settings = self.g.settings_data.copy()
                settings["automa"]["automa_checkpoint"] = model


                if pos_lora != 'None':
                    settings["iti"]["prompt"]["pos_lora"] = f"<lora:{pos_lora}:0.8>"
                else:
                    settings["iti"]["prompt"]["pos_lora"] = ""

                if neg_lora != 'None':
                    settings["iti"]["prompt"]["neg_lora"] = f"<lora:{neg_lora}:0.8>"
                else:
                    settings["iti"]["prompt"]["neg_lora"] = ""
                if pos_embedding != 'None':
                    settings["iti"]["prompt"]["pos_style"] = pos_embedding
                else:
                    settings["iti"]["prompt"]["pos_style"] = ""
                if neg_embedding != 'None':
                    settings["iti"]["prompt"]["neg_style"] = neg_embedding
                else:
                    settings["iti"]["prompt"]["neg_style"] = ""

                settings["automa"]["sampler"] = sampler
                settings["automa"]["scheduler"] = scheduler
                settings["automa"]["vae"] = vae


                if not self.g.job_running:
                    return
                image_counter += 1
                for log, running, good in self.process_and_score_image(
                        None, prompt, None, 0, i, "learning", input_folder
                ):
                    score = self.actual_score["total"]
                    with sqlite3.connect("image_scores.db") as db:
                        db.execute("""
                                INSERT INTO image_scores (image_path, model, lora_combo, pos_lora, neg_lora, pos_embedding, neg_embedding, sampler, scheduler, vae, score, prompt, iteration, topic, filename, scores_json, penalties_json)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(model, lora_combo, pos_lora, neg_lora, pos_embedding, neg_embedding, sampler, scheduler, vae, prompt, topic)
                                DO UPDATE SET score = excluded.score, iteration = excluded.iteration
                                WHERE excluded.score > image_scores.score
                            """, (None, model, pos_lora, pos_lora, neg_lora, pos_embedding, neg_embedding, sampler, scheduler, vae, score, prompt, 0, topic, self.last_file, json.dumps(self.actual_score['scores']), json.dumps(self.actual_score['penalties'])))
                        db.commit()
                    self.log_step(f"Combo {combo} scored {score} (kept if highest)")
                    self.last_file = None
                    yield log, running, good, status_message.format(image_counter=image_counter)

        self.log_step(f"Learning complete—{total_images} images evaluated")

    def view_image_scores(self, sort_by="score", limit=100):
        """Return a raw table view of image_scores.db, sorted by a column."""
        with sqlite3.connect("image_scores.db") as db:
            query = f"SELECT image_path, model, lora_combo, embedding, sampler, scheduler, vae, score, prompt, iteration, topic FROM image_scores ORDER BY {sort_by} DESC LIMIT {limit}"
            rows = db.execute(query).fetchall()
            headers = ["image_path", "model", "lora_combo", "embedding", "sampler", "scheduler", "vae", "score", "prompt", "iteration", "topic"]
            self.log_step(f"Top {limit} scores sorted by {sort_by}:")
            self.log_step("\t".join(headers))
            for row in rows:
                self.log_step("\t".join(str(x) for x in row))
        return rows

    def scan_folder(self, folder_path):
        images = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    images.append(os.path.join(root, file))
        return images

    def ensure_spacy_model(self, model="en_core_web_sm"):
        """Check if spaCy model is installed; install it if missing."""
        try:
            return spacy.load(model)
        except OSError:
            print(f"Model '{model}' not found. Installing...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
            return spacy.load(model)

    def get_one_word_summary(self, text):
        nlp = self.ensure_spacy_model()
        doc = nlp(text)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        del nlp
        return Counter(nouns).most_common(1)[0][0] if nouns else "None"

    def get_prompt_topic(self, prompt):
        if not hasattr(self, 'vision') or self.vision is None:
            self.vision = molmo(self)
        if self.vision.model is None:
            self.vision.load_model()
        prompt = f"By reading this prompt: {prompt}. What’s the main theme? Output one short phrase: [theme]"
        prompt = self.vision.process_prompt(prompt).strip()
        return self.get_one_word_summary(prompt)

    def get_topic(self, image_path, prompt):
        if not hasattr(self, 'vision') or self.vision is None:
            self.vision = molmo(self)
        if self.vision.model is None:
            self.vision.load_model()
        # Cheap Molmo topic guess
        topic = self.vision.process_image(
            Image.open(image_path),
            "What’s the main theme? Output one short phrase: [theme]",
            ext_max_tokens=10
        ).strip("[]")

        self.vision.unload_model()
        self.vision = None
        self.cleanup_vram('Vram cleaned post get_topic')
        self.log_step(f"Guessed topic: {topic}")
        return self.get_one_word_summary(topic)

    def get_learned_combos(self, topic):

        print(f'topic type: {type(topic)} value: {topic}')

        with sqlite3.connect("image_scores.db") as db:
            cursor = db.cursor()
            # Get combos with enough runs
            cursor.execute("""
            SELECT model, pos_lora, neg_lora, pos_embedding, neg_embedding, 
                   sampler, scheduler, vae, AVG(score) as avg_score, COUNT(*) as run_count
            FROM image_scores
            WHERE topic = ? AND score IS NOT NULL
            GROUP BY model, pos_lora, neg_lora, pos_embedding, neg_embedding, 
                     sampler, scheduler, vae
            HAVING run_count >= 5
            ORDER BY avg_score DESC
        """, (str(topic),))
            results = cursor.fetchall()

        min_combos = 8  # Target: 8 unique combos (adjust as needed)
        learned_combos = [
            (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])  # model, pos_lora, neg_lora, pos_embedding, neg_embedding, sampler, scheduler, vae
            for r in results
        ]

        if len(learned_combos) >= min_combos:
            self.log_step(f"Learned {len(learned_combos)} combos: {learned_combos}")
            return learned_combos

        # Blend with defaults if insufficient
        default_models = self.g.settings_data['iti']["model_lists"]['models']
        default_loras = self.g.settings_data['iti']["model_lists"]['loras']
        sailing_settings = self.g.settings_data["iti"]["sailing"]
        default_pos_embeddings = sailing_settings.get("pos_sail_embedding", ["None"])
        default_neg_embeddings = sailing_settings.get("neg_sail_embedding", ["None"])
        default_samplers = sailing_settings.get("sail_sampler", ["Euler a"])
        default_schedulers = sailing_settings.get("sail_scheduler", ["DPM++ 2M"])
        default_vaes = sailing_settings.get("sail_vae", ["default"])

        default_combos = list(itertools.product(
            default_models[:4], default_loras[:2], ["None"],  # neg_lora default
            default_pos_embeddings, default_neg_embeddings, default_samplers, default_schedulers, default_vaes
        ))
        if results:
            combos = list(set(learned_combos + default_combos))[:min_combos]
            self.log_step(f"Partial learning ({len(learned_combos)} combos)—blending: {combos}")
            return combos

        self.log_step(f"No learned combos for {topic}—using defaults: {default_combos[:min_combos]}")
        return default_combos[:min_combos]



    def get_prompt(self, image_path):
        if not hasattr(self, 'vision') or self.vision is None:
            self.vision = molmo(self)
        if self.vision.model is None:
            self.vision.load_model()
        self.log_step(f"VRAM before Molmo load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        image = Image.open(image_path).convert("RGB")

        prompt_style = self.g.settings_data["iti"]["molmo"].get("prompt_style", "concise")
        prompt_map = {
            "concise": "Describe key elements and colors briefly for Stable Diffusion.",
            "vivid": "Describe this image in a detailed, vivid way for generating a similar image.",
            "custom": "[DESCRIBE] Generate a concise, comma-separated list of key objects, colors, actions, exact spatial positions, and a scene description from this image for a Stable Diffusion prompt, emphasizing precise context like 'in water' or 'on boat', only unique and relevant tangible details (no camera, meta, or photographic terms), no redundant repetitions, strictly comma-separated with no 'and', aim for 50+ distinct tokens of meaningful content, e.g., 'blue bikini woman laying in water, wooden boat nearby, tropical scene with clear waves'."
        }

        if prompt_style == "custom":
            prompt_instruction = self.g.settings_data["iti"]["molmo"].get("custom_prompt", prompt_map["custom"])
        else:
            prompt_instruction = prompt_map[prompt_style]

        if prompt_style not in prompt_map:
            prompt_instruction = prompt_map["concise"]

        description = self.vision.process_image(image, prompt_instruction, ext_max_tokens=self.g.settings_data["iti"]["molmo"].get("max_tokens", 100)).strip()
        self.vision.unload_model()
        self.vision = None
        self.cleanup_vram('Vram cleaned post get_prompt')
        self.log_step(f"Raw Molmo description: '{description}'")

        if prompt_style == "custom":
            prompt = self.clean_prompt(description.replace(" and ", ", "))
        else:
            prompt = description

        self.log_step(f"Processed prompt: '{prompt}'")
        return prompt

    def find_weakest_metric(self, result):
        """
        Identifies the weakest scoring aspect based on the image evaluation.
        Returns: (weakest_metric, prev_weak_score, current_weak_score, target_weak_score)
        """
        if "scores" not in result:
            return "color", 30, 30, 50  # Fallback: Assume color is weak

        scores = result["scores"]  # Dict of metric scores: {'color': 26, 'composition': 50, ...}

        # Find the lowest scoring metric
        weakest_metric = min(scores, key=scores.get)  # Get metric with lowest score
        prev_weak_score = scores[weakest_metric]  # Previous score
        current_weak_score = prev_weak_score  # Assume same if no improvement yet
        target_weak_score = prev_weak_score + 20  # Goal: improve it by 20 points

        return weakest_metric, prev_weak_score, current_weak_score, target_weak_score

    def clean_prompt(self, prompt):
        if not prompt:
            return None

        prompt = prompt.replace(" and ", ", ")

        # First pass: preserve phrases, remove fillers
        filler_words = {"a", "an", "the", "is", "are", "here", "this", "that"}
        phrases = [phrase.strip() for phrase in prompt.replace(".", ",").split(",")]
        cleaned_phrases = []
        for phrase in phrases:
            words = phrase.split()
            if len(words) > 1:
                cleaned = " ".join(word for word in words if word.lower() not in filler_words)
            else:
                cleaned = words[0] if words and words[0].lower() not in filler_words else ""
            if cleaned:
                cleaned_phrases.append(cleaned)

        # Second pass: stricter deduping (catch "model's X" variants)
        seen = set()
        unique_tokens = []
        banned_terms = {"camera", "photographer", "photography", "angle", "perspective", "focus", "shot", "capture", "artistry", "professional", "meta", "composition", "settings"}  # Kill meta noise
        for t in cleaned_phrases:
            norm_t = t.lower().strip()
            # Skip if it contains banned terms or is a repeat
            if not any(banned in norm_t for banned in banned_terms) and norm_t not in seen:
                seen.add(norm_t)
                unique_tokens.append(t)

        token_count = len(unique_tokens)
        self.log_step(f"Cleaned to {token_count} unique tokens")
        if token_count < 50:
            self.log_step(f"Under 50 tokens ({token_count}), keeping quality output")
        return ", ".join(unique_tokens)

    def vision_refine_prompt(self, result, image_path, original_image_path=None):
        # Initialize Molmo
        if not hasattr(self, 'vision') or self.vision is None:
            self.vision = molmo(self)
        if self.vision.model is None:
            self.vision.load_model()
        self.log_step(f"VRAM after Molmo load (refine): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Score the original image if not already done
        if not hasattr(self, 'original_scores') and original_image_path:
            scorer = ImageScorer()
            original_result = scorer.score_single(original_image_path)
            self.original_scores = original_result["scores"]
        original_scores = getattr(self, 'original_scores', {})

        # Parse current scores and compare to original
        scores = result.get("scores", {})
        feedback = []
        for metric, score in scores.items():
            score = int(score or 0)
            orig_score = int(original_scores.get(metric, 0) or 0)
            is_comparative = metric in ['clip', 'ssim', 'phash']

            if score < orig_score + 10:  # Behind or close
                if metric == 'clip':
                    feedback.append(f"add more stuff that fits the vibe{' tons' if score < orig_score else ''}")
                elif metric == 'ssim':
                    feedback.append(f"shake up the layout, move things around{' a lot' if score < orig_score else ''}")
                elif metric == 'color':
                    feedback.append(f"boost colors a bit{' more' if score < orig_score else ''}")  # Softer push
                elif metric == 'phash':
                    feedback.append(f"change the big shapes, make it fresh{' totally' if score < orig_score else ''}")
                elif metric == 'aesthetic':
                    feedback.append(f"make it prettier, add cool details{' loads' if score < orig_score else ''}")
                elif metric == 'detail':
                    feedback.append(f"add more little details{' tons' if score < orig_score else ''}")
            elif score > orig_score + 10:  # Improved
                if metric == 'clip':
                    feedback.append("nice start on the vibe, keep adding more stuff")
                elif metric == 'ssim':
                    feedback.append("good shake-up, tweak the layout more")
                elif metric == 'color':
                    feedback.append("colors are nicer, tweak them a bit more")  # Less aggressive
                elif metric == 'phash':
                    feedback.append("cool fresh shapes, mix it up a bit more")
                elif metric == 'aesthetic':
                    feedback.append("it’s prettier now, toss in more cool details")
                elif metric == 'detail':
                    feedback.append("details are sharper, pile on even more")

        # Default nudge
        feedback_str = ", ".join(feedback) + "." if feedback else "make it way better than the first picture with richer details and stronger vibes."

        # Prompt with improvement focus

        molmo_prompt = self.g.settings_data["iti"]["molmo"].get("refine_prompt", "refine as hard as you can {feedback_str}")
        molmo_prompt = molmo_prompt.format(
            feedback_str=feedback_str
        )

        self.log_step(f"Sending this prompt to Molmo:\n\n{molmo_prompt}")

        # Process image
        with Image.open(image_path) as img:
            refined_prompt = self.vision.process_image(img, molmo_prompt, ext_max_tokens=self.g.settings_data["iti"]["molmo"].get("refine_max_tokens", 200)).strip()

        # Cleanup

        self.vision.unload_model()
        self.vision = None

        self.cleanup_vram("post-Molmo-refine")
        refined_prompt = self.clean_prompt(refined_prompt)
        self.log_step(f"Molmo refined prompt: '{refined_prompt}'")
        return refined_prompt

    def cycle_models_and_loras(self, image_path, settings, effective_target, input_folder, best_prompt):
        best_score, best_image = 0, image_path
        topic = self.get_topic(image_path, best_prompt)
        topic_key = self.get_one_word_summary(topic)
        self.log_step(f"Using fixed topic: {topic} for this cycle, reduced to key: {topic_key}")

        min_trials = 5
        subset_size = self.g.settings_data["iti"]["refinement"]["combo_size"]
        flop_factor = 0.8

        sailing_settings = self.g.settings_data["iti"]["sailing"]
        models = sailing_settings.get("sail_checkpoint", ["artcore_v10"])
        pos_loras = sailing_settings.get("pos_sail_lora", ["None"])
        neg_loras = sailing_settings.get("neg_sail_lora", ["None"])
        pos_embeddings = sailing_settings.get("pos_sail_embedding", ["None"])
        neg_embeddings = sailing_settings.get("neg_sail_embedding", ["None"])
        samplers = sailing_settings.get("sail_sampler", ["Euler a"])
        schedulers = sailing_settings.get("sail_scheduler", ["DPM++ 2M"])
        vaes = sailing_settings.get("sail_vae", ["default"])

        all_combos = list(itertools.product(models, pos_loras, neg_loras, pos_embeddings, neg_embeddings, samplers, schedulers, vaes))
        self.log_step(f"Total possible combos: {len(all_combos)}")

        base_pos_lora = settings["iti"]["prompt"].get("pos_lora", "")
        combos_run = 0
        combo_scores = {}

        while combos_run < subset_size: #len(all_combos):
            if not self.g.job_running:
                return
            with sqlite3.connect("image_scores.db") as db:
                cursor = db.cursor()
                cursor.execute("""
                    SELECT MAX(avg_score) 
                    FROM (
                        SELECT AVG(score) as avg_score
                        FROM image_scores
                        WHERE topic = ? AND score IS NOT NULL
                        GROUP BY model, pos_lora, neg_lora, pos_embedding, neg_embedding, 
                                 sampler, scheduler, vae
                        HAVING COUNT(*) >= ?
                    )
                """, (topic_key, min_trials))
                top_score = cursor.fetchone()[0] or effective_target

                cursor.execute("""
                    SELECT model, pos_lora, neg_lora, pos_embedding, neg_embedding, 
                           sampler, scheduler, vae, AVG(score) as avg_score, COUNT(*) as run_count
                    FROM image_scores
                    WHERE topic = ? AND score IS NOT NULL
                    GROUP BY model, pos_lora, neg_lora, pos_embedding, neg_embedding, 
                             sampler, scheduler, vae
                    HAVING run_count >= ?
                """, (topic_key, min_trials))
                combo_stats = cursor.fetchall()

            top_threshold = top_score * flop_factor
            self.log_step(f"Top score: {top_score:.1f}, Flop threshold: {top_threshold:.1f}")

            good_combos = {(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in combo_stats if r[8] >= top_threshold}
            flop_combos = {(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in combo_stats if r[8] < top_threshold}

            learned_combos = self.get_learned_combos(topic_key)
            unexplored_combos = [c for c in all_combos if c not in good_combos and c not in flop_combos]
            random.shuffle(unexplored_combos)

            # Dynamic split: more planned early, more learned later
            explored_ratio = combos_run / len(all_combos)
            learned_count = min(int(subset_size * explored_ratio), subset_size - 2)  # At least 2 planned
            planned_count = subset_size - learned_count
            subset = learned_combos[:learned_count] + unexplored_combos[:planned_count]

            if not subset:
                self.log_step("No more combos to explore!")
                break

            self.log_step(f"Running subset of {len(subset)} combos (learned: {learned_count}, planned: {planned_count})")
            for i, combo in enumerate(subset):
                if combo in flop_combos:
                    self.log_step(f"Skipping known flop: {combo}")
                    continue

                model, pos_lora, neg_lora, pos_embedding, neg_embedding, sampler, scheduler, vae = combo
                combo_scores.setdefault(combo, [])

                settings["automa"]["automa_checkpoint"] = model
                settings["iti"]["prompt"]["pos_lora"] = (
                    f'{base_pos_lora},<lora:{pos_lora}:0.8>' if pos_lora and pos_lora != "None" else base_pos_lora
                )
                settings["iti"]["prompt"]["neg_lora"] = (
                    f'<lora:{neg_lora}:0.8>' if neg_lora and neg_lora != "None" else ""
                )

                if self.g.settings_data['iti']['enhance_prompt']:
                    best_prompt = self.prompt_enhancer.enhance_prompt(best_prompt)

                for log, running, good in self.process_and_score_image(
                        image_path, best_prompt, effective_target, i, 0, "regular", input_folder
                ):
                    result = self.actual_score
                    score = result["total"]
                    combo_scores[combo].append(score)
                    combos_run += 1


                    with sqlite3.connect("image_scores.db") as db:
                        cursor = db.cursor()
                        cursor.execute("""
                            INSERT INTO image_scores (
                                image_path, basename, model, pos_embedding, neg_embedding, 
                                pos_lora, neg_lora, sampler, scheduler, vae, score, prompt, 
                                iteration, topic, filename, mode, scores_json, penalties_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            self.last_file, os.path.basename(self.last_file), model,
                            pos_embedding, neg_embedding, pos_lora, neg_lora, sampler, scheduler, vae,
                            score, best_prompt, i, topic_key, self.last_file, "single",
                            json.dumps(self.actual_score['scores']), json.dumps(self.actual_score['penalties'])
                        ))
                        db.commit()


                    if score > best_score:
                        best_score = score
                        best_image = self.last_file
                        self.log_step(f"New best score: {best_score} with combo: {combo}")
                    else:
                        self.log_step(f"New score: {score} with combo: {combo}")

                    yield log, running, good

        settings["iti"]["prompt"]["pos_lora"] = base_pos_lora
        settings["iti"]["prompt"]["neg_lora"] = ""
        self.log_step(f"Cycle ended with {combos_run} combos run, best score: {best_score}")
        return best_image, best_score


    def get_image_size(self, file_path):
        """
        Returns the width and height of an image, scaled to ~1M pixels if larger,
        matching a common Stable Diffusion size close to 1M pixels with the closest aspect ratio, divisible by 8.

        Args:
            file_path (str): Path to the image file.

        Returns:
            tuple: (width, height) in pixels, or (1024, 1024) if the image can't be opened.
        """
        try:
            with Image.open(file_path) as img:
                orig_width, orig_height = img.size
                orig_ratio = orig_width / orig_height
                pixels = orig_width * orig_height

                # Common upscale factors
                upscale_factors = [3, 4, 2]  # Most common upscaling factors

                # Estimate the original size before upscaling
                estimated_sizes = []
                for factor in upscale_factors:
                    w, h = orig_width // factor, orig_height // factor
                    if w * h >= 900_000:  # Keep it close to 1M pixels
                        estimated_sizes.append((w, h))

                # Pick the best estimated size (prefer closest to 1M pixels)
                if estimated_sizes:
                    estimated_sizes.sort(key=lambda s: abs((s[0] * s[1]) - 1_000_000))
                    width, height = estimated_sizes[0]
                else:
                    width, height = orig_width, orig_height

                # Common Stable Diffusion sizes close to 1M pixels
                sd_sizes = [
                    (1024, 1024), (1152, 864), (1216, 912), (864, 1152), (912, 1216),
                    (1152, 768), (1216, 816), (768, 1152), (816, 1216),
                    (1280, 720), (1344, 760), (720, 1280), (760, 1344)
                ]

                # Find the closest SD size by aspect ratio and pixel count
                def size_score(size):
                    w, h = size
                    ratio_diff = abs((w / h) - orig_ratio)
                    pixel_diff = abs((w * h) - 1_000_000)
                    return ratio_diff * 1000 + pixel_diff  # Weight ratio more heavily

                best_sd_size = min(sd_sizes, key=size_score)
                final_width, final_height = best_sd_size

                # Ensure divisibility by 8
                final_width = (final_width // 8) * 8
                final_height = (final_height // 8) * 8

                return final_width, final_height

        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
            return 1024, 1024


    def load_model_lora_lists(self, input_folder):
        """
        Loads model and LoRA lists from model_list.txt and lora_list.txt in the input folder.
        Populates self.g.settings_data['iti']['models'] and ['iti']['loras'] if files exist.
        If files are missing, assumes defaults are already set.

        Args:
            input_folder (str): Root folder where model_list.txt and lora_list.txt are expected.
        """
        # Ensure 'iti' key exists in settings_data
        if 'iti' not in self.g.settings_data:
            self.g.settings_data['iti'] = {}

        # Paths to the files
        model_file = os.path.join(input_folder, "model_list.txt")
        lora_file = os.path.join(input_folder, "lora_list.txt")

        # Load models
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r', encoding='utf-8') as f:
                    models = [line.strip() for line in f if line.strip()]
                self.g.settings_data['iti']["model_lists"]['models'] = models
                self.log_step(f"Loaded {len(models)} models from {model_file}: {models}")
                settings_io().write_settings(self.g.settings_data)
            except Exception as e:
                self.log_step(f"Failed to load {model_file}: {str(e)}")

        # Load LoRAs
        if os.path.exists(lora_file):
            try:
                with open(lora_file, 'r', encoding='utf-8') as f:
                    loras = [line.strip() for line in f if line.strip()]
                self.g.settings_data['iti']["model_lists"]['loras'] = loras
                self.log_step(f"Loaded {len(loras)} LoRAs from {lora_file}: {loras}")
                settings_io().write_settings(self.g.settings_data)
            except Exception as e:
                self.log_step(f"Failed to load {lora_file}: {str(e)}")

        # If no files found, assume defaults are set elsewhere
        if not os.path.exists(model_file) and not os.path.exists(lora_file):
            self.log_step(f"No model_list.txt or lora_list.txt found in {input_folder}—assuming defaults are set.")

    def run_batch(self, input_folder):
        """Main flow: process input images with regular, sailing, and model/LoRA cycling."""
        self.g.job_running = True
        self.original_image = None
        if self.llm:
            self.llm.del_llm_model()
            self.llm = None

        self.load_model_lora_lists(input_folder)

        sailing_mode = self.g.settings_data["iti"]['sailing']['enable_sailing']
        sail_steps = self.g.settings_data["iti"]['sailing']['sail_steps']


        if sailing_mode:
            self.g.sail_history = []
        self.log = ""
        self.last6_good = deque(maxlen=6)
        self.last6_running = deque(maxlen=6)
        images = self.scan_folder(input_folder)
        self.log_step(f"Found {len(images)} images in {input_folder}")
        self.g.settings_data["automa"]["automa_seed"] = -1
        processed_count = 0
        for i, image_path in enumerate(images):
            if not self.g.job_running:
                return
            with torch.no_grad():

                self.g.settings_data['automa']['automa_width'], self.g.settings_data['automa']['automa_height'] = self.get_image_size(image_path)
                settings_io().write_settings(self.g.settings_data)
                self.original_image = image_path
                self.g.settings_data["Temperature"] = self.startup_temperature
                self.log_step(f"Processing {image_path} ({i+1}/{len(images)})")

                # Initial prompt and scoring setup
                prompt = self.get_prompt(image_path)

                if self.g.settings_data['iti']['enhance_prompt']:
                    prompt = self.prompt_enhancer.enhance_prompt(prompt)


                self.log_step(f"Generated prompt: '{prompt}'")
                self.scorer = ImageScorer()
                self.original_scores = self.scorer.score_single(image_path)
                self.log_step(f"Original score: {self.original_scores['total']}")
                self.scorer = None
                self.cleanup_vram("post-original-score")

                # Track best prompt and score
                best_score, best_prompt = self.original_scores["total"], prompt
                prompt_history = [(best_score, best_prompt)]

                # Regular generation
                target_score = self.g.settings_data["iti"]["refinement"].get("target_score", 80)
                improvement_threshold = self.g.settings_data["iti"]["refinement"].get("improvement_threshold", 20)
                effective_target = max(target_score, self.original_scores["total"] + improvement_threshold)
                self.log_step(f"Effective target for regular generation: {effective_target}")
                for log, last6_running, good in self.generate_regular_images(image_path, prompt, self.original_scores["total"], i, input_folder):
                    self.log = log
                    if good:
                        self.log_step(f"Got good paths from regular: {good}")
                    # Format last6_good as (path, caption) tuples
                    good_gallery = [(x[2], f"Score: {x[0]} (Iter {x[1]})") for x in self.last6_good]
                    yield self.log, list(self.last6_running), good_gallery, f"Processed {processed_count} images"
                    if "total" in self.actual_score and self.actual_score["total"] > best_score:
                        best_score, best_prompt = self.actual_score["total"], prompt


                # Sailing generation if enabled
                if sailing_mode:
                    for log, last6_running, good in self.generate_sailing_images(...):
                        self.log = log
                        if good:
                            self.log_step(f"Got good paths from sailing: {good}")
                        good_gallery = [(x[2], f"Score: {x[0]} (Iter {x[1]})") for x in self.last6_good]
                        yield self.log, list(self.last6_running), good_gallery, f"Processed {processed_count} images"
                        if self.actual_score["total"] > best_score:
                            best_score, best_prompt = self.actual_score["total"], prompt

                # Model/LoRA cycle if enabled
                if self.g.settings_data["iti"]["refinement"]["enable_model_cycle"]:

                    self.log_step(f"Best prompt for {image_path}: '{best_prompt}' (score: {best_score})")
                    effective_target = max(target_score, self.original_scores["total"] + improvement_threshold)
                    self.log_step(f"Effective target for cycle_models_and_loras: {effective_target}")
                    for log, last6_running, good in self.cycle_models_and_loras(image_path, self.g.settings_data, effective_target, input_folder, best_prompt):
                        self.log = log
                        if good:
                            self.log_step(f"Got good paths from cycle: {good}")
                        good_gallery = [(x[2], f"Score: {x[0]} (Iter {x[1]})") for x in self.last6_good]
                        yield self.log, list(self.last6_running), good_gallery, f"Processed {processed_count} images"

                processed_count += 1
                self.original_image = None
                self.original_scores = None
                good_gallery = [(x[2], f"Score: {x[0]} (Iter {x[1]})") for x in self.last6_good]
                yield self.log, list(self.last6_running), good_gallery, f"Processed {processed_count} images"
        good_gallery = [(x[2], f"Score: {x[0]} (Iter {x[1]})") for x in self.last6_good]
        return self.log, list(self.last6_running), good_gallery, f"Processed {processed_count} images"

    def generate_regular_images(self, image_path, initial_prompt, original_score, image_index, input_folder):
        """Generate and process regular images for max_iterations."""
        max_iterations = self.g.settings_data["iti"]["refinement"]["max_iterations"]
        self.log_step(f"Normal mode: {max_iterations} iterations")
        prompt = initial_prompt
        prev_prompt = prompt
        target_score = self.g.settings_data["iti"]["refinement"]["target_score"]
        improvement_threshold = self.g.settings_data["iti"]["refinement"].get("improvement_threshold", 20)
        effective_target = max(target_score, original_score + improvement_threshold)

        for iteration in range(max_iterations):
            if not self.g.job_running:
                return
            self.log_step(f"Running iteration: {iteration + 1} of {max_iterations} iterations")
            if self.g.settings_data['iti']["refinement"]['enhance_prompt']:
                prompt = self.prompt_enhancer.enhance_prompt(prompt)
            try:
                for log, last6_running, last6_good in self.process_and_score_image(
                        image_path, prompt, effective_target, iteration, image_index, "regular", input_folder
                ):
                    yield log, last6_running, last6_good
            except Exception as e:
                pass

            self.log_step(f"Refining Prompt for iteration {iteration + 1}: '{prompt}'")
            if iteration < max_iterations - 1:
                self.log_step(f"Refining prompt for iteration {iteration + 1}: '{prompt}'")
                prompt = self.vision_refine_prompt(result=self.actual_score, image_path=image_path)
                self.log_step(f"Refined prompt for iteration {iteration + 1}: '{prompt}'")
                prev_prompt = prompt

            if self.g.settings_data["iti"]["sailing"]["enable_sailing_step"]:
                try:
                    self.generate_sailing_image(image_path, prompt, original_score, effective_target, image_index, input_folder)
                except Exception as e:
                    pass

    def iti_set_sailing_vars(self, prompt):
        self.g.settings_data['sailing']['sail_text'] = prompt
        self.g.settings_data['sailing']['sail_neg_embed'] = self.g.settings_data["iti"]["prompt"]["neg_lora"]
        self.g.settings_data['sailing']['sail_add_neg'] = self.g.settings_data["iti"]["prompt"]["neg_style"]
        self.g.settings_data['sailing']['sail_neg_prompt'] = self.g.settings_data["iti"]["automa"]["negative_prompt"]
        self.g.settings_data['sailing']["sail_style"] = self.g.settings_data["iti"]["prompt"]["pos_style"]
        self.g.settings_data["sailing"]["sail_pos_embed"] = self.g.settings_data["iti"]["prompt"]["pos_lora"]

    def generate_sailing_images(self, image_path, initial_prompt, original_score, image_index, input_folder):

        """Generate and process extra sailing images for sail_steps."""
        sail_keep_text = False
        self.llm = llm_interface_qdrant.get_interface()
        sail_steps = self.g.settings_data["iti"]['sailing']['sail_steps'] or self.g.settings_data['sailing']['sail_width']
        self.g.settings_data['sailing']['sail_target'] = True
        self.g.settings_data['sailing']['sail_text'] = initial_prompt
        # [Other sailing settings unchanged]
        new_nodes = self.llm.direct_search(initial_prompt, self.g.settings_data['sailing']['sail_depth'], 0)
        query = self.sailing_manager.get_next_target_new(new_nodes)
        self.log_step(f"Sailing mode: {sail_steps} extra steps")

        for iteration in range(sail_steps):
            if not self.g.job_running:
                return
            if query != -1:
                if sail_keep_text:
                    self.iti_set_sailing_vars(query)
                else:
                    self.iti_set_sailing_vars(initial_prompt)

                prompt, _, _, _, _ = self.sailing_manager.get_new_prompt(
                    query, iteration + 1, 0, sail_steps, "sail_log.txt", sail_keep_text
                )
                new_nodes = self.llm.direct_search(prompt, self.g.settings_data['sailing']['sail_depth'], iteration + 1)
                query = self.sailing_manager.get_next_target_new(new_nodes)
                target_score = self.g.settings_data["iti"]["refinement"]["target_score"]
                improvement_threshold = self.g.settings_data["iti"]["refinement"].get("improvement_threshold", 20)
                effective_target = max(target_score, original_score + improvement_threshold)

                if self.g.settings_data["iti"]["sailing"]["sail_enhance_prompt"]:
                    prompt = self.prompt_enhancer.enhance_prompt(prompt)

                for log, last6_running, last6_good in self.process_and_score_image(
                        image_path, prompt, effective_target, iteration, image_index, "sailing", input_folder
                ):
                    yield log, last6_running, last6_good
            else:
                self.log_step(f"Sailing ended early at step {iteration + 1}")
                break

        self.llm.del_llm_model()
        self.cleanup_vram("post-sailing")

    def generate_sailing_image(self, image_path, initial_prompt, original_score, effective_target, image_index, input_folder):

        """Generate and process extra sailing images for sail_steps."""
        sail_keep_text = False
        self.llm = llm_interface_qdrant.get_interface()
        sail_steps = self.g.settings_data["iti"]['sailing']['sail_steps'] or self.g.settings_data['sailing']['sail_width']
        self.g.settings_data['sailing']['sail_target'] = True
        self.g.settings_data['sailing']['sail_text'] = initial_prompt
        # [Other sailing settings unchanged]
        new_nodes = self.llm.direct_search(initial_prompt, self.g.settings_data['sailing']['sail_depth'], 0)
        query = self.sailing_manager.get_next_target_new(new_nodes)
        self.log_step(f"Sailing mode: {sail_steps} extra steps")
        iteration = 1

        if query != -1:
            if sail_keep_text:
                self.iti_set_sailing_vars(query)
            else:
                self.iti_set_sailing_vars(initial_prompt)

            prompt, _, _, _, _ = self.sailing_manager.get_new_prompt(
                query, iteration + 1, 0, sail_steps, "sail_log.txt", sail_keep_text
            )

            if self.g.settings_data["iti"]["sailing"]["sail_enhance_prompt"]:
                prompt = self.prompt_enhancer.enhance_prompt(prompt)

            for log, last6_running, last6_good in self.process_and_score_image(
                    image_path, prompt,  effective_target, iteration, image_index, "sailing", input_folder
            ):
                yield log, last6_running, last6_good
        else:
            self.log_step(f"Sailing ended early at step {iteration + 1}")

        self.llm.del_llm_model()
        self.cleanup_vram("post-sailing")

    def process_and_score_image(self, image_path, prompt, effective_target, iteration, image_index, mode, base_folder=None):
        settings = self.g.settings_data
        self.log_step(f"raw prompt arrived for generating {prompt}")
        negative_prompt = settings["iti"]["automa"]["negative_prompt"]
        gen_prompt = f'{settings["iti"]["prompt"]["pos_style"]}, {prompt}, {settings["iti"]["prompt"]["pos_lora"]}'
        gen_neg_prompt = f'{settings["iti"]["prompt"]["neg_style"]}, {negative_prompt}, {settings["iti"]["prompt"]["neg_lora"]}'

        processor = shared.WildcardResolver()
        gen_prompt = processor.resolve_prompt(gen_prompt)
        gen_neg_prompt = processor.resolve_prompt(gen_neg_prompt)

        self.log_step(f"final prompt for generating {gen_prompt}")
        response = self.generator.request_generation(gen_prompt, gen_neg_prompt, settings_data=settings)
        self.generator.unload_checkpoint()
        self.cleanup_vram(f"post-{mode}-iter{iteration}")

        if response and 'images' in response:
            generated_image = Image.open(BytesIO(base64.b64decode(response['images'][0]))).convert('RGB')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = f"temp_{timestamp}_{image_index}_iter{iteration}_{mode}.png"
            generated_image.save(temp_path, format="PNG")
            settings_io().write_settings(self.g.settings_data)
            self.scorer = ImageScorer()
            compare = bool(image_path)  # False if image_path=None
            result = self.scorer.score(image_path, temp_path, compare=compare)
            self.log_step(f"Scores ({mode} iter {iteration}): {', '.join(f'{k}: {v}' for k, v in result['scores'].items())}, Total: {result['total']}")
            self.scorer = None
            self.actual_score = result
            total_score = result["total"]
            self.log_step(f"Scores ({mode} iter {iteration}): {result['scores']}, Total: {total_score}")

            output_dir = (settings["iti"]["output_dirs"]["good"] if effective_target and total_score >= effective_target
                          else settings["iti"]["output_dirs"]["drafts"])

            self.log_step(f"Output dir chosen: {output_dir}")


            # Handle image_path=None with a fallback name
            if image_path:
                rel_path = os.path.relpath(image_path, base_folder)
                file_name = os.path.splitext(os.path.basename(rel_path))[0]
                base_folder = base_folder or os.path.dirname(image_path)
                rel_path = os.path.relpath(image_path, base_folder)
                self.log_step(f"rel_path: {rel_path}")
                filename = os.path.splitext(os.path.basename(image_path))[0]
                base_save_path = os.path.join(output_dir, os.path.dirname(rel_path),
                                              f"{filename}_score_{total_score}_iter{iteration}_{mode}.png")

            else:
                base_save_path = os.path.join(output_dir, 'research',
                                              f"research_score_{total_score}_iter{iteration}_{mode}.png")

            os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
            save_path = base_save_path
            counter = 1
            while os.path.exists(save_path):
                save_path = os.path.splitext(base_save_path)[0] + f"_{counter}.png"
                counter += 1
            self.last_file = save_path

            shutil.move(temp_path, save_path)
            self.log_step(f"Saved to {save_path} ({mode} iter {iteration})")

            self.last6_running.append(save_path)
            if effective_target and total_score >= effective_target:
                self.log_step(f"Appending to last6_good: {(total_score, iteration, save_path)}")
                self.last6_good.append((total_score, iteration, save_path))  # Append to deque

            self.scorer = None
            self.cleanup_vram(f"post-scorer-{mode}-iter{iteration}")
            good_paths = [x[2] for x in self.last6_good if isinstance(x, tuple) and len(x) == 3]
            self.log_step(f"Yielding good paths: {good_paths}")
            yield self.log, list(self.last6_running), good_paths
        else:
            self.log_step(f"No images generated—API response empty ({mode} iter {iteration})")
            yield self.log, list(self.last6_running), []

