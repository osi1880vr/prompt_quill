
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import torch
import random
from PIL import Image, ImageStat
import globals
import os
import numpy as np
from huggingface_hub import snapshot_download

class molmo:

    def __init__(self):
        self.g = globals.get_globals()
        self.arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
        self.molmo_model = 'cyan2k/molmo-7B-D-bnb-4bit'
        self.model_pat = None
        self.model = None
        

    def load_model(self):

        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model_path = snapshot_download(repo_id=self.molmo_model)
        self.processor = AutoProcessor.from_pretrained(self.model_path, **self.arguments)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, quantization_config=quant_config, **self.arguments)


    def preprocess_image(self, image):
        # 将 numpy 数组转换为 PIL Image

        image = np.array(image)

        pil_image = Image.fromarray(np.uint8(image[0] * 255)).convert('RGB')

        # 计算平均亮度
        gray_image = pil_image.convert('L')
        stat = ImageStat.Stat(gray_image)
        average_brightness = stat.mean[0]

        # 根据亮度定义背景颜色
        bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

        # 创建带有背景色的新图像
        new_image = Image.new('RGB', pil_image.size, bg_color)
        new_image.paste(pil_image, (0, 0), pil_image if pil_image.mode == 'RGBA' else None)

        return new_image


    def process_image(self, image, prompt):

        if self.model is None:
            self.load_model()

        processed_image = self.preprocess_image(image)

        inputs = self.processor.process(
            images=[image],
            text=prompt
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 确保 CuDNN 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        generation_config = GenerationConfig( max_new_tokens=self.g.settings_data['molmo_max_new_tokens'],
                                              stop_strings="<|endoftext|>",
                                              do_sample=True,
                                              temperature=self.g.settings_data['molmo_temperature'],
                                              top_k=self.g.settings_data['molmo_top_k'],
                                              top_p=self.g.settings_data['molmo_top_p'],)

        with torch.random.fork_rng(devices=[self.model.device]):
            torch.random.manual_seed(seed)

            output = self.model.generate_from_batch(
                inputs,
                generation_config,
                tokenizer=self.processor.tokenizer
            )

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if self.g.settings_data['molmo_unload_model_after_generation']:
            self.unload_model()
        return generated_text

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        print("Model and processor have been unloaded, and CUDA cache has been cleared.")

    def get_filename(self, img):
        return self.process_image(img, self.g.settings_data['molmo_file_renamer_prompt'])

    def story_teller(self, img):
        return self.process_image(img, self.g.settings_data['molmo_story_teller_prompt'])

    def process_folder(self, root_folder):
        process_count = 0
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                    file_path = os.path.join(dirpath, filename)
                    try:
                        # Open the image file using Pillow
                        with Image.open(file_path) as img:

                            #raw_data = np.array(img)

                            new_file_name = self.get_filename(img)
                            base_name, extension = os.path.splitext(filename)
                            new_filename = f"{new_file_name}{extension}"
                            new_file_path = os.path.join(dirpath, new_filename)
                            counter = 1
                            while os.path.exists(new_file_path):
                                new_filename_numbered = f"{new_file_name}_{counter}{extension}"
                                new_file_path = os.path.join(dirpath, new_filename_numbered)
                                counter += 1

                            # Rename the file to the new unique name
                            os.rename(file_path, new_file_path)
                            if self.g.settings_data["molmo_story_teller_enabled"]:
                                story = self.story_teller(img)
                                # Split the file name and extension
                                name, ext = os.path.splitext(new_file_path)

                                # Change the extension to '.txt'
                                new_filename = name + '.txt'
                                with open(new_filename, 'w') as file:
                                    file.write(story)

                            process_count += 1

                    except Exception as e:
                        print(f"Error processing image {file_path}: {e}")
                if self.g.job_running is False:
                    break

        return process_count




