import shutil

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import torch
import random
from PIL import Image, ImageStat
import globals
import os
import numpy as np
from huggingface_hub import snapshot_download
import re
from io import BytesIO
from collections import deque
import base64
from string import Template

class i2i:

    def __init__(self, parent):
        self.parent = parent
        self.images_done = 0
        self.sail_log = ''
        pass



    def clean_prompt(self, prompt):
        prompt = prompt.replace('\n', ' ')

        return prompt


    def get_image_prompt(self, img):

        prompt = self.parent.process_image(img, self.parent.g.settings_data['interrogate']['iti_description_prompt']).strip()

        prompt = self.clean_prompt(prompt)
        return prompt

    def process_folder(self,root_folder):
        process_count = 0
        self.images_done = 0
        self.sail_log = ''
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                    file_path = os.path.join(dirpath, filename)
                    self.parent.reset_temperature()
                    try:
                        # Open the image file using Pillow
                        with Image.open(file_path) as img:

                            #raw_data = np.array(img)

                            retry = True
                            retry_count = 0
                            images = deque(maxlen=6)
                            while retry:
                                try:
                                    # Generate a new filename
                                    print(f'Processing {file_path}')
                                    prompt = self.get_image_prompt(img)  # Assuming get_filename generates a unique base name
                                    print(f'Prompt: {prompt}')
                                    self.sail_log = f'{self.sail_log}\n'


                                    if self.parent.g.settings_data['sailing']['sail_generate']:
                                        response = self.parent.run_sail_automa_gen(prompt)
                                        if response != '':
                                            for index, image in enumerate(response.get('images')):
                                                img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                                                save_path = os.path.join(self.parent.out_dir_t2i, f'txt2img-{self.parent.timestamp()}-{index}.png')
                                                self.parent.automa_client.decode_and_save_base64(image, save_path)
                                                self.images_done += 1
                                                yield self.sail_log, list(images),f'{self.images_done} image(s)'
                                        else:
                                            yield self.sail_log, [] , None
                                    else:
                                        yield self.sail_log, [], None

                                        yield self.sail_log,[],f'{self.images_done} prompts(s) done'
                                    retry = False



                                except Exception as e:
                                    # Retry the entire renaming process if an error occurs
                                    self.parent.increase_temperature()
                                    retry = True  # This ensures the loop continues until renaming succeeds
                                finally:
                                    retry_count += 1
                                    if retry_count > 5:
                                        break


                            process_count += 1

                    except Exception as e:
                        print(f"Error processing image {file_path}: {e}")
                if self.parent.g.job_running is False:
                    break

        return process_count

class molmo:

    def __init__(self, parent):
        self.g = globals.get_globals()
        self.parent = parent
        self.arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
        self.molmo_model = 'cyan2k/molmo-7B-D-bnb-4bit'
        self.model_pat = None
        self.model = None
        self.temperature = None
        self.reset_temperature()
        self.iti = i2i(self)
        self.processor = None


    def reset_temperature(self):
        self.temperature = self.g.settings_data['interrogate']['molmo_temperature']


    def increase_temperature(self):
        self.temperature += 0.1

    def load_model(self):
        self.model_path = snapshot_download(repo_id=self.molmo_model)
        self.processor = AutoProcessor.from_pretrained(self.model_path, **self.arguments)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.arguments)


    def preprocess_image(self, image):
        # Ensure the image is in the right format (1024x1024, 3 channels)
        pil_image = None
        if isinstance(image, Image.Image):
            # Convert directly if it's already a PIL image
            pil_image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            # If it's a numpy array, ensure correct shape and convert
            if image.ndim == 4:
                image = image[0]  # Remove batch dimension if present
            pil_image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
        else:
            raise ValueError("Unsupported image format. Expected a PIL image or numpy array.")


        # Calculate average brightness
        gray_image = pil_image.convert('L')
        stat = ImageStat.Stat(gray_image)
        average_brightness = stat.mean[0]

        # Define background color based on brightness
        bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

        # Create a new image with the defined background color
        new_image = Image.new('RGB', pil_image.size, bg_color)
        new_image.paste(pil_image, (0, 0))

        resized_image = new_image.resize((int(image.width // 3), int(image.height // 3)), Image.Resampling.LANCZOS)


        return resized_image


    def process_image(self, image, prompt):

        if self.model is None:
            self.load_model()

        processed_image = self.preprocess_image(image)

        inputs = self.processor.process(
            images=[processed_image],
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

        generation_config = GenerationConfig( max_new_tokens=self.g.settings_data['interrogate']['molmo_max_new_tokens'],
                                              stop_strings="<|endoftext|>",
                                              do_sample=True,
                                              temperature=self.temperature,
                                              top_k=self.g.settings_data['interrogate']['molmo_top_k'],
                                              top_p=self.g.settings_data['interrogate']['molmo_top_p'],
                                              )


        with torch.random.fork_rng(devices=[self.model.device]):
            torch.random.manual_seed(seed)

            output = self.model.generate_from_batch(
                inputs,
                generation_config,
                tokenizer=self.processor.tokenizer
            )

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if self.g.settings_data['interrogate']['molmo_unload_model_after_generation']:
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



    def clean_filename(self, filename, max_length=255):
        # Remove non-ASCII characters
        filename = re.sub(r'[^\x00-\x7F]+', '', filename)

        # Remove illegal characters for filenames on most systems, including single quote
        filename = re.sub(r"[\/:*?\"'<>|]", '', filename)

        # Replace spaces with underscores (optional, but often useful)
        filename = filename.replace(" ", "_")

        # Remove non-alphanumeric characters from the beginning of the filename
        filename = re.sub(r"^[^a-zA-Z0-9]+", '', filename)


        # Replace multiple underscores or hyphens with a single underscore
        filename = re.sub(r"[-_]+", '_', filename)



        # Truncate the filename if it's too long
        if len(filename) > max_length:
            filename = filename[:max_length]

        # Ensure the filename is not empty
        if not filename:
            filename = "default_filename"

        return filename


    def get_filename(self, img):
        filename = self.process_image(img, self.g.settings_data['interrogate']['molmo_file_renamer_prompt']).strip()
        filename, ext = os.path.splitext(filename)
        filename = self.clean_filename(filename)
        return filename


    def story_teller(self, img):
        story = self.process_image(img, self.g.settings_data['interrogate']['molmo_story_teller_prompt'])
        story = re.sub(r'[^\x00-\x7F]', '', story)
        return story

    def process_folder(self, root_folder):
        process_count = 0
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                    file_path = os.path.join(dirpath, filename)
                    self.reset_temperature()
                    try:
                        # Open the image file using Pillow
                        with Image.open(file_path) as img:

                            #raw_data = np.array(img)

                            retry = True
                            retry_count = 0
                            while retry:
                                try:
                                    # Generate a new filename
                                    print(f'Processing {file_path}')
                                    new_file_name = self.get_filename(img)  # Assuming get_filename generates a unique base name
                                    print(f'new filename: {new_file_name}')
                                    base_name, extension = os.path.splitext(filename)
                                    if retry_count > 5:
                                        new_file_name = 'broken_filename_please_change_manual'

                                    new_filename = f"{new_file_name}{extension}"
                                    new_file_path = os.path.join(dirpath, new_filename)
                                    counter = 1

                                    # Ensure the filename is unique within the folder
                                    while os.path.exists(new_file_path):
                                        new_filename_numbered = f"{new_file_name}_{counter}{extension}"
                                        new_file_path = os.path.join(dirpath, new_filename_numbered)
                                        counter += 1

                                    # Attempt to rename the file

                                    os.rename(file_path, new_file_path)

                                    # If rename succeeds, exit the loop
                                    retry = False


                                except Exception as e:
                                    # Retry the entire renaming process if an error occurs
                                    self.increase_temperature()
                                    retry = True  # This ensures the loop continues until renaming succeeds
                                finally:
                                    retry_count += 1
                                    if retry_count > 5:
                                        break
                            if self.g.settings_data['interrogate']["molmo_story_teller_enabled"]:
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

    def classify_image_molmo_caption(self, image, categories):
        """Classifies an image using Molmo captioning and keyword matching."""

        prompt = f"Based on the body, the cultural origin of the person, the skin color or any other typical feature you see. if you have to put the image into a single category of out of this list [{categories}] where would you put this. You are creating a technical output so your answer is only one single category." # we dont need a prompt here for captioning

        template = Template(self.g.settings_data['interrogate']["molmo_organize_prompt"])
        template.substitute(categories=categories)
        prompt = template.substitute(categories=categories) #self.g.settings_data['interrogate']["molmo_story_teller_enabled"].format(categories=categories) #f"Here's an image. Choose the best category from the following: [{categories}]. Output the most suitable category and explain why."


        try:
            caption = self.process_image(image, prompt) # use your existing process_image function
        except Exception as e:
            print(f"Error during image captioning: {e}")
            return "Error"

        best_match = None
        highest_score = 0

        for category in categories:
            score = caption.lower().count(category.lower())
            if score > highest_score:
                highest_score = score
                best_match = category

        if best_match is None or best_match == "":
            best_match = "Uncategorized"

        return best_match

    def resize_image(self, image_path, max_size=224):
        """Resizes an image while maintaining aspect ratio."""
        try:
            img = Image.open(image_path)
            img.thumbnail((max_size, max_size))  # Resize while maintaining aspect ratio
            return img
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
            return None


    def get_categories(self, output_directory):
        """Gets category names from subfolders, excluding those with numbers at the end."""
        categories = []
        for item in os.listdir(output_directory):
            item_path = os.path.join(output_directory, item)
            if os.path.isdir(item_path) and not re.search(r"\d+$", item):  # Check for directory and no trailing numbers
                categories.append(item)
        return categories

    def find_unique_filename(self, destination_path, filename, extension):
        """Finds a unique filename by appending a number if necessary."""
        base_name, _ = os.path.splitext(filename)
        new_filename = filename
        counter = 1
        while os.path.exists(os.path.join(destination_path, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        return new_filename



    def organize_images(self, image_directory, output_directory):
        """Organizes images using Molmo captioning."""

        categories = self.get_categories(output_directory)

        for category in categories:
            category_path = os.path.join(output_directory, category)
            if not os.path.exists(category_path):
                os.makedirs(category_path)
        if not os.path.exists(os.path.join(output_directory, "Uncategorized")):
            os.makedirs(os.path.join(output_directory, "Uncategorized"))
        n = 0
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(image_directory, filename)
                try:
                    resized_image = self.resize_image(image_path=image_path, max_size=512)
                    if resized_image is None:
                        print(f"Skipping file {filename}")
                        continue
                    category = self.classify_image_molmo_caption(resized_image, categories)
                    if category == "Error":
                        print(f"Skipping file {filename}")
                        continue
                    continue
                    new_filename = self.find_unique_filename(os.path.join(output_directory, category), filename, os.path.splitext(filename)[1])
                    destination_path = os.path.join(output_directory, category, new_filename)
                    shutil.move(image_path, str(destination_path))
                    print(f"Moved {filename} to {category} as {new_filename}")

                    # Handle associated .txt file (using the SAME new filename)
                    txt_filename = os.path.splitext(filename)[0] + ".txt"
                    txt_path = os.path.join(image_directory, txt_filename)
                    if os.path.exists(txt_path):
                        txt_destination_path = os.path.join(output_directory, category, os.path.splitext(new_filename)[0] + ".txt") # use the same base name with .txt extension
                        shutil.move(txt_path, str(txt_destination_path))
                        print(f"Moved associated txt file {txt_filename} to {category} as {os.path.splitext(new_filename)[0] + '.txt'}")

                    n += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


        return n

