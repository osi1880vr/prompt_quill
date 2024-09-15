from datetime import datetime

from interrogate.moon import moon
import os
from PIL import Image
import re
from ollama import Client
import ollama
from tqdm import tqdm
from io import BytesIO
import base64

import globals
from llm_fw import llm_interface_qdrant

class OllamaUtil:
    def __init__(self):
        pass

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()

        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()

        image = Image.fromarray(image_np, mode='RGB')
        return image

    def image_to_bytes(self, image: Image.Image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()
        return image_bytes

    def image_to_base64(self, image: Image.Image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()
            image_base64 = base64.b64encode(image_bytes)
        return image_base64

    def load_image_to_base64(self, filename):
        # Load the image using PIL
        image = Image.open(filename)

        # Get the format of the image (e.g., 'PNG', 'JPEG')
        image_format = image.format

        # Create a BytesIO buffer to hold the image data in memory
        buffer = BytesIO()

        # Save the image to the buffer in your desired format (e.g., JPEG, PNG)
        image.save(buffer, format=image_format)  # You can change the format if needed

        # Get the byte data from the buffer
        image_bytes = buffer.getvalue()

        # Encode the byte data to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        return image_base64

    def pull_model(self, model, client):
        current_digest, bars = '', {}
        for progress in client.pull(model, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest


vision_models = ["llava:7b-v1.6-vicuna-q2_K (Q2_K, 3.2GB)",
                 "llava:7b-v1.6-mistral-q2_K (Q2_K, 3.3GB)",
                 "llava:7b-v1.6 (Q4_0, 4.7GB)",
                 "llava:13b-v1.6 (Q4_0, 8.0GB)",
                 "llava:34b-v1.6 (Q4_0, 20.0GB)",
                 "llava-llama3:8b (Q4_K_M, 5.5GB)",
                 "llava-phi3:3.8b (Q4_K_M, 2.9GB)",
                 "moondream:1.8b (Q4, 1.7GB)",
                 "moondream:1.8b-v2-q6_K (Q6, 2.1GB)",
                 "moondream:1.8b-v2-fp16 (F16, 3.7GB)"]


class OllamaImageDescriber:
    def __init__(self):
        self.g = globals.get_globals()
        self.interface = llm_interface_qdrant.get_interface()

    def get_story(self, description):


        client = Client(host=self.g.settings_data["story_teller_host"],
                        timeout=int(self.g.settings_data["story_teller_timeout"]))

        ollama_util = OllamaUtil()

        models = [model_l['name'] for model_l in client.list()['models']]

        model = self.g.settings_data["story_teller_model"]


        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)

        full_response = client.generate(model=model,
                                        system=self.g.settings_data["story_teller_system_context"],
                                        prompt=f'{self.g.settings_data["story_teller_prompt"]} {description}',
                                        keep_alive=-1,
                                        stream=False,
                                        options={
                                            'num_predict': int(self.g.settings_data["story_teller_max_tokens"]),
                                            'temperature': float(self.g.settings_data["story_teller_temperature"]),
                                            'top_k': 40,
                                            'top_p': 0.9,
                                            'repeat_penalty': 1.1,
                                            'seed': -1,
                                            'main_gpu': 0,
                                            'low_vram': True,
                                        })

        result = full_response['response']
        return result

    def write_story_file(self, story):
        single_line_text = re.sub(r'\s+', ' ', story)
        single_line_text = single_line_text.replace('\u2013', '')
        single_line_text = single_line_text.replace('\u2014', '')
        # Get today's date and format it as YYYY-MM-DD
        today_date = datetime.today().strftime('%Y-%m-%d')

        # Define the filename with today's date
        filename = f"story_text_{today_date}.txt"
        story_path = 'api_out/img2txt'
        filename = os.path.join(story_path,filename)

        # Open the file in append mode, create it if it doesn't exist
        with open(filename, 'a') as file:
            # Write the single line to the file and append a newline at the end
            file.write(single_line_text + '\n')

    def ollama_image_describe(self, image_name):
        client = Client(host=self.g.settings_data["story_teller_host"],
                        timeout=int(self.g.settings_data["story_teller_timeout"]))

        ollama_util = OllamaUtil()

        models = [model_l['name'] for model_l in client.list()['models']]

        model = self.g.settings_data["image_description_model"].split(' ')[0].strip()

        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)

        full_response = ""

        images_base64 = []

        img_base64 = ollama_util.load_image_to_base64(image_name)
        images_base64.append(img_base64)

        print('Generating Description from Image')
        full_response = client.generate(model=model,
                                        system=self.g.settings_data["image_description_system_context"],
                                        prompt=self.g.settings_data["image_description_prompt"],
                                        images=images_base64,
                                        keep_alive=-1,
                                        stream=False,
                                        options={
                                            'num_predict': int(self.g.settings_data["story_teller_max_tokens"]),
                                            'temperature': float(self.g.settings_data["story_teller_temperature"]),
                                            'top_k': 40,
                                            'top_p': 0.9,
                                            'repeat_penalty': 1.1,
                                            'seed': -1,
                                            'main_gpu': 0,
                                            'low_vram': True,
                                        })

        result = full_response['response']

        if self.g.settings_data["story_teller_seconds_step_enabled"]:
            result = self.get_story(result)
        self.write_story_file(result)
        print('Finalized')
        return result


class MoonFilenames:

    def __init__(self):
        self.g = globals.get_globals()
        self.moon_interrogate = moon()
        self.story_teller = OllamaImageDescriber()

    def fix_draft_filename(self, description):
        # Remove single characters (words with length 1)
        description = ' '.join(word for word in description.split() if len(word) > 1)

        # Replace spaces with underscores
        filename = description.replace(' ', '_')

        # Optional: Remove non-alphanumeric characters except underscores
        # This is to ensure the filename is valid on all filesystems
        filename = re.sub(r'[^\w_]', '', filename)

        return filename

    def get_filename(self, img):
        filename_prompt = 'generate a concise filename for the image that captures its core essence in the fewest words possible.'

        file_name_draft = self.moon_interrogate.run_interrogation(img, filename_prompt, 10)
        filename = self.fix_draft_filename(file_name_draft)
        return filename

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
                            if self.g.settings_data["story_teller_enabled"]:
                                story = self.story_teller.ollama_image_describe(new_file_path)
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
