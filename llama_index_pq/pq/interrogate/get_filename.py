from interrogate.moon import moon
import os
from PIL import Image
import re

class MoonFilenames:


    def __init__(self):
        self.moon_interrogate = moon()



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
                                new_filename = f"{base_name}_processed_{counter}{extension}"
                                new_file_path = os.path.join(dirpath, new_filename)
                                counter += 1

                            # Rename the file to the new unique name
                            os.rename(file_path, new_file_path)
                            process_count += 1

                    except Exception as e:
                        print(f"Error processing image {file_path}: {e}")


        return process_count


