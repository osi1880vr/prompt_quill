# interrogation_manager.py
import globals
from settings.io import settings_io
from llm_fw import llm_interface_qdrant
from interrogate.molmo import molmo
from PIL import Image
import os
from collections import deque
from ui.sailing_manager import SailingManager  # New import


class InterrogationManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.settings_io = settings_io()
        self.interface = llm_interface_qdrant.get_interface()
        self.molmo = molmo(self)  # Matches ui_actions instantiation
        self.sailing = SailingManager()
        self.sail_log = ''
        self.images_done = 0

    def png_info_get(self, image):
        return self.interface.get_png_info(image)

    def set_molmo(self, molmo_folder_name, molmo_file_renamer_prompt, molmo_story_teller_enabled, molmo_story_teller_prompt,
                  molmo_temperatur, molmo_max_new_tokens, molmo_top_k, molmo_top_p, molmo_source_folder_name,
                  molmo_destination_folder_name, molmo_organize_prompt):
        self.g.settings_data['interrogate']['molmo_folder_name'] = molmo_folder_name
        self.g.settings_data['interrogate']['molmo_file_renamer_prompt'] = molmo_file_renamer_prompt
        self.g.settings_data['interrogate']['molmo_story_teller_enabled'] = molmo_story_teller_enabled
        self.g.settings_data['interrogate']['molmo_story_teller_prompt'] = molmo_story_teller_prompt
        self.g.settings_data['interrogate']['molmo_temperatur'] = molmo_temperatur
        self.g.settings_data['interrogate']['molmo_max_new_tokens'] = molmo_max_new_tokens
        self.g.settings_data['interrogate']['molmo_top_k'] = molmo_top_k
        self.g.settings_data['interrogate']['molmo_top_p'] = molmo_top_p
        self.g.settings_data['interrogate']['molmo_source_folder_name'] = molmo_source_folder_name
        self.g.settings_data['interrogate']['molmo_destination_folder_name'] = molmo_destination_folder_name
        self.g.settings_data['interrogate']['molmo_organize_prompt'] = molmo_organize_prompt
        self.settings_io.write_settings(self.g.settings_data)

    def set_iti(self, iti_folder_name, iti_file_renamer_prompt):
        self.g.settings_data['interrogate']['iti_folder_name'] = iti_folder_name
        self.g.settings_data['interrogate']['iti_file_renamer_prompt'] = iti_file_renamer_prompt
        self.settings_io.write_settings(self.g.settings_data)

    def molmo_file_rename(self, folder):
        self.interface.del_llm_model()
        self.g.job_running = True
        count = self.molmo.process_folder(folder)
        return count

    def molmo_file_rename_stop(self):
        self.g.running = False
        return "Stopped"

    def molmo_organize(self, source_folder_name, destination_folder_name):
        return self.molmo.organize_files(source_folder_name, destination_folder_name)

    def run_iti(self, root_folder):
        self.g.job_running = True
        process_count = 0
        images = deque(maxlen=int(self.g.settings_data['sailing']['sail_max_gallery_size']))

        #count = self.molmo.iti.process_folder(file_path)
        self.interface.del_llm_model()
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                with Image.open(file_path) as img:

                    retry = True
                    retry_count = 0

                    while retry:
                        try:
                            # Generate a new filename
                            print(f'Processing {file_path}')
                            prompt = self.molmo.iti.get_image_prompt(img)  # Assuming get_filename generates a unique base name
                            print(f'Prompt: {prompt}')
                            self.sail_log = f'{prompt}\n\n{self.sail_log}'

                            if self.g.settings_data['sailing']['sail_generate']:
                                images = self.sailing.run_sail_automa_gen(prompt, images)
                                if len(images) > 0:

                                    yield self.sail_log, list(images),f'{self.images_done} prompts(s) done'
                                    retry = False
                                    break
                                else:
                                    yield self.sail_log, [], f'{self.images_done} prompts(s) done'
                            else:
                                yield self.sail_log, [], f'{self.images_done} prompts(s) done'
                            retry = False
                            break

                        except Exception as e:
                            # Retry the entire renaming process if an error occurs
                            self.molmo.increase_temperature()
                            retry = True  # This ensures the loop continues until renaming succeeds
                        finally:
                            retry_count += 1
                            if retry_count > 5 or not retry:
                                break

                    process_count += 1

        self.molmo.unload_model()
