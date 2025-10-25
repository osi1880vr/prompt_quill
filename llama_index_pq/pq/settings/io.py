import json
import os
import copy
from pathlib import Path

from .defaults import default
from .check_file_name import is_path_exists_or_creatable_portable
class settings_io:


    def __init__(self):
        self.default = default
        self.settings = self.default
        self.settings_dir = Path(__file__).resolve().parent


    def get_defaults(self):
        return self.default

    def update_settings_with_defaults(self):
        self.settings['model_list'] = self.default['model_list']
        self.settings["model_test"]['model_test_dimensions_list'] = self.default["model_test"]['model_test_dimensions_list']
        self.write_settings(self.settings)


    def auto_move_sub_objects(self):
        sub_objects = ["horde", "automa", "sailing", "model_test", "interrogate"]
        for sub_object in sub_objects:
            if sub_object not in self.settings:
                self.settings[sub_object] = {}
                for key in self.default[sub_object]:
                    self.settings[sub_object][key] = self.settings[key]
                    del self.settings[key]


    def cleanup_settings(self):
        settings_keys = copy.deepcopy(self.settings)
        for key in settings_keys:
            if key not in self.default:
                del self.settings[key]
            else:
                if type(self.settings[key]) == dict:
                    sub_keys = copy.deepcopy(self.settings[key])
                    for sub_key in sub_keys:
                        if sub_key not in self.default[key]:
                            del self.settings[key][sub_key]

    def check_missing_settings(self):
        def recursive_update(defaults, settings):
            missing = 0
            for key, value in defaults.items():
                if key not in settings:
                    settings[key] = value
                    missing += 1
                elif isinstance(value, dict) and isinstance(settings[key], dict):
                    missing += recursive_update(value, settings[key])
            return missing

        self.auto_move_sub_objects()
        missing = recursive_update(self.default, self.settings)
        self.cleanup_settings()

        if missing:
            self.write_settings(self.settings)

        self.update_settings_with_defaults()

    def load_settings(self):
        settings_file = self.settings_dir / 'settings.dat'
        if settings_file.is_file():
            with settings_file.open('r') as f:
                self.settings = json.loads(f.read())
            self.check_missing_settings()
        return self.settings


    def write_settings(self, settings):
        settings_file = self.settings_dir / 'settings.dat'
        with settings_file.open('w') as f:
            f.write(json.dumps(settings,indent=4))


    def load_preset_list(self):
        presets_dir = self.settings_dir / 'presets'
        if not presets_dir.exists():
            return None
        filelist = os.listdir(presets_dir)
        out_list = []
        for file in filelist:
            if file != 'presets_go_here.txt':
                out_list.append(file.replace('_preset.dat', ''))
        if len(out_list) > 0:
            return out_list
        else:
            return None

    def load_preset(self, name):
        preset_file = self.settings_dir / 'presets' / f'{name}_preset.dat'
        if preset_file.is_file():
            with preset_file.open('r') as f:
                self.settings = json.loads(f.read())
            self.check_missing_seettings()
        return self.settings

    def save_preset(self, name, settings):
        filename = self.settings_dir / 'presets' / f'{name}_preset.dat'
        check = is_path_exists_or_creatable_portable(str(filename))

        if check:
            with filename.open('w') as f:
                f.write(json.dumps(settings))
            return 'OK'
        else:
            return 'Filename not OK'

    def load_prompt_data(self):
        data_file = self.settings_dir / 'data.json'
        if data_file.is_file():
            with data_file.open('r') as f:
                self.prompt_iteration_raw = json.loads(f.read())
        else:
            self.prompt_iteration_raw = {}
        return self.prompt_iteration_raw
