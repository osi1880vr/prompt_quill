import json
import os
from settings.defaults import default
from settings.check_file_name import is_path_exists_or_creatable_portable
class settings_io:


    def __init__(self):
        self.default = default
        self.settings = self.default


    def get_defaults(self):
        return self.default

    def update_settings_with_defaults(self):
        self.settings['model_list'] = self.default['model_list']
        self.write_settings(self.settings)

    def check_missing_seettings(self):
        missing = 0
        for key in self.default.keys():
            if key not in self.settings:
                self.settings[key] = self.default[key]
                missing += 1
        if missing != 0:
            self.write_settings(self.settings)
        self.update_settings_with_defaults()

    def load_settings(self):
        if os.path.isfile('pq/settings/settings.dat'):
            f = open('pq/settings/settings.dat','r')
            self.settings = json.loads(f.read())
            f.close()
            self.check_missing_seettings()
        return self.settings


    def write_settings(self, settings):
        f = open('pq/settings/settings.dat','w')
        f.write(json.dumps(settings))
        f.close()


    def load_preset_list(self):
        filelist = os.listdir('pq/settings/presets')
        out_list = []
        for file in filelist:
            if file != 'presets_go_here.txt':
                out_list.append(file.replace('_preset.dat', ''))
        if len(out_list) > 0:
            return out_list
        else:
            return None

    def load_preset(self, name):
        if os.path.isfile('pq/settings/settings.dat'):
            f = open(f'pq/settings/presets/{name}_preset.dat','r')
            self.settings = json.loads(f.read())
            f.close()
            self.check_missing_seettings()
        return self.settings

    def save_preset(self, name, settings):
        filename = f'pq/settings/presets/{name}_preset.dat'
        check = is_path_exists_or_creatable_portable(filename)

        if check:
            f = open(filename,'w')
            f.write(json.dumps(settings))
            f.close()
            return 'OK'
        else:
            return 'Filename not OK'

    def load_prompt_data(self):
        if os.path.isfile('pq/settings/data.json'):
            f = open('pq/settings/data.json','r')
            self.prompt_iteration_raw = json.loads(f.read())
            f.close()
        return self.prompt_iteration_raw