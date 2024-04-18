import json
import os
from settings.defaults import default

class settings_io:


    def __init__(self):
        self.default = default
        self.settings = self.default

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
            self.check_missing_seettings()
            f.close()
        return self.settings


    def write_settings(self, settings):
        f = open('pq/settings/settings.dat','w')
        f.write(json.dumps(settings))
        f.close()




