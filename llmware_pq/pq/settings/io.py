import json
import os
from settings.defaults import default

class settings_io:
    def __init__(self):
        self.default = default
        self.settings = self.default



    def load_settings(self):
        if os.path.isfile('pq/settings/settings.dat'):
            f = open('pq/settings/settings.dat','r')
            self.settings = json.loads(f.read())
            f.close()
        return self.settings


    def write_settings(self, settings):
        f = open('pq/settings/settings.dat','w')
        f.write(json.dumps(settings))
        f.close()




