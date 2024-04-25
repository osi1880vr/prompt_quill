# Copyright 2023 osiworx

# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

import json
import os
from settings.defaults import default

class settings_io:


    def __init__(self):
        self.default = default
        self.settings = self.default


    def check_missing_seettings(self):
        missing = 0
        for key in self.default.keys():
            if key not in self.settings:
                self.settings[key] = self.default[key]
                missing += 1

        if missing != 0:
            self.write_settings(self.settings)

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




