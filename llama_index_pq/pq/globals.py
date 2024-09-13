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

import threading

class _globals_store:
    globals_store = None
    sail_running = False


    def __init__(self):
        if _globals_store.globals_store == None:
            _globals_store.globals_store = self


def get_globals():
    if _globals_store.globals_store == None:
        with threading.Lock():
            if _globals_store.globals_store == None:
                _globals_store()
    return _globals_store.globals_store
