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

import sys
from pydantic import BaseModel
BaseModel.model_config['protected_namespaces'] = ()

class llama_cpp_hijack:
    def __init__(self):

        # Try to import different versions of llama_cpp
        try:
            import llama_cpp_cuda_tensorcores as hijacked_llama
        except Exception as e:
            print(f'no tensor cores due to {e}')
            try:
                import llama_cpp_cuda as hijacked_llama
            except Exception as e:
                print(f'no cudo cores due to {e}')
                print(f'running CPU only')
                import llama_cpp as hijacked_llama# Default to CPU version if no GPU versions exist

        # Replace `llama_cpp` globally
        sys.modules["llama_cpp"] = hijacked_llama



