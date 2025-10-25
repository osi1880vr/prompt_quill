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
        """
        Modern llama-cpp-python wheels expose all CUDA/CPU backends through the
        canonical `llama_cpp` module. The legacy `llama_cpp_cuda*` module names
        only existed in older oobabooga wheels, so attempting to import them
        today just produces noisy “running CPU only” messages even when GPU
        kernels are available. By importing `llama_cpp` directly we let the
        installed wheel decide which backend to use (CUDA if present, CPU
        otherwise) without spurious warnings.
        """
        try:
            import llama_cpp as hijacked_llama
            print("llama_cpp_hijack: loaded llama_cpp (GPU-enabled if wheel supports it)")
        except Exception as exc:
            raise RuntimeError("Failed to import llama_cpp; is llama-cpp-python installed?") from exc

        # Replace `llama_cpp` globally
        sys.modules["llama_cpp"] = hijacked_llama
