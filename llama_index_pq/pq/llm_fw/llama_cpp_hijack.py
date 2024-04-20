import sys

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None

try:
    import llama_cpp_cuda_tensorcores
except:
    llama_cpp_cuda_tensorcores = None



class llama_cpp_hijack:
    def __init__(self):
        if llama_cpp_cuda_tensorcores is not None:
            sys.modules['llama_cpp'] = llama_cpp_cuda_tensorcores
        elif llama_cpp_cuda is not None:
            sys.modules['llama_cpp'] = llama_cpp_cuda



