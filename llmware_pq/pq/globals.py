import threading

class _globals_store:
    globals_store = None

    def __init__(self):
        if _globals_store.globals_store == None:
            _globals_store.globals_store = self


def get_globals():
    if _globals_store.globals_store == None:
        with threading.Lock():
            if _globals_store.globals_store == None:
                _globals_store()
    return _globals_store.globals_store