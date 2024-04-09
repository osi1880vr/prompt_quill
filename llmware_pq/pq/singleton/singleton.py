import threading
from pq.settings.settings_io import settings_io

class settings:
	def __init__(self):
		self.settings_data = settings_io().load_settings()



class Singleton:
	_instance = None
	_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			with cls._lock:
				# another thread could have created the instance
				# before we acquired the lock. So check that the
				# instance is still nonexistent.
				if not cls._instance:
					cls._instance = super(Singleton, cls).__new__(cls)
		return cls._instance
