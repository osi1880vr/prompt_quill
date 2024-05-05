import time
import datetime

def timestamp():
	return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

