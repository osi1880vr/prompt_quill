import datetime
import threading
from types import SimpleNamespace
from backend.torch_gc import torch_gc

system = {}
diffusion = {}
models = {}
current_images = []
current_video_frames = []
current_video = ''
rendering = False
render_mode = ''
device = ''
modelFS = ''
modelCS = ''
callbackBusy = False
windows_views = {}
aesthetic_embedding_path = None
T = 0
lr = 0
ti_grad_flag_switch = False
textual_inversion_stop = False
denoiser = None
stop_all = False


#hijack
model_hijack = None

# lowram
lowvram = False
medvram = False
precision = "autocast" #["full", "autocast"], default="autocast"

#interogate
no_half = False
interrogate_keep_models_in_memory = False
interrogate_clip_dict_limit = 1500 # "CLIP: maximum number of lines in text file (0 = No limit)"
interrogate_clip_num_beams = 1 # "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
interrogate_clip_min_length = 24 # "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
interrogate_clip_max_length = 48 # "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
interrogate_use_builtin_artists = True # "Interrogate: use artists from artists.csv"
interrogate_return_ranks = False # "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."
deepbooru_sort_alpha = True # "Interrogate: deepbooru sort alphabetically"
deepbooru_use_spaces = False # "use spaces for tags in deepbooru"
deepbooru_escape = True # "escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"

#preprocess
upscaler_for_img2img= None


#hypernetwork training
loaded_hypernetwork = None
xformers_available = False
opt_split_attention_v1 = False #print("Applying v1 cross attention optimization.") there is no reference to this whatsoever
disable_opt_split_attention = False # there is no reference to this whatsoever where it gets set
opt_split_attention_invokeai = False # there is no reference to this whatsoever where it gets set
opt_split_attention = False # there is no reference to this whatsoever where it gets set
enable_emphasis = True # True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention")
comma_padding_backtrack = 20 # (20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 })
use_old_emphasis_implementation = False # "Use old emphasis implementation. Can be useful to reproduce old seeds."
CLIP_stop_at_last_layers = 1 # "Stop At last layers of CLIP model", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
dataset_filename_word_regex = "" # "Filename word regex"
dataset_filename_join_string = "" # "Filename join string"
training_write_csv_every = 500 # 500, "Save an csv containing the loss to log directory every N steps, 0 to disable")
training_image_repeats_per_epoch = 1 # "Number of repeats for a single input image per epoch; used only for displaying epoch number"
samples_format = None

hypernetworks = {}
class State:
	skipped = False
	interrupted = False
	job = ""
	job_no = 0
	job_count = 0
	job_timestamp = '0'
	sampling_step = 0
	sampling_steps = 0
	current_latent = None
	current_image = None
	current_image_sampling_step = 0
	textinfo = None

	def skip(self):
		self.skipped = True

	def interrupt(self):
		self.interrupted = True

	def nextjob(self):
		self.job_no += 1
		self.sampling_step = 0
		self.current_image_sampling_step = 0

	def dict(self):
		obj = {
			"skipped": self.skipped,
			"interrupted": self.skipped,
			"job": self.job,
			"job_count": self.job_count,
			"job_no": self.job_no,
			"sampling_step": self.sampling_step,
			"sampling_steps": self.sampling_steps,
		}

		return obj

	def begin(self):
		self.sampling_step = 0
		self.job_count = -1
		self.job_no = 0
		self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
		self.current_latent = None
		self.current_image = None
		self.current_image_sampling_step = 0
		self.skipped = False
		self.interrupted = False
		self.textinfo = None

		torch_gc()

	def end(self):
		self.job = ""
		self.job_count = 0

		torch_gc()

state = State()

# interrogate
deepdanbooru = False
interrogate_deepbooru_score_threshold = 0.0

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
