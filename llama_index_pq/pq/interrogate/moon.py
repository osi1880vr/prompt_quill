from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re
from PIL import ImageDraw
from torchvision.transforms.v2 import Resize

class moon:


	def __init__(self):
		self.model = None
		self.tokenizer = None

		pass

	def detect_device(self):
		"""
		Detects the appropriate device to run on, and return the device and dtype.
		"""

		if torch.cuda.is_available():
			return torch.device("cuda"), torch.float16
		elif torch.backends.mps.is_available():
			return torch.device("mps"), torch.float16
		else:
			return torch.device("cpu"), torch.float32

	def setModel(self):
		device, dtype = self.detect_device()
		model_id = "vikhyatk/moondream2"
		revision = "2024-05-20"
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id, trust_remote_code=True, revision=revision
		).to(device=device,dtype=dtype)
		self.model.eval()


	def unload(self):
		del self.model
		del self.tokenizer
		self.model = None
		self.tokenizer = None
		gc.collect()
		torch.cuda.empty_cache()


	def run_interrogation(self, img, prompt,max_new_tokens):
		if prompt == '':
			prompt = "Describe this image."

		if self.model is None:
			self.setModel()

		enc_image = self.model.encode_image(img)

		answers = []

		if '\n' in prompt:
			questions = prompt.split('\n')
			for question in questions:
				question = f"<image>\n\nQuestion: {question}\n\nAnswer:"

				answer = self.model.generate(enc_image, question, self.tokenizer, max_new_tokens=max_new_tokens)[0]
				answers.append(f'{question}:<br>{answer}<br>')
		else:
			prompt = f"<image>\n\nQuestion: {prompt}\n\nAnswer:"
			answers.append(self.model.generate(enc_image, prompt, self.tokenizer, max_new_tokens=max_new_tokens)[0])
		gc.collect()
		torch.cuda.empty_cache()

		return '<br><br>'.join(answers)


	def extract_floats(self, text):
		# Regular expression to match an array of four floating point numbers
		pattern = r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
		match = re.search(pattern, text)
		if match:
			# Extract the numbers and convert them to floats
			return [float(num) for num in match.groups()]
		return None  # Return None if no match is found


	def extract_bbox(self, text):
		bbox = None
		if self.extract_floats(text) is not None:
			x1, y1, x2, y2 = self.extract_floats(text)
			bbox = (x1, y1, x2, y2)
		return bbox

	def moon_process_answer(self, img, answer):
		if self.extract_bbox(answer) is not None:
			x1, y1, x2, y2 = self.extract_bbox(answer)
			draw_image = Resize(768)(img)
			width, height = draw_image.size
			x1, x2 = int(x1 * width), int(x2 * width)
			y1, y2 = int(y1 * height), int(y2 * height)
			bbox = (x1, y1, x2, y2)
			ImageDraw.Draw(draw_image).rectangle(bbox, outline="red", width=3)
			return gr.update(visible=True, value=draw_image)

		return gr.update(visible=False, value=None)