from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc



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
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id, trust_remote_code=True, revision=revision
		).to(device=device)
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)



	def unload(self):
		del self.model
		del self.tokenizer
		self.model = None
		self.tokenizer = None
		gc.collect()
		torch.cuda.empty_cache()


	def run_interrogation(self, img, prompt):
		if prompt == '':
			prompt = "Describe this image."

		if self.model is None:
			self.setModel()

		enc_image = self.model.encode_image(img)
		answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
		gc.collect()
		torch.cuda.empty_cache()

		return answer