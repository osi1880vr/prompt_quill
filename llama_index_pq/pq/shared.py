import time
import datetime
import re

def timestamp():
	return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def clean_llm_artefacts(prompt):
	"""
	Cleans potential artefacts (artifacts) left behind by large language models (LLMs) from a given prompt.

	This function specifically removes two common artefacts:

		* Newline character at the beginning: This can occur when LLMs generate text that starts on a new line.
		* "Answer: " prefix: This might be added by some LLMs as a prefix to their generated response.

	Args:
		prompt (str): The prompt text to be cleaned.

	Returns:
		str: The cleaned prompt text without the identified artefacts.
	"""
	prompt = re.sub(r"\s+", " ", prompt)
	if '\n' in prompt:
		prompt = re.sub(r'.*\n', '', prompt)
	if 'Answer: ' in prompt:
		prompt = re.sub(r'.*Answer: ', '', prompt)
	return prompt