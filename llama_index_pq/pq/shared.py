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
	unfixed = prompt
	prompt = fix_brackets(prompt)
	if prompt == '':
		prompt = unfixed

	prompt = re.sub(r"\s+", " ", prompt)
	if '\n' in prompt:
		prompt = re.sub(r'.*\n', '', prompt)

	artefacts = ['Answer: ','Steps: ','scale: ', 'Seed: ', 'Face restoration: ', 'Size: ', 'Model hash: ', 'Model: ', 'Clip skip: ',
				 'Token merging ratio: ', 'ADetailer .*?: ']

	for artefact in artefacts:
		if artefact in prompt:
			prompt = re.sub(rf'.*{artefact}', '', prompt)

	return prompt

def fix_brackets(text):
	"""
	Cleans the text by removing all brackets if they are not closed and non-nested,
	while also removing patterns like ":1.2" (colon followed by number-dot-number).

	Args:
		text (str): The input string.

	Returns:
		str: The cleaned string with brackets removed or the original text if
			brackets are balanced and non-nested.
	"""
	opening_brackets = {"(", "[", "{"}
	closing_brackets = {")": "(", "]": "[", "}": "{"}
	stack = []
	cleaned_text = ""
	number_pattern = r":\d+\.\d+"  # Regular expression for number pattern (:number.number)

	for char in text:
		if char in opening_brackets:
			stack.append(char)
			cleaned_text += char
		elif char in closing_brackets:
			if not stack or stack.pop() != closing_brackets[char]:
				# Not balanced or non-nested, remove brackets and number patterns
				cleaned_text = re.sub(r"[()]|\[|\]|\{|\}", "", cleaned_text, flags=re.MULTILINE)  # Remove all bracket types
				return re.sub(number_pattern, "", cleaned_text, flags=re.MULTILINE)  # Remove number patterns
			else:
				cleaned_text += char
		else:
			# Check for number pattern before adding non-bracket characters
			if not re.match(number_pattern, cleaned_text + char):
				cleaned_text += char

	# All brackets are closed and non-nested, return the original text
	if not stack:
		return text
	else:
		# Unclosed brackets, remove brackets and number patterns
		cleaned_text = re.sub(r"[()]|\[|\]|\{|\}", "", cleaned_text, flags=re.MULTILINE)  # Remove all bracket types
		text = re.sub(number_pattern, "", text, flags=re.MULTILINE)
		return text