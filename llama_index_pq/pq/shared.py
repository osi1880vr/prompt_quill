import time
import datetime
import re
import os
import numpy as np
import globals
import random
from typing import Dict, Union, Optional, Tuple, List


g = globals.get_globals()


class PromptProcessor:
	def __init__(self):
		# This dictionary tracks state for each array token.
		# The key is the array token's full text.
		# For sequential modes, we store a dict with keys:
		#   "index"     -> current position in the array,
		#   "rep_count" -> how many times the current item has been repeated,
		#   "rep_target"-> the number of times to repeat the item (1 for "iter", or n for "iter n")
		self.prompt_array_index = {}

	def process_prompt_arrays(self, input_string):
		tokens = self.tokenize_input(input_string)
		output_tokens = []

		for token_type, token_value in tokens:
			if token_type == 'text':
				# Literal text is passed through unchanged.
				output_tokens.append(token_value)
			elif token_type == 'array':
				# Process the array token.
				# token_value includes the outer brackets.
				array_content = token_value[1:-1]  # remove [ and ]
				elements = self.parse_array(array_content)

				# Determine the mode and repetition.
				# Default mode: random selection.
				mode = "random"
				rep = 1
				if elements and elements[0].lower().startswith("iter"):
					parts = elements[0].split()
					if len(parts) == 1:
						mode = "sequential"
						rep = 1
					elif len(parts) == 2 and parts[0].lower() == "iter":
						try:
							rep = int(parts[1])
							mode = "sequential"
						except ValueError:
							mode = "random"
					# Remove the iteration marker from the choices.
					elements = elements[1:]

				# Process nested arrays recursively in each element.
				processed_elements = []
				for item in elements:
					if '[' in item and ']' in item:
						processed_elements.append(self.process_prompt_arrays(item))
					else:
						processed_elements.append(item)

				if mode == "sequential":
					# Get state for this token.
					state = self.prompt_array_index.get(token_value, {"index": 0, "rep_count": 0, "rep_target": rep})
					if processed_elements:
						selected_item = processed_elements[state["index"]]
						# Update state:
						if state["rep_count"] < rep - 1:
							state["rep_count"] += 1
						else:
							state["rep_count"] = 0
							state["index"] = (state["index"] + 1) % len(processed_elements)
						self.prompt_array_index[token_value] = state
					else:
						selected_item = ""
				else:
					# Default (random) mode: choose a random element.
					selected_item = random.choice(processed_elements) if processed_elements else ""

				output_tokens.append(selected_item)
		return "".join(output_tokens).strip()

	def find_balanced_bracket(self, s, start_idx):
		"""
        Starting at s[start_idx] (which should be '['), finds the matching ']'
        while handling nested brackets and quoted strings.
        """
		i = start_idx
		bracket_count = 0
		in_quotes = False

		while i < len(s):
			char = s[i]
			if char == '"' and (i == start_idx or s[i - 1] != '\\'):
				in_quotes = not in_quotes
			elif char == '[' and not in_quotes:
				bracket_count += 1
			elif char == ']' and not in_quotes:
				bracket_count -= 1
				if bracket_count == 0:
					return i
			i += 1
		return -1  # No matching bracket found.

	def tokenize_input(self, s):
		"""
        Splits the input string into tokens:
          - ("text", literal text)
          - ("array", full array text including surrounding [ and ]).
        """
		tokens = []
		i = 0
		last = 0
		while i < len(s):
			if s[i] == '[':
				if i > last:
					tokens.append(("text", s[last:i]))
				end_idx = self.find_balanced_bracket(s, i)
				if end_idx != -1:
					tokens.append(("array", s[i:end_idx + 1]))
					i = end_idx + 1
					last = i
					continue
			i += 1
		if last < len(s):
			tokens.append(("text", s[last:]))
		return tokens

	def parse_array(self, array_string):
		"""
        Parses an array string (without the outer brackets) into items.
        Respects quoted substrings and ignores commas inside quotes or nested brackets.
        Returns a list of strings with any outer quotes removed.
        """
		elements = []
		buffer = ""
		in_quotes = False
		bracket_level = 0
		i = 0
		while i < len(array_string):
			char = array_string[i]
			if char == '"' and (i == 0 or array_string[i - 1] != '\\'):
				in_quotes = not in_quotes
				buffer += char
			elif char == '[' and not in_quotes:
				bracket_level += 1
				buffer += char
			elif char == ']' and not in_quotes:
				bracket_level -= 1
				buffer += char
			elif char == ',' and not in_quotes and bracket_level == 0:
				elements.append(buffer.strip())
				buffer = ""
			else:
				buffer += char
			i += 1
		if buffer.strip():
			elements.append(buffer.strip())
		# Remove outer quotes from each element if they exist.
		result = []
		for el in elements:
			if el.startswith('"') and el.endswith('"'):
				result.append(el[1:-1])
			else:
				result.append(el)
		return result

class WildcardResolver:
	def __init__(self, wildcards_dir: str = "wildcards", cache_files: bool = True):
		self.wildcards_dir = wildcards_dir
		self.cache_files = cache_files
		self.wildcard_cache: Dict[str, List[str]] = {}
		self.iter_state: Dict[str, int] = {}
		os.makedirs(wildcards_dir, exist_ok=True)

	def load_wildcard_file(self, wildcard: str, count: int = 1) -> List[str]:
		if wildcard in self.wildcard_cache and self.cache_files:
			options = self.wildcard_cache[wildcard]
		else:
			wildcard_file = os.path.join(self.wildcards_dir, f"{wildcard}.txt")
			if os.path.exists(wildcard_file):
				with open(wildcard_file, "r", encoding="utf-8") as f:
					options = [line.strip() for line in f if line.strip()]
			else:
				for root, _, files in os.walk(self.wildcards_dir):
					if f"{wildcard}.txt" in files:
						wildcard_file = os.path.join(root, f"{wildcard}.txt")
						with open(wildcard_file, "r", encoding="utf-8") as f:
							options = [line.strip() for line in f if line.strip()]
						break
				else:
					options = [f"__{wildcard}__"]
			if self.cache_files:
				self.wildcard_cache[wildcard] = options

		if not options:
			return [f"__{wildcard}__"] * count

		unique_options = list(set(options))
		if len(unique_options) < count:
			selected = unique_options
		else:
			selected = random.sample(unique_options, count)
		return selected

	def parse_inline_options(self, options_str: str) -> List[Tuple[str, float]]:
		if not options_str:
			return [(" ", 1.0)]
		parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', options_str)
		is_iter = False
		repeat_count = 1
		if parts and parts[0].strip().startswith("iter"):
			is_iter = True
			iter_part = parts[0].strip()
			if iter_part.startswith("iter "):
				try:
					repeat_count = int(iter_part.split(" ")[1])
					parts = parts[1:]
				except (IndexError, ValueError):
					parts = parts[1:]
			else:
				parts = parts[1:]

		options = []
		wildcard_pattern = r"(\d*)x?__([^_]+)__"
		for part in parts:
			part = part.strip()
			# Parse weight (e.g., "red:0.7")
			if ':' in part and not part.startswith('"'):
				opt, weight = part.rsplit(':', 1)
				try:
					weight = float(weight)
					if weight < 0:
						weight = 0.0  # No negative weights
				except ValueError:
					weight = 1.0  # Default if invalid
				opt = opt.strip()
			else:
				opt = part
				weight = 1.0

			if opt.startswith('"') and opt.endswith('"'):
				opt = opt[1:-1]
				if re.match(wildcard_pattern, opt):
					match = re.findall(wildcard_pattern, opt)[0]
					count = int(match[0]) if match[0] else 1
					wildcard = match[1]
					options.extend([(item, weight) for item in self.load_wildcard_file(wildcard, count)])
				else:
					options.append((opt, weight))
			else:
				if re.match(wildcard_pattern, opt):
					match = re.findall(wildcard_pattern, opt)[0]
					count = int(match[0]) if match[0] else 1
					wildcard = match[1]
					options.extend([(item, weight) for item in self.load_wildcard_file(wildcard, count)])
				else:
					options.append((opt, weight))

		if is_iter:
			return [("iter", repeat_count, options)]
		# Weighted random choice
		if options:
			items, weights = zip(*options)
			return [(random.choices(items, weights=weights, k=1)[0], 1.0)]
		return [(" ", 1.0)]

	def find_inline_matches(self, prompt: str) -> List[str]:
		matches = []
		i = 0
		while i < len(prompt):
			if prompt[i] == '[':
				start = i + 1
				depth = 1
				while i + 1 < len(prompt) and depth > 0:
					i += 1
					if prompt[i] == '[':
						depth += 1
					elif prompt[i] == ']':
						depth -= 1
				if depth == 0:
					matches.append(prompt[start:i])
			i += 1
		return matches

	def resolve_prompt(
			self,
			prompt: str,
			max_combinations: Optional[int] = None,
			recursive: bool = True,
			separator: str = " and ",
			max_depth: int = 10,
			max_retries: int = 1  # Reduce retries since we resolve/skip each time
	) -> Union[str, List[str]]:
		wildcard_pattern = r"(\d*)x?__([\w-]+)__"
		multi_wildcard_pattern = r"\{(\d+)\$\$__([\w-]+)__(:[\d\.]+)?\}"
		weighted_wildcard_pattern = r"__([\w-]+)__:([\d\.]+)"

		wildcards = re.findall(wildcard_pattern, prompt)
		multi_wildcards = re.findall(multi_wildcard_pattern, prompt)
		weighted_wildcards = re.findall(weighted_wildcard_pattern, prompt)
		inline_matches = self.find_inline_matches(prompt)
		inline_options = [self.parse_inline_options(match) for match in inline_matches]

		if not wildcards and not multi_wildcards and not weighted_wildcards and not inline_matches:
			return prompt if max_combinations is None else [prompt]

		def attempt_resolution(prompt, attempt):
			processed_wildcards = set()
			wildcard_options = {}
			# Plain wildcards (no weight)
			for count, wildcard in wildcards:
				if wildcard not in [w for w, _ in weighted_wildcards]:
					count = int(count) if count else 0
					options = self.load_wildcard_file(wildcard, count if count > 0 else 1)
					wildcard_options[(count, wildcard)] = separator.join(options)
			# Weighted plain wildcards
			for wildcard, weight in weighted_wildcards:
				weight_value = float(weight)
				options = self.load_wildcard_file(wildcard, 1)
				replacement = separator.join(options) if random.random() < weight_value else ""
				wildcard_options[(0, wildcard, "weighted", weight)] = replacement
			# Multi-wildcards (store full options)
			for count, wildcard, weight in multi_wildcards:
				count = int(count)
				options = self.load_wildcard_file(wildcard, count)
				if len(options) < count:
					options = options * (count // len(options) + 1)[:count]
				wildcard_options[(count, wildcard, "multi", weight)] = separator.join(options)

			resolved = prompt
			# Multi-wildcards first
			for key, replacement in wildcard_options.items():
				if len(key) == 4 and key[2] == "multi":
					count, wildcard, _, weight = key
					weight_value = float(weight[1:]) if weight else 1.0
					weight_str = weight if weight else ""
					target = f"{{{count}$$__{wildcard}__{weight_str}}}"
					if target in resolved:
						if random.random() < weight_value:
							resolved = resolved.replace(target, replacement, 1)
						else:
							resolved = resolved.replace(target, "")  # Remove on skip
						processed_wildcards.add(wildcard)  # Mark as processed either way
			# Then weighted plain wildcards
			for key, replacement in wildcard_options.items():
				if len(key) == 4 and key[2] == "weighted":
					_, wildcard, _, weight = key
					target = f"__{wildcard}__:{weight}"
					if target in resolved:
						resolved = resolved.replace(target, replacement, 1)
						processed_wildcards.add(wildcard)
			# Then plain wildcards
			for key, replacement in wildcard_options.items():
				if len(key) == 2:
					count, wildcard = key
					target = f"{count}x__{wildcard}__" if count > 0 else f"__{wildcard}__"
					if target in resolved:
						resolved = resolved.replace(target, replacement, 1)
						processed_wildcards.add(wildcard)

			depth = 0
			if recursive:
				while re.search(wildcard_pattern, resolved) or re.search(multi_wildcard_pattern, resolved) or re.search(weighted_wildcard_pattern, resolved):
					if depth >= max_depth:
						break
					nested_wildcards = re.findall(wildcard_pattern, resolved)
					nested_multi_wildcards = re.findall(multi_wildcard_pattern, resolved)
					nested_weighted_wildcards = re.findall(weighted_wildcard_pattern, resolved)
					for count, wildcard in nested_wildcards:
						if wildcard not in processed_wildcards and wildcard not in [w for w, _ in nested_weighted_wildcards]:
							count = int(count) if count else 0
							options = self.load_wildcard_file(wildcard, count if count > 0 else 1)
							target = f"{count}x__{wildcard}__" if count > 0 else f"__{wildcard}__"
							if target in resolved:
								resolved = resolved.replace(target, separator.join(options), 1)
								processed_wildcards.add(wildcard)
					for wildcard, weight in nested_weighted_wildcards:
						if wildcard not in processed_wildcards:
							weight_value = float(weight)
							if random.random() < weight_value:
								options = self.load_wildcard_file(wildcard, 1)
								replacement = separator.join(options)
							else:
								replacement = ""
							target = f"__{wildcard}__:{weight}"
							if target in resolved:
								resolved = resolved.replace(target, replacement, 1)
								processed_wildcards.add(wildcard)
					for count, wildcard, weight in nested_multi_wildcards:
						if wildcard not in processed_wildcards:
							count = int(count)
							weight_value = float(weight[1:]) if weight else 1.0
							if random.random() < weight_value:
								options = self.load_wildcard_file(wildcard, count)
								if len(options) < count:
									options = options * (count // len(options) + 1)[:count]
							else:
								options = [""]
							target = f"{{{count}$$__{wildcard}__{weight if weight else ''}}}"
							if target in resolved:
								resolved = resolved.replace(target, separator.join(options), 1)
								processed_wildcards.add(wildcard)
					depth += 1
			return resolved, processed_wildcards

		num_outputs = 1 if max_combinations is None else 1
		results = []
		for _ in range(num_outputs):
			best_resolved = prompt
			cycle_detected = False

			for attempt in range(max_retries):
				resolved, _ = attempt_resolution(prompt, attempt)
				best_resolved = resolved
				if not (re.search(wildcard_pattern, resolved) or re.search(multi_wildcard_pattern, resolved) or re.search(weighted_wildcard_pattern, resolved)):
					break
				cycle_detected = True

			for match, opts in zip(inline_matches, inline_options):
				if len(opts) == 1 and isinstance(opts[0], tuple) and opts[0][0] == "iter":
					repeat_count, iter_opts = opts[0][1], opts[0][2]
					state_key = match
					if state_key not in self.iter_state:
						self.iter_state[state_key] = 0
					total_items = len(iter_opts) * repeat_count
					current_idx = self.iter_state[state_key] % total_items
					opt_idx = current_idx // repeat_count
					replacement = iter_opts[opt_idx][0]
					if recursive and re.search(r"[\[\]]|(\d*)x?__[\w-]+__|\{(\d+)\$\$__[\w-]+__\}", best_resolved):
						replacement = self.resolve_prompt(replacement, None, True, separator)
					self.iter_state[state_key] += 1
				else:
					replacement = opts[0][0]
					if recursive and re.search(r"[\[\]]|(\d*)x?__[\w-]+__|\{(\d+)\$\$__[\w-]+__\}", best_resolved):
						replacement = self.resolve_prompt(replacement, None, True, separator)
				best_resolved = best_resolved.replace(f"[{match}]", replacement, 1)
			results.append(best_resolved)

		return results[0]





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
	prompt = repair_brackets(prompt)
	if prompt == '':
		prompt = unfixed

	prompt = re.sub(r"\s+", " ", prompt)
	if '\n' in prompt:
		prompt = re.sub(r'.*\n', '', prompt)

	artefacts = ['Answer: ','Steps: ','scale: ', 'Seed: ', 'Face restoration: ', 'Size: ', 'Model hash: ', 'Model: ', 'Clip skip: ',
				 'Token merging ratio: ', 'ADetailer .*?: ','"','\[', '\]', '\{', '\}']

	for artefact in artefacts:
		if artefact in prompt:
			prompt = re.sub(rf'.*{artefact}', '', prompt)

	pattern = r"<\|(.*?)\|>"
	prompt = re.sub(pattern, "", prompt)
	pattern = r"<(.*?)>"
	prompt = re.sub(pattern, "", prompt)

	return prompt

def repair_brackets(txt):
	# split the text into words
	words = txt.split()
	# create an empty list to store the repaired words
	repaired_words = []
	# iterate over the words
	for word in words:
		# check if the word starts with a bracket
		if word.startswith('('):
			# check if the word ends with a bracket
			# there might be  punctuation marks in the word that are not brackets
			# check for that situation to not add opening bracket because the word does not end exactly with a closing bracket but a closing bracket is present
			if word.endswith(')'):
				# if the word starts and ends with a bracket, append the word to the repaired words list

				repaired_words.append(word)
			else:
				# if the word starts with a bracket but does not end with a bracket, append the word with a closing bracket to the repaired words list
				# check if there is  punctuation mark in the word
				# the bracket has to be at the end of the word but before the  punctuation mark
				# the bracket has to be at the end of the word if there is no  punctuation mark
				if word[-1] in ',.!?':
					# if there is a  punctuation mark in the word, append the word with a closing bracket before the  punctuation mark to the repaired words list
					repaired_words.append(word[:-1] + ')' + word[-1])
				else:
					# if there is no  punctuation mark in the word, append the word with a closing bracket to the repaired words list
					repaired_words.append(word + ')')

		# check if the word ends with a bracket
		# or with a bracket and a  punctuation mark

		elif word.endswith(')'):

			# if the word ends with a bracket but does not start with a bracket, append the word with an opening bracket to the repaired words list
			# but check for if there is a  punctuation mark in the word
			# the bracket has to be at the start of the word but after the  punctuation mark
			# the bracket has to be at the start of the word if there is no  punctuation mark
			if word[0] in ',.!?':
				# if there is a  punctuation mark in the word, append the word with an opening bracket before the  punctuation mark to the repaired words list
				repaired_words.append(word[0] + '(' + word[1:])
			else:
				# if there is no  punctuation mark in the word, append the word with an opening bracket to the repaired words list
				repaired_words.append('(' + word)
		# elif word ends with ),.!?:
		# if the word ends with a bracket and a  punctuation mark, append the word with an opening bracket before the  punctuation mark to the repaired words list
		# the bracket has to be at the start of the word but after the  punctuation mark
		elif word[-1] in ',.!?':
			if ')' in word:
				repaired_words.append('(' + word[:-1] + word[-1])
			else:
				repaired_words.append(word)
		else:
			# if the word does not start or end with a bracket, append the word to the repaired words list
			repaired_words.append(word)
	# join the repaired words list into a repaired text
	repaired_txt = ' '.join(repaired_words)
	# return the repaired text
	return repaired_txt





def repair_brackets_snipets(text):
	# split the text into a list of characters
	text = list(text)
	out_text = list(text)
	# create a stack to store the indexes of the brackets
	stack = []
	# iterate over the characters in the text
	# create an array of the text

	for i, char in enumerate(text):
		# if the character is an opening bracket
		if char == '(':
			# add the index to the stack
			stack.append(i)
		# if the character is a closing bracket
		elif char == ')':
			# if the stack is not empty
			if stack:
				# pop the last index from the stack
				stack.pop()
			# if the stack is empty
			else:
				# add an opening bracket to the beginning of the text
				out_text.insert(0, '(')

	# iterate over the indexes in the stack
	for i in stack:
		# add a closing bracket at the end of the text
		out_text.append(')')
	# join the characters back into a string
	return ''.join(out_text)


def fix_array_brackets(text):
	out = []
	for word in text:
		if word != '':
			out.append(repair_brackets_snipets(word.strip()))

	return out

def get_negative_prompt():

	negative_out = ''
	if len(g.negative_prompt_list) > 0:

		negative_prompt = fix_array_brackets(g.negative_prompt_list)
		g.last_negative_prompt = ','.join(negative_prompt)

		if len(g.last_negative_prompt) < 30:
			g.last_negative_prompt = g.settings_data['negative_prompt']
		if g.last_negative_prompt != '':
			negative_out = g.last_negative_prompt
	else:
		negative_out = g.settings_data['negative_prompt']

	return negative_out

def check_filtered(prompt):

	not_check = False
	check = False
	if len(g.settings_data['sailing']['sail_filter_not_text']) > 0:
		not_check = True
		check_count = 0
		search = set(word.strip().lower() for word in g.settings_data['sailing']['sail_filter_not_text'].split(','))
		for word in search:
			if word in prompt:
				check_count += 1
		if check_count == len(search):
			not_check = False


	if len(g.settings_data['sailing']['sail_filter_text']) > 0:
		search = set(word.strip().lower() for word in g.settings_data['sailing']['sail_filter_text'].split(','))
		check_count = 0
		for word in search:
			if word in prompt:
				check_count += 1
		if check_count == len(search):
			check = True


	return not_check + check


def is_image_black(image):
	"""
	Check if a PIL image is completely black.

	Parameters:
	image (PIL.Image): The image to check.

	Returns:
	bool: True if the image is completely black, False otherwise.
	"""
	# Convert the image to a NumPy array
	image_array = np.array(image)

	# If the image has an alpha channel, ignore the alpha channel in the comparison
	if image_array.shape[-1] == 4:
		image_array = image_array[:, :, :3]

	# Check if all values in the array are 0 (black)
	return np.all(image_array == 0)

def sanitize_path_component(component, replacement='_'):
	"""
	Sanitizes a string to make it a valid file path component.

	Parameters:
	- component (str): The string to sanitize.
	- replacement (str): The string to replace invalid characters with. Default is '_'.

	Returns:
	- str: The sanitized string.
	"""
	# List of invalid characters for Windows
	invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'

	# Replace invalid characters with the specified replacement
	sanitized = re.sub(invalid_chars, replacement, component)

	# Trim leading and trailing spaces
	sanitized = sanitized.strip()

	return sanitized

def to_utf8_string(text):
	cleaned_text = re.sub(r'[^\x00-\x7F]+', '?', text)
	try:
		return cleaned_text.encode('utf-8').decode('utf-8')
	except UnicodeDecodeError:
		# Return encoded bytes on error (may contain invalid characters)
		return 'text did contain non utf8 convertible data, therefore lets make a cute kitten prompt'


