
import re
import globals
from post_process.summary import extractive_summary
from llm_fw import llm_interface_qdrant

class api_sail:

	def __init__(self):
		self.g = globals.get_globals()
		self.interface = llm_interface_qdrant.get_interface()
		self.last_api_sail_query = None
		self.api_sail_depth_start = 0
		self.api_sail_depth = 0

	def shorten_string(self, text, max_bytes=1000):
		"""Shortens a string to a maximum of 1000 bytes.

		Args:
		  text: The string to shorten.
		  max_bytes: The maximum number of bytes allowed (default: 1000).

		Returns:
		  The shortened string, truncated at the last whole word before reaching
		  max_bytes.
		"""
		if len(text) <= max_bytes:
			return text

		# Encode the text as UTF-8 to get byte length
		encoded_text = text.encode('utf-8')

		# Truncate the string while staying under the byte limit
		while len(encoded_text) > max_bytes:
			# Split text by words on space
			words = text.rsplit()
			# Remove the last word and try again
			text = ' '.join(words[:-1])
			encoded_text = text.encode('utf-8')

		return text
	def clean_llm_artefacts(self, prompt):
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
		if '\n' in prompt:
			prompt = re.sub(r'.*\n', '', prompt)
		if 'Answer: ' in prompt:
			prompt = re.sub(r'.*Answer: ', '', prompt)
		return prompt

	def get_next_target(self, nodes):
		target_dict = self.interface.get_query_texts(nodes)

		if len(target_dict.keys()) < self.api_sail_depth:
			self.api_sail_depth = self.api_sail_depth_start + len(self.g.api_sail_history)

		if len(target_dict.keys()) > 0:
			out =  target_dict[min(target_dict.keys())]
			self.g.api_sail_history.append(out)
			return out

		else:
			return -1

	def run_api_sail(self, data):

		if data['reset_journey'] is True:
			self.last_api_sail_query = None
			self.api_sail_depth_start = 0
			self.g.api_sail_history = []

		if self.last_api_sail_query is None:
			self.last_api_sail_query = data['query']

		if self.api_sail_depth_start == 0:
			self.api_sail_depth_start = data['distance']
			self.api_sail_depth = data['distance']

		query = self.last_api_sail_query

		#if self.g.settings_data['translate']:
		#    query = self.interface.translate(self.g.settings_data['sail_text'])

		try:

			if data['add_search'] is True:
				query = f'{data["search"]}, {query}'

			if len(query) > 1000:
				query = extractive_summary(query,num_sentences=2)
				if len(query) > 1000:
					query = self.shorten_string(query)

			prompt = self.interface.retrieve_query(query)

			prompt = self.clean_llm_artefacts(prompt)

			if data['summary'] is True:
				prompt = extractive_summary(prompt)

			if data['rephrase'] is True:
				prompt = self.interface.rephrase(prompt, data['rephrase_prompt'])

			if data['add_style'] is True:
				prompt = f'{data["style"]}, {prompt}'

			nodes = self.interface.retrieve_top_k_query(query, self.api_sail_depth)

			self.last_api_sail_query = self.get_next_target(nodes)

			negative_out = ''

			if len(self.g.negative_prompt_list) > 0:
				self.g.last_negative_prompt = ",".join(self.g.negative_prompt_list).lstrip(' ')
				if len(self.g.last_negative_prompt) < 30:
					self.g.last_negative_prompt = self.g.settings_data['negative_prompt']
				if self.g.last_negative_prompt != '':
					negative_out = self.g.last_negative_prompt
				else:
					negative_out = self.g.settings_data['negative_prompt']


			if query == -1:
				out_dict = {
					"prompt":f'sail is finished early due to rotating context',
					"neg_prompt":''

				}
				return out_dict
			else:
				out_dict = {
					"prompt":prompt,
					"neg_prompt":negative_out

				}
				return out_dict
		except Exception as e:

			out_dict = {
				"prompt":'some error happened: ' + str(e),
				"neg_prompt":''

			}
			return out_dict