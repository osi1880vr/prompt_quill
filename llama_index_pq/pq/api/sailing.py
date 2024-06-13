
import re
import globals
from post_process.summary import extractive_summary
from llm_fw import llm_interface_qdrant
import shared
import json


class api_sail:

	def __init__(self):
		self.g = globals.get_globals()
		self.interface = llm_interface_qdrant.get_interface()
		self.last_api_sail_query = None
		self.api_sail_depth_start = 0
		self.api_sail_depth = 0
		self.api_sail_count = 0

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
	def get_next_target(self, nodes):

		if len(nodes) < self.api_sail_depth:
			self.api_sail_depth = self.api_sail_depth_start + len(self.g.api_sail_history)

		if len(nodes) > 0:

			if self.g.settings_data['sail_target']:
				node = nodes[len(nodes)-1]
				payload = json.loads(node.payload['_node_content'])
				out = payload['text']
				self.g.api_sail_history.append(out)
				return out
			else:
				node = nodes[0]
				payload = json.loads(node.payload['_node_content'])
				out = payload['text']
				self.g.api_sail_history.append(out)
				return out
		else:
			return -1

	def run_api_sail(self, data):

		if data['reset_journey'] is True:
			self.last_api_sail_query = None
			self.api_sail_depth_start = 0
			self.g.api_sail_history = []
			self.api_sail_count = 0

		if self.last_api_sail_query is None:
			self.last_api_sail_query = data['query']

		if self.api_sail_depth_start == 0:
			self.api_sail_depth_start = data['distance']
			self.api_sail_depth = data['distance']

		query = self.last_api_sail_query

		#if self.g.settings_data['translate']:
		#    query = self.interface.translate(self.g.settings_data['sail_text'])

		try:

			if 'unload_llm' in data:
				if data['unload_llm'] is True:
					self.g.settings_data['unload_llm'] = True

			if data['add_search'] is True:
				query = f'{data["search"]}, {query}'

			if len(query) > 1000:
				query = extractive_summary(query,num_sentences=2)
				if len(query) > 1000:
					query = self.shorten_string(query)

			prompt = self.interface.retrieve_llm_completion(query)

			prompt = shared.clean_llm_artefacts(prompt)

			if data['summary'] is True:
				prompt = extractive_summary(prompt)

			if data['rephrase'] is True:
				prompt = self.interface.rephrase(prompt, data['rephrase_prompt'])

			if data['add_style'] is True:
				prompt = f'{data["style"]}, {prompt}'

			nodes = self.interface.direct_search(self.g.settings_data['sail_text'],self.api_sail_depth,self.api_sail_count)
			self.api_sail_count += 1

			self.last_api_sail_query = self.get_next_target(nodes)

			negative_out = shared.get_negative_prompt()



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