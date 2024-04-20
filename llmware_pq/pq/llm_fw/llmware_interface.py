import gc
import os

from llmware.library import Library
from llmware.retrieval import Query
from llmware.prompts import Prompt
from llmware.gguf_configs import GGUFConfigs
import globals

host = 'localhost'
mongo_host = 'localhost'

if os.getenv("QDRANT_HOST") is not None:
    host = os.getenv("QDRANT_HOST")

if os.getenv("MONGO_HOST") is not None:
    mongo_host = os.getenv("MONGO_HOST")

os.environ['COLLECTION_DB_URI'] = f'mongodb://{mongo_host}:27017/'
os.environ["USER_MANAGED_QDRANT_HOST"] = host
os.environ["USER_MANAGED_QDRANT_PORT"] = "6333"



class adapter:
    def __init__(self):
        self.g = globals.get_globals()
        GGUFConfigs().set_config("n_gpu_layers", 50)

        self.embedding_model_name = 'mini-lm-sbert' #'nomic-embed-text-v1' #'mini-lm-sbert'

        self.library_name = 'llmware_meta_qdrant'
        self.lib = Library().load_library(self.library_name)



        self.max_tokens=self.g.settings_data['max output Tokens']
        self.temperature=self.g.settings_data['Temperature']
        self.top_k=self.g.settings_data['top_k']
        self.n_ctx=self.g.settings_data['Context Length']
    
    
    
        self.run_order_list = ["blurb1", "$context", "blurb2", "$query", "instruction"]
    
        self.prompt_dict = self.g.settings_data['prompt_templates']['prompt_template_a']
        self.prompt_template = self.prompt_dict["blurb1"]
    
        self.model_name = 'TheBloke/Panda-7B-v0.1-GGUF'
        self.hf_repo_name = self.g.settings_data['model_list'][self.model_name]['repo_name']
        self.model_file = self.g.settings_data['model_list'][self.model_name]['file']
        self.model_type = 'deep_link'
    
        self.set_pipeline()

    def retrieve_context(self,query):
        return self.query.semantic_query(query, result_count=self.g.settings_data['top_k'])

    def get_context_text(self, query):
        query_results = self.retrieve_context(query)
        return [s['text'].replace('\n','') for s in query_results]

    def get_query_texts(self, query_results):
        out_dict = {}
        for result in query_results:
            if result['text'] not in self.g.sail_history:
                out_dict[result['distance']] = result['text']
        return out_dict

    def retrieve_query(self, query):
        context = self.aggregate_text_by_query(query, top_n=self.g.settings_data['top_k'])
        response = self.prompter.prompt_main(query, prompt_name="image_prompt",context=context)
        return response["llm_response"].lstrip(" ")

    def retrieve_top_k_query(self, query, top_k):
        context = self.query.semantic_query(query, result_count=top_k)
        return context


    def prepare_meta_data(self,query_results):
        self.g.negative_prompt_list = []
        self.g.models_list = []
        for result in query_results:
            if result['special_field1']['negative_prompt'] != '':
                self.g.negative_prompt_list.append(result['special_field1']['negative_prompt'])
            if result['special_field1']['model'] != '':
                self.g.models_list.append(f"https://civitai.com/models/{result['special_field1']['model']}")


    def aggregate_text_by_query(self, query, top_n=5):

        # run query
        query_results = self.retrieve_context(query)
        self.prepare_meta_data(query_results)
        self.g.last_context = [s['text'].replace('\n','') for s in query_results]

        prompt_consolidator = ""
        for j, results in enumerate(query_results):
            prompt_consolidator += results["text"] + "\n"

        return prompt_consolidator

    def change_model(self, model, temperature, n_ctx, max_tokens, gpu_layers, top_k, instruct):

        GGUFConfigs().set_config("n_gpu_layers", gpu_layers)
        GGUFConfigs().set_config("n_ctx", n_ctx)

        self.temperature=float(temperature)
        self.top_k=top_k
        self.max_tokens=max_tokens
        self.instruct = instruct
        self.n_ctx = n_ctx
        self.model_name = model
        self.model_type = self.g.settings_data['model_list'][self.model_name]['type']
        if self.model_type == 'deep_link':
            self.hf_repo_name = self.g.settings_data['model_list'][self.model_name]['repo_name']
            self.model_file = self.g.settings_data['model_list'][self.model_name]['file']
        else:
            self.hf_repo_name = None
            self.model_file = None


        del self.prompter


        # delete the model from Ram
        gc.collect()

        self.set_pipeline()
        return f'Model set to {model}'

    def log(self,logfile, text):
        f = open(logfile, 'a')
        f.write(f"{text}\n")
        f.close()

    def set_prompt(self,prompt_text):
        self.prompt_template = prompt_text

        self.log('magic_prompt_logfile.txt',f"Magic Prompt: \n{prompt_text} \n")


        self.prompt_dict["blurb1"] = prompt_text

        del self.prompter

        # delete the model from Ram
        gc.collect()

        self.set_pipeline()
        return f'Magic Prompt set to:\n {prompt_text}'


    def set_pipeline(self):

        self.prompter = Prompt()

        if self.model_type == 'deep_link':
            self.prompter.model_catalog.register_gguf_model(self.model_name,
                                                            self.hf_repo_name,
                                                            self.model_file,
                                                            prompt_wrapper="open_chat",
                                                            context_window=self.n_ctx)

        self.prompter.load_model(self.model_name)
        self.prompter.pc.add_custom_prompt_card("image_prompt",
                                                self.run_order_list,
                                                self.prompt_dict,
                                                prompt_description="Image Gen Search")

        #  the temperatures are from 0-1, and lower number is closer to the text and reduces hallucinations
        self.prompter = self.prompter.set_inference_parameters(temperature=self.temperature,
                                                               llm_max_output_len=self.max_tokens)

        self.query =  Query(self.lib)

