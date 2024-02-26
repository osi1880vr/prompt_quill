import datetime
import os


from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.storage.storage_context import StorageContext

vector_store = MilvusVectorStore(
    uri = "http://192.168.0.127:19530",
    port = 19530   ,
    collection_name = 'llama_index_prompts_large',
    dim = 384,
    similarity_metric = "L2",
    #text_key="paragraph",
    #overwrite=True
)

sample_files_path = "E:\short_large"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")

storage_context = StorageContext.from_defaults(vector_store=vector_store)


for subdir, dirs, files in os.walk(sample_files_path):
    if len(files) > 0:
        now = datetime.datetime.now()
        print(f'{now.strftime("%H:%M:%S")} adding folder: {subdir}')

        documents = SimpleDirectoryReader(subdir).load_data()


        # here we set the file_path to become no part of the embedding, its not for this usecase
        # also we check if a doc has zero content then we dont try to embedd it as it would result in an error
        docs = []
        for doc in documents:
            doc.excluded_llm_metadata_keys.append("file_path")
            doc.excluded_embed_metadata_keys.append("file_path")
            if doc.text != '':
                docs = docs + [doc]

        del documents

        vector_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context,embed_model=embed_model, show_progress=True)

        #vector_store.collection.flush() # seems to be not needed





