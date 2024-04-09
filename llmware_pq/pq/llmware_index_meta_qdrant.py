# Copyright 2023 osiworx

# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

import os
host = 'localhost'


if os.getenv("QDRANT_HOST") is not None:
    host = os.getenv("QDRANT_HOST")


os.environ["USER_MANAGED_QDRANT_HOST"] = host
os.environ["USER_MANAGED_QDRANT_PORT"] = "6333"


from llmware.parsers import Parser
from llmware.library import Library
from llmware.status import Status
import gc




in_path = 'E:\prompt_sources\sfw_meta'

def rag (library_name):

    # Step 0 - Configuration - we will use these in Step 4 to install the embeddings
    #embedding_model = "industry-bert-contracts"
    embedding_model = 'mini-lm-sbert'
    vector_db = "qdrant"

    # Step 1 - Create library which is the main 'organizing construct' in llmware
    print ("\nupdate: Step 1 - Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)
    parser = Parser(library)

    # Step 3 - point ".add_files" method to the folder of documents that was just created
    #   this method parses all of the documents, text chunks, and captures in MongoDB
    print("update: Step 2 - Parsing and Text Indexing Files")

    sample_files_path = in_path

    metadata = {"text": -1, "model": 0, "negative_prompt": 1}
    columns = 3

    # for subdir, dirs, files in os.walk(sample_files_path):
    #     if len(files) > 0:
    #         for file in files:
    #             t0 = time.time()
    #             parser_output = parser.parse_delimiter_config(subdir, file, cols=columns, mapping_dict=metadata, delimiter="§")
    #             print(f"done parsing - time - {time.time() - t0} - summary - {parser_output}")

    library.install_new_embedding(embedding_model_name=embedding_model, vector_db=vector_db)

    gc.collect()
    # Step 4 - Install the embeddings
    print("\nupdate: Step 3 - Generating Embeddings in {} db - with Model- {}".format(vector_db, embedding_model))


    # note: for using llmware as part of a larger application, you can check the real-time status by polling Status()
    #   --both the EmbeddingHandler and Parsers write to Status() at intervals while processing
    update = Status().get_embedding_status(library_name, embedding_model)
    print("update: Embeddings Complete - Status() check at end of embedding - ", update)


if __name__ == "__main__":

    # pick any name for the library
    user_selected_name = "llmware_meta_qdrant"
    rag(user_selected_name)

