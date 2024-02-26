import datetime

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter

from pathlib import Path


document_store = QdrantDocumentStore(
    url="http://192.168.0.127:6333",
    index='haystack_prompt_out_all',
    return_embedding=True,
    wait_result_from_api=True,

)


p = Pipeline()
p.add_component(instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]), name="file_type_router")
p.add_component(instance=TextFileToDocument(), name="text_file_converter")
p.add_component(instance=DocumentJoiner(), name="joiner")
p.add_component(instance=DocumentCleaner(), name="cleaner")
p.add_component(instance=DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30), name="splitter")
p.add_component(
    instance=SentenceTransformersDocumentEmbedder(model="BAAI/bge-base-en-v1.5",device='cuda'), name="embedder"
)

p.add_component("writer", DocumentWriter(document_store=document_store))

p.connect("file_type_router.text/plain", "text_file_converter.sources")
p.connect("text_file_converter.documents", "joiner.documents")
p.connect("joiner.documents", "cleaner.documents")
p.connect("cleaner.documents", "splitter.documents")
p.connect("splitter.documents", "embedder.documents")
p.connect("embedder.documents", "writer.documents")


import os
sample_files_path = "E:\prompt_sources\lexica_split"
for subdir, dirs, files in os.walk(sample_files_path):
    if len(files) > 0:
        now = datetime.datetime.now()
        print(f'{now.strftime("%H:%M:%S")} adding folder: {subdir}')
        result = p.run({"file_type_router": {"sources": list(Path(subdir).iterdir())}},debug=True)

