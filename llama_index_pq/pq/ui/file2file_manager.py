# file2file_manager.py
import globals
import os
from settings.io import settings_io
from llm_fw import llm_interface_qdrant

class File2FileManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.settings_io = settings_io()
        self.interface = llm_interface_qdrant.get_interface()
        self.out_dir_t2t = os.path.join('../api_out', 'txt2txt')
        os.makedirs(self.out_dir_t2t, exist_ok=True)

    def run_batch(self, files):
        output = ''
        yield 'Batch started'
        for file in files:
            filename = os.path.basename(file)
            with open(file, 'r', encoding='utf8', errors='ignore') as f:
                file_content = f.readlines()
            outfile = os.path.join(self.out_dir_t2t, filename)
            with open(outfile, 'a', encoding='utf8', errors='ignore') as f:
                n = 0
                for query in file_content:
                    response = self.interface.run_llm_response_batch(query)
                    f.write(f'{response}\n')
                    output = f'{output}{response}\n{n} ---------\n'
                    n += 1
                    yield output

    def set_summary(self, summary):
        self.g.settings_data['summary'] = summary
        self.settings_io.write_settings(self.g.settings_data)