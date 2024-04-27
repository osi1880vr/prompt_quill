import time
import threading
import json
from flask import Flask

from llm_fw import llm_interface_qdrant
interface = llm_interface_qdrant.get_interface()
app = Flask(__name__)

@app.route('/get_prompt')
def hello_world():
    return json.dumps(interface.run_api_llm_response('a nice cat'))


def web():
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


def run_api():
    threading.Thread(target=web, daemon=True).start()


