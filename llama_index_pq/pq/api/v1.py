import time
import threading
import json
from flask import Flask, request, jsonify


from llm_fw import llm_interface_qdrant
interface = llm_interface_qdrant.get_interface()
app = Flask(__name__)






@app.route('/get_prompt', methods=['POST'])
def hello_world():
    try:
        data = request.json
        return json.dumps(interface.run_api_llm_response(data['query']))
    except:
        return jsonify({'error': 'Invalid JSON format'}), 400



def web():
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=64738)


def run_api():
    threading.Thread(target=web, daemon=True).start()


