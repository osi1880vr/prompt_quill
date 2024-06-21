import time
import threading
import json
from flask import Flask, request, jsonify
from api.sailing import api_sail
from api.telegram_bot import Telegram
from llm_fw import llm_interface_qdrant

app = Flask(__name__)

sail_api = api_sail()
telegram_api = Telegram()


interface = llm_interface_qdrant.get_interface()
telegram_api.interface = interface

@app.route('/get_image_telegram', methods=['POST'])
def get_image_telegram():
    try:
        data = request.json
        return json.dumps(telegram_api.get_image(data))
    except:
        return jsonify({'error': 'Invalid JSON format'}), 400


@app.route('/get_prompt_telegram', methods=['POST'])
def get_prompt_telegram():
    try:
        data = request.json
        return json.dumps(telegram_api.get_prompt(data))
    except:
        return jsonify({'error': 'Invalid JSON format'}), 400


@app.route('/get_prompt', methods=['POST'])
def get_prompt():
    try:
        data = request.json
        return json.dumps(interface.run_api_llm_response(data['query']))
    except:
        return jsonify({'error': 'Invalid JSON format'}), 400


@app.route('/get_next_prompt', methods=['POST'])
def get_next_prompt():
    try:
        data = request.json
        return json.dumps(sail_api.run_api_sail(data))
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400



def web():
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=64738)


def run_api():
    threading.Thread(target=web, daemon=True).start()


