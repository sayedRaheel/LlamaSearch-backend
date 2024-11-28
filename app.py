# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from agents import Orchestrator
from config import Config

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": Config.CORS_ORIGINS}})

# Global orchestrator instance
orchestrator = None

def get_orchestrator(custom_key=None):
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None or custom_key:
        groq_key = Config.get_groq_key(custom_key)
        orchestrator = Orchestrator(groq_api_key=groq_key)
    return orchestrator

@app.route('/api/initialize', methods=['POST'])
def initialize_api():
    """Initialize with custom API key if provided"""
    try:
        data = request.get_json()
        custom_key = data.get('groq_key')
        orchestrator = get_orchestrator(custom_key)
        return jsonify({
            'status': 'success',
            'message': 'API initialized successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint"""
    try:
        orchestrator = get_orchestrator()
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query is required'
            }), 400

        response = orchestrator.process_query(query)

        if isinstance(response, str):
            return jsonify({
                'status': 'success',
                'content': response,
            })
        
        return jsonify({
            'status': 'success',
            'content': response
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    port = Config.PORT
    debug = Config.DEBUG
    app.run(host='0.0.0.0', port=port, debug=debug)