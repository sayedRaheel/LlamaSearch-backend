importRenderRead os
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
    print("Initialize endpoint hit")  # Debug log
    try:
        data = request.get_json()
        print(f"Received data: {data}")  # Debug log
        custom_key = data.get('groq_key')
        orchestrator = get_orchestrator(custom_key)
        return jsonify({
            'status': 'success',
            'message': 'API initialized successfully'
        })
    except Exception as e:
        print(f"Initialize error: {str(e)}")  # Debug log
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint"""
    print("Search endpoint hit")  # Debug log
    try:
        orchestrator = get_orchestrator()
        data = request.get_json()
        print(f"Received search data: {data}")  # Debug log
        query = data.get('query')

        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query is required'
            }), 400

        print(f"Processing query: {query}")  # Debug log
        response = orchestrator.process_query(query)
        print(f"Got response: {response[:100]}...")  # Debug log

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
        print(f"Search error: {str(e)}")  # Debug log
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({'status': 'healthy'})

# Root endpoint (/) for API information
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'status': 'online',
        'version': '1.0',
        'endpoints': {
            'health': '/health [GET]',
            'search': '/api/search [POST]',
            'initialize': '/api/initialize [POST]'
        }
    })

# Keep your existing health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == "__main__":
    # Get port from environment variable first, then fallback to Config
    port = int(os.environ.get("PORT", Config.PORT))
    debug = os.environ.get("DEBUG", Config.DEBUG).lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)