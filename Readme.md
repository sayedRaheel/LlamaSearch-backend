# LlamaSearch Backend 🦙

LlamaSearch is an advanced, real-time web and research search tool powered by a multi-agent AI system and Llama 70B model through Groq. It provides natural language search capabilities across both web content and academic research databases.

A Flask-based backend service that provides intelligent search capabilities using Groq LLM and web scraping.

## Features

- Intelligent query analysis and categorization
- Academic research paper search and analysis
- Web content search with source authority ranking
- Conversational memory for context-aware responses
- DuckDuckGo-based web scraping
- Custom user API key support

## Tech Stack

- Python 3.9+
- Flask
- Groq API
- BeautifulSoup4
- DuckDuckGo Search

## Setup

1. Clone the repository
```bash
git clone <repository-url>
cd backend
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set environment variables
```bash
export GROQ_API_KEY=your_groq_api_key
export PORT=10000
export DEBUG=true
```

4. Run locally
```bash
python app.py
```

## API Endpoints

### Initialize API
- POST `/api/initialize`
  - Initialize with custom Groq API key
  - Body: `{"groq_key": "your-api-key"}`

### Search
- POST `/api/search`
  - Perform intelligent search
  - Body: `{"query": "your search query"}`

### Health Check
- GET `/health`
  - Check API health status

## Deployment

The backend is configured for deployment on Render.com using render.yaml configuration.

## Architecture

- QueryAnalyzerAgent: Analyzes and categorizes search queries
- ResearchAgent: Handles academic and research paper searches
- SearchAgent: Performs general web searches with content analysis
- Orchestrator: Coordinates between agents and manages responses

MIT License
a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


