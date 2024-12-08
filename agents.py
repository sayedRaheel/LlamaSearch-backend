# Standard library imports
import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

# Third-party imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchConfig:
    # Source authority weights
    SOURCE_AUTHORITY = {
        # Academic/Research
        'edu': 0.9,
        'acm.org': 0.9,
        'arxiv.org': 0.85,
        'ieee.org': 0.85,
        'technologyreview.com': 0.8,
        'nature.com': 0.8,
        'science.org': 0.8,
        'research.google': 0.75,
        'research.microsoft': 0.75,
        'forbes.com': 0.7,
        'reuters.com': 0.7,
        'bloomberg.com': 0.7,
        'techcrunch.com': 0.6,
        'wired.com': 0.6,
        'default': 0.3,
        'linkedin':0.8,
        'github': 0.8
    }

    # Minimum required sources for diverse responses
    MIN_SOURCES = 3
    MAX_SOURCES = 5

    # Domain categories for diversity
    # DOMAIN_CATEGORIES = {
    #     'academic': ['edu', 'arxiv.org', 'ieee.org'],
    #     'tech_major': ['technologyreview.com', 'wired.com'],
    #     'news': ['reuters.com', 'bloomberg.com', 'forbes.com'],
    #     'industry': ['research.google', 'research.microsoft']
    # }

    DOMAIN_CATEGORIES = {
    'academic': ['edu', 'arxiv.org', 'ieee.org', 'researchgate.net', 'sciencedirect.com', 'nature.com', 'scholar.google.com'],
    'tech_major': ['technologyreview.com', 'wired.com', 'techcrunch.com', 'arstechnica.com', 'theverge.com', 'venturebeat.com'],
    'news': ['reuters.com', 'bloomberg.com','linkedin', 'forbes.com', 'wsj.com', 'ft.com', 'economist.com', 'nytimes.com'],
    'industry': ['research.google', 'research.microsoft', 'research.facebook.com', 'labs.amazon.com', 'research.ibm.com', 'openai.com'],
    'tech_communities': ['github.com', 'stackoverflow.com', 'medium.com', 'dev.to', 'huggingface.co', 'kaggle.com'],
    'government': ['gov', 'nasa.gov', 'nih.gov', 'nsf.gov', 'europa.eu', 'who.int']
    }



# New Cell: DuckDuckGo Search Implementation
class DuckDuckGoSearch:
    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def web_search(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            params = {
                'q': query,
                'b': '',
                'kl': 'us-en'
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=params
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            for result in soup.select('.result'):
                title_elem = result.select_one('.result__title a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = title_elem.get('href')
                snippet = result.select_one('.result__snippet')
                description = snippet.get_text(strip=True) if snippet else ""

                results.append({
                    'title': title,
                    'link': link,
                    'snippet': description
                })

                if len(results) >= max_results:
                    break

            return results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def scholar_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Simulate scholar search using DDG with academic terms"""
        scholarly_query = f"site:scholar.google.com OR site:arxiv.org OR site:researchgate.net {query}"
        try:
            results = self.web_search(scholarly_query, max_results)
            processed_papers = []

            for result in results:
                processed_paper = {
                    'title': result['title'],
                    'link': result['link'],
                    'snippet': result['snippet'],
                    'publication_info': {},
                    'authors': '',
                    'year': self._extract_year(result['snippet']),
                    'citations': 0,
                    'abstract': result['snippet']
                }

                # Try to fetch abstract
                if processed_paper['link']:
                    try:
                        abstract = self._fetch_abstract(processed_paper['link'])
                        if abstract:
                            processed_paper['abstract'] = abstract
                    except:
                        pass

                processed_papers.append(processed_paper)

            return processed_papers

        except Exception as e:
            print(f"Scholar search error: {e}")
            return []

    def _extract_year(self, text: str) -> str:
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        return year_match.group(0) if year_match else ''

    def _fetch_abstract(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            abstract_selectors = [
                'abstract',
                'paper-abstract',
                'article-abstract',
                ['div', {'class': re.compile(r'abstract|summary', re.I)}],
                ['p', {'class': re.compile(r'abstract|summary', re.I)}]
            ]

            for selector in abstract_selectors:
                if isinstance(selector, list):
                    abstract = soup.find(selector[0], selector[1])
                else:
                    abstract = soup.find(selector)

                if abstract:
                    return abstract.get_text(strip=True)
            return None

        except Exception:
            return None

# Cell 4: Base Classes and Memory
# [Paste your ConversationMemory and BaseAgent classes here]
class ConversationMemory:
    """Manages conversation history and context"""
    def __init__(self, max_history: int = 2):
        self.messages = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]

    def get_context(self) -> List[Dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages = []

# class BaseAgent(ABC):
#     """Base class for all agents"""
#     def __init__(self, groq_api_key: str):
#         self.llm_client = Groq(api_key=groq_api_key)
#         self.memory = ConversationMemory()

#     @abstractmethod
#     def process(self, query: str, **kwargs) -> Dict:
#         pass
        
class BaseAgent(ABC):
    """Base class for all agents"""
    def __init__(self, groq_api_key: str):
        print(f"Initializing Groq client with key: {groq_api_key[:8]}...")  # Debug log
        try:
            import groq
            self.llm_client = groq.Groq(api_key=groq_api_key)
            print("Groq client initialized successfully")
        except Exception as e:
            print(f"Groq initialization error: {str(e)}")
            raise
        self.memory = ConversationMemory()

    @abstractmethod
    def process(self, query: str, **kwargs) -> Dict:
        pass

# Cell 5: Agent Classes
class QueryAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing and categorizing queries"""
    def process(self, query: str, **kwargs) -> Dict:
        system_prompt = """
        Analyze the search query and categorize it. Consider the conversation history for context.
        Return a JSON with:
        {
            "query_type": "research" or "general" or "shopping" or "news",
            "search_terms": "optimized search terms",
            "requires_research": boolean (true if academic sources needed),
            "domain": "specific field if research (e.g., medical, tech)",
            "context_relevance": "how this relates to previous questions"
        }
        """

        try:
            response = self.llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.memory.get_context(),
                    {"role": "user", "content": f"Current query: {query}"}
                ],
                model="llama3-70b-8192"
            )

            analysis = json.loads(response.choices[0].message.content)
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", json.dumps(analysis))
            return analysis
        except:
            return self._fallback_analysis(query)

    def _fallback_analysis(self, query: str) -> Dict:
        query_patterns = {
            'factual': r'^(what|who|when|where|why|how|tell me about|explain)',
            'research': r'(research|study|paper|scientific|academic)',
            'statistics': r'(how many|count|number of|statistics|data)',
            'shopping': r'(price|cost|buy|under \$|cheaper than)',
            'news': r'(latest|news|current|recent|2024|2023)'
        }

        query_type = 'general'
        for type_, pattern in query_patterns.items():
            if re.search(pattern, query.lower()):
                query_type = type_
                break

        return {
            "query_type": query_type,
            "search_terms": query,
            "requires_research": query_type == 'research',
            "domain": "general",
            "context_relevance": "no context available"
        }

class ResearchAgent(BaseAgent):
    """Agent for handling academic and research queries"""
    def __init__(self, groq_api_key: str, serpapi_key: str=None):
        super().__init__(groq_api_key)
        self.search_engine = DuckDuckGoSearch()

    def process(self, query: str, **kwargs) -> Dict:
        try:
            # Enhance query for academic results
            academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:researchgate.net OR site:semanticscholar.org OR site:github.com"

            # Get initial results
            results = self.search_engine.web_search(academic_query, max_results=10)

            processed_papers = []
            for result in results:
                try:
                    # Verify if it's a genuine research paper link
                    if self._is_valid_paper_source(result['link']):
                        paper_info = self._extract_paper_info(result)
                        if paper_info:
                            processed_papers.append(paper_info)
                except Exception as e:
                    print(f"Error processing paper: {e}")
                    continue

            return {
                'type': 'research',
                'papers': processed_papers,
                'total_results': len(processed_papers)
            }

        except Exception as e:
            print(f"Research search error: {e}")
            return None

    def _is_valid_paper_source(self, url: str) -> bool:
        """Verify if the URL is from a legitimate research source"""
        valid_domains = [
            'arxiv.org',
            'github.com',
            'researchgate.net',
            'semanticscholar.org',
            'scholar.google.com',
            'dl.acm.org',
            'ieee.org'
        ]
        return any(domain in url.lower() for domain in valid_domains)

    def _extract_paper_info(self, result: Dict) -> Optional[Dict]:
        """Extract and verify paper information"""
        try:
            title = result['title']
            link = result['link']
            snippet = result['snippet']

            # Only process if we have minimum required information
            if title and link:
                # Extract year if present in title or snippet
                year_match = re.search(r'(19|20)\d{2}', title + ' ' + snippet)
                year = year_match.group(0) if year_match else 'N/A'

                return {
                    'title': title,
                    'link': link,  # Actual, verifiable link
                    'snippet': snippet,
                    'year': year,
                    'source': self._extract_source_name(link)
                }
            return None

        except Exception:
            return None

    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        if 'arxiv.org' in url:
            return 'arXiv'
        elif 'github.com' in url:
            return 'GitHub'
        elif 'researchgate.net' in url:
            return 'ResearchGate'
        elif 'semanticscholar.org' in url:
            return 'Semantic Scholar'
        elif 'scholar.google.com' in url:
            return 'Google Scholar'
        elif 'dl.acm.org' in url:
            return 'ACM Digital Library'
        elif 'ieee.org' in url:
            return 'IEEE'
        else:
            return urlparse(url).netloc

print("Updated ResearchAgent loaded!")

class SearchAgent(BaseAgent):
    """Enhanced Agent for general web searches"""
    def __init__(self, groq_api_key: str, serpapi_key: str=None):
        super().__init__(groq_api_key)
        self.search_engine = DuckDuckGoSearch()
        self.config = SearchConfig()

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from webpage"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Try to find main content area
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_=re.compile(r'content|main|article', re.I))
        )

        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\r\n]+', '\n', text)
        return text.strip()

    def _get_source_authority(self, url: str) -> float:
        """Get authority score for a source"""
        for domain, score in self.config.SOURCE_AUTHORITY.items():
            if domain in url:
                return score
        return self.config.SOURCE_AUTHORITY['default']

    def _calculate_relevance(self, content: str, query: str, url: str) -> float:
        """Enhanced relevance calculation"""
        # Base relevance
        query_terms = query.lower().split()
        content_lower = content.lower()
        term_frequency = sum(content_lower.count(term) for term in query_terms)

        # Position scoring
        positions = [content_lower.find(term) for term in query_terms]
        position_score = sum(1/(pos+1) if pos != -1 else 0 for pos in positions)

        # Authority score
        authority_score = self._get_source_authority(url)

        # Combine scores
        base_score = term_frequency / (len(content.split()) + 1)
        final_score = (
            base_score * 0.4 +
            position_score * 0.3 +
            authority_score * 0.3
        )

        return final_score

    def _ensure_diversity(self, results: List[Dict]) -> List[Dict]:
        """Ensure source diversity"""
        selected_results = []
        category_count = defaultdict(int)

        for result in results:
            url = result['link']
            category = 'default'

            # Determine category
            for cat_name, domains in self.config.DOMAIN_CATEGORIES.items():
                if any(domain in url for domain in domains):
                    category = cat_name
                    break

            if category_count[category] < 2:  # Max 2 sources per category
                selected_results.append(result)
                category_count[category] += 1

            if len(selected_results) >= self.config.MIN_SOURCES:
                break

        return selected_results

    def process(self, query: str, **kwargs) -> Dict:
        try:
            # Get search results
            results = self.search_engine.web_search(query, max_results=10)

            processed_content = []
            for result in results:
                try:
                    response = requests.get(result['link'], timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    main_content = self._extract_main_content(soup)

                    score = self._calculate_relevance(
                        main_content,
                        query,
                        result['link']
                    )

                    processed_content.append({
                        'title': result['title'],
                        'link': result['link'],
                        'snippet': result['snippet'],
                        'content': main_content[:5000],
                        'score': score,
                        'authority': self._get_source_authority(result['link'])
                    })

                except requests.exceptions.RequestException:
                    continue

            if processed_content:
                # Sort by score
                processed_content.sort(key=lambda x: x['score'], reverse=True)
                # Ensure diversity
                diverse_results = self._ensure_diversity(processed_content)

                return {
                    'type': 'web_search',
                    'results': diverse_results,
                    'total_results': len(diverse_results)
                }
            return None

        except Exception as e:
            print(f"Search error: {e}")
            return None



# Check your Cell for Orchestrator class and update it to match this:
class Orchestrator:
    """Coordinates all agents"""
    def __init__(self, groq_api_key: str):
        self.query_analyzer = QueryAnalyzerAgent(groq_api_key)
        self.research_agent = ResearchAgent(groq_api_key, None)  # Removed serpapi_key
        self.search_agent = SearchAgent(groq_api_key, None)      # Removed serpapi_key
        self.groq_client = Groq(api_key=groq_api_key)
        self.memory = ConversationMemory()

    def process_query(self, query: str) -> str:  # This was missing or named differently
        self.memory.add_message("user", query)
        conversation_context = self.memory.get_context()

        # 1. Analyze query with context
        analysis = self.query_analyzer.process(query)

        # 2. Route to appropriate agent
        if analysis.get('requires_research', False):
            research_results = self.research_agent.process(query)
            search_results = self.search_agent.process(query)
            context = self._prepare_context(research_results, search_results)
        else:
            search_results = self.search_agent.process(query)
            context = self._prepare_context(None, search_results)

        # 3. Generate response
        response = self._generate_response(query, context, analysis, conversation_context)
        self.memory.add_message("assistant", response)
        return response



    def _prepare_context(self, research_results: Optional[Dict],
                        search_results: Optional[Dict]) -> str:
        context = []
        if research_results:
            context.append("Academic Sources:")
            for paper in research_results.get('papers', []):
                # Use .get() method with default values for all fields
                context.append(f"""
    Title: {paper.get('title', 'No title available')}
    Authors: {paper.get('authors', 'No authors listed')}
    Year: {paper.get('year', 'Year not specified')}
    Citations: {paper.get('citations', 'N/A')}
    Abstract: {paper.get('abstract', paper.get('snippet', 'No abstract available'))}
    Link: {paper.get('link', 'No link available')}
    """)

        if search_results:
            context.append("\nWeb Sources:")
            if isinstance(search_results, dict):
                if 'results' in search_results:  # For multiple results
                    for result in search_results['results']:
                        context.append(f"""
    Title: {result.get('title', 'No title')}
    Source: {result.get('link', 'No link')}
    Content: {result.get('content', result.get('snippet', 'No content available'))}
    """)
                else:  # For single result
                    context.append(f"""
    Title: {search_results.get('title', 'No title')}
    Source: {search_results.get('link', 'No link')}
    Content: {search_results.get('content', 'No content available')}
    """)

        return "\n".join(context)



    def _generate_response(self, query: str, context: str, analysis: Dict,
                        conversation_context: List[Dict]) -> str:
        try:
            if analysis.get('requires_research', False):
                system_prompt = f"""You are a research paper analyst focusing on academic sources.
    Query type: {analysis['query_type']}
    Domain: {analysis['domain']}

    STRICT RESEARCH PAPER RESPONSE FORMAT:
    1. First, list and number all available research papers as:
    Available Research Papers:
    [1] Title: [exact title]
        URL: [exact url]
        Year: [if available]
    [2] Title: [exact title]
        URL: [exact url]
        Year: [if available]

    2. Then provide a detailed analysis where:
      - ONLY cite the listed papers
      - Use numbered citations: Finding/Point [n] where n is the paper number
      - Make the citation number a hyperlink to the paper's URL
      - Focus on technical details and methodologies
      - No hallucinated references or papers
      Example: "The study demonstrates significant improvements in LLM performance [1]"


    3. Additional requirements:
   - Synthesize information from all provided sources
   - Reference relevant points from previous conversation
   - Maintain context and continuity
   - Do not hallucinate or invent citations
   - If unsure about a source detail, stick to verifiable information

    4. End with:
    Web Sources Referenced:
    - [Source Name]: [exact URL]


    Use these research papers to answer the query:
    {context}
    """
            else:
                system_prompt = f"""You are a web information analyst focusing on current sources.
    Query type: {analysis['query_type']}
    Domain: {analysis['domain']}

    STRICT OUTPUT FORMAT:

1. Start with listing available sources:
Available Sources:
[1] Source Name: [name]
    URL: [exact url]
[2] Source Name: [name]
    URL: [exact url]

2. Then create a comprehensive analysis following this structure:

**Sources Listed Above**

**Answer**
Begin with a concise overview paragraph introducing the topic and its significance.

Then provide detailed analysis where:
- Organize information into clear, titled sections
- Use **Bold Headers** for each main topic
- Format each section as:
  * Topic introduction
  * Detailed explanation
  * Practical applications or examples
  * Supporting evidence with citations

CITATION RULES:
- Use superscript numbers for citations [n]
- Place citations after relevant statements
- Multiple citations should be grouped [1][5]
- Citations should be clean and non-intrusive

FORMAT REQUIREMENTS:
- Use proper spacing between sections
- Keep paragraphs focused and concise
- Maintain consistent formatting
- Create a clear visual hierarchy
- End with a concluding paragraph

3. End with:
References:
[1] Source Name: exact_url
[2] Source Name: exact_url


    {context}
    """

            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_context,
                    {"role": "user", "content": query}
                ],
                model="llama-3.3-70b-versatile"
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

