# main.py - Updated with agent tracking integration and timeout fixes

import requests
import time
from paper_fetcher import PaperFetcher, ResearchPaper
from typing import List
from tracker_integration import get_tracker, get_calc

OLLAMA_API_URL = "http://localhost:11434/api/generate"


def research_topic(topic: str, skip_tools: bool = False, fetch_papers: bool = True, 
                   max_papers: int = 5, timeout: int = 600) -> str:
    """
    Research a topic with agent tracking and improved timeout handling.
    
    Args:
        topic: Research topic or query
        skip_tools: Skip paper fetching (for PDF chunk processing)
        fetch_papers: Whether to fetch papers from external sources
        max_papers: Maximum number of papers to fetch
        timeout: Timeout in seconds for LLM calls (default: 600 = 10 minutes)
    """
    tracker = get_tracker()
    calc = get_calc()
    
    # Skip paper fetching for chunk summarization
    if skip_tools or not fetch_papers:
        return _generate_summary_only(topic, timeout=timeout)
    
    print(f"\n{'='*70}")
    print(f"RESEARCHING TOPIC: {topic}")
    print(f"{'='*70}\n")
    
    # LOG ACTION: Start research
    start_time = time.time()
    tracker.log_action("research_topic", 
                      topic=topic[:50], 
                      max_papers=max_papers,
                      fetch_papers=fetch_papers)
    
    # Step 1: Fetch research papers
    print("Step 1: Fetching research papers...")
    fetcher = PaperFetcher()
    
    try:
        # LOG ACTION: Fetch papers
        fetch_start = time.time()
        tracker.log_action("fetch_papers", 
                          query=topic[:50],
                          max_results=max_papers)
        
        papers = fetcher.search_papers(
            query=topic,
            max_results=max_papers,
            sources=['arxiv', 'semantic_scholar']
        )
        
        fetch_duration = time.time() - fetch_start
        
        if not papers:
            print("No papers found, generating summary from LLM knowledge...")
            tracker.add_reward(-3, "No papers found")
            tracker.add_reward(calc.task_completion(False), "Paper fetch failed")
            return _generate_summary_only(topic, timeout=timeout)
        
        print(f"Retrieved {len(papers)} papers\n")
        
        # REWARD: Papers found successfully
        tracker.add_reward(calc.task_completion(True), f"Found {len(papers)} papers")
        tracker.add_reward(calc.response_time(fetch_duration, 5.0), 
                          f"Fetch time: {fetch_duration:.2f}s")
        
        # Quality bonus for highly cited papers
        total_citations = sum(p.citations for p in papers)
        if total_citations > 100:
            tracker.add_reward(5, f"High-quality papers ({total_citations} citations)")
        elif total_citations > 50:
            tracker.add_reward(3, f"Good papers ({total_citations} citations)")
        
    except Exception as e:
        print(f"Error fetching papers: {e}")
        tracker.add_reward(calc.error_penalty(), f"Fetch error: {str(e)}")
        print("Falling back to LLM-only summary...")
        return _generate_summary_only(topic, timeout=timeout)
    
    # Step 2: Build context from papers
    print("Step 2: Processing paper abstracts...")
    context_start = time.time()
    tracker.log_action("build_context", num_papers=len(papers))
    
    context = _build_research_context(papers)
    context_duration = time.time() - context_start
    
    tracker.add_reward(calc.task_completion(True), "Context built")
    tracker.add_reward(calc.response_time(context_duration, 2.0), 
                      f"Context time: {context_duration:.2f}s")
    
    # Step 3: Generate comprehensive summary
    print("Step 3: Generating comprehensive analysis...\n")
    summary_start = time.time()
    tracker.log_action("generate_summary", 
                      papers_count=len(papers),
                      context_length=len(context))
    
    summary = _generate_research_summary(topic, papers, context, timeout=timeout)
    summary_duration = time.time() - summary_start
    
    # REWARD: Summary generation
    if summary and len(summary) > 500:
        tracker.add_reward(calc.task_completion(True), "Summary generated")
        tracker.add_reward(calc.quality_score(0.8), 
                          f"Summary length: {len(summary)} chars")
        tracker.add_reward(calc.response_time(summary_duration, 10.0),
                          f"Generation time: {summary_duration:.2f}s")
    else:
        tracker.add_reward(-5, "Summary too short or empty")
    
    # Total workflow reward
    total_duration = time.time() - start_time
    tracker.add_reward(10, f"Research workflow completed in {total_duration:.1f}s")
    
    return summary


def _build_research_context(papers: List[ResearchPaper]) -> str:
    """Build research context from papers"""
    
    context_parts = []
    
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        
        context_parts.append(f"""
[Paper {i}] {paper.title}
Authors: {authors}
Year: {paper.year}
Source: {paper.source}
Citations: {paper.citations if paper.citations > 0 else 'N/A'}
Abstract: {paper.abstract}
""".strip())
    
    return "\n\n---\n\n".join(context_parts)


def _generate_research_summary(topic: str, papers: List[ResearchPaper], 
                               context: str, timeout: int = 600) -> str:
    """Generate comprehensive research summary using LLM with timeout handling"""
    tracker = get_tracker()
    calc = get_calc()
    
    # Build paper list for reference
    paper_list = "\n".join([
        f"{i}. {paper.title} ({paper.year}) - {paper.authors[0]} et al."
        for i, paper in enumerate(papers, 1)
    ])
    
    prompt = f"""You are Athena, an expert AI research assistant. Analyze these recent research papers on "{topic}" and provide a comprehensive summary.

RESEARCH PAPERS:
{context}

Your task:
1. **Overview**: Provide a clear introduction to "{topic}" based on these papers
2. **Key Findings**: Summarize the main contributions and findings from each paper
3. **Common Themes**: Identify patterns, shared methodologies, or consensus across papers
4. **Recent Advances**: Highlight what's new or cutting-edge in this area
5. **Challenges & Future Work**: Discuss open problems and research directions mentioned
6. **Practical Impact**: Explain real-world applications and implications

Be specific and reference the papers by number [Paper 1], [Paper 2], etc.
Write in an academic yet accessible style. Aim for 800-1000 words.

PAPER REFERENCES:
{paper_list}

COMPREHENSIVE RESEARCH SUMMARY:"""

    try:
        # LOG ACTION: Call LLM
        llm_start = time.time()
        tracker.log_action("call_ollama", 
                          model="llama3",
                          prompt_length=len(prompt),
                          max_tokens=1500)
        
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 1500,
                "num_ctx": 8192
            }
        }
        
        print("   Calling Ollama API...")
        
        # Increased timeout with better error handling
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        
        llm_duration = time.time() - llm_start
        
        if response.status_code != 200:
            print(f"   API Error: {response.status_code}")
            tracker.add_reward(calc.error_penalty(), 
                             f"Ollama API error: {response.status_code}")
            return _fallback_summary(papers)
        
        data = response.json()
        summary = data.get("response", "").strip()
        
        if not summary:
            print("   Empty response from LLM")
            tracker.add_reward(-3, "Empty LLM response")
            return _fallback_summary(papers)
        
        # REWARD: LLM call success
        tracker.add_reward(calc.task_completion(True), "LLM generated response")
        tracker.add_reward(calc.response_time(llm_duration, 30.0),
                          f"LLM time: {llm_duration:.2f}s")
        tracker.add_reward(calc.quality_score(0.85), 
                          f"Response length: {len(summary)} chars")
        
        # Add paper references at the end
        full_summary = f"{summary}\n\n{'='*70}\n\n## SOURCE PAPERS\n\n"
        
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors += f" et al. ({len(paper.authors)} authors)"
            
            full_summary += f"\n**[Paper {i}]** {paper.title}\n"
            full_summary += f"- **Authors:** {authors}\n"
            full_summary += f"- **Year:** {paper.year} | **Source:** {paper.source}"
            
            if paper.citations > 0:
                full_summary += f" | **Citations:** {paper.citations}"
            
            full_summary += f"\n- **Link:** {paper.url}\n"
            
            if paper.pdf_url:
                full_summary += f"- **PDF:** {paper.pdf_url}\n"
        
        print("   Summary generated\n")
        return full_summary
        
    except requests.exceptions.Timeout:
        print(f"   Request timed out after {timeout}s")
        tracker.add_reward(calc.error_penalty(), "LLM timeout")
        
        print("   Suggestions:")
        print("      - Try a shorter query")
        print("      - Reduce max_papers parameter")
        print("      - Use smaller model: ollama pull llama3.2:1b")
        print("      - Pre-load model: ollama run llama3 'test'")
        
        return _fallback_summary(papers)
    except requests.exceptions.ConnectionError as e:
        print(f"   Connection Error: Cannot connect to Ollama")
        tracker.add_reward(calc.error_penalty(), "Ollama connection error")
        print("   Make sure Ollama is running (check system tray)")
        return _fallback_summary(papers)
    except Exception as e:
        print(f"   Error: {e}")
        tracker.add_reward(calc.error_penalty(), f"LLM error: {str(e)}")
        return _fallback_summary(papers)


def _generate_summary_only(topic: str, timeout: int = 480) -> str:
    """Generate summary using only LLM knowledge (no paper fetching) with timeout handling"""
    tracker = get_tracker()
    calc = get_calc()
    
    start = time.time()
    tracker.log_action("generate_summary_only", topic=topic[:50])
    
    prompt = f"""You are Athena, an expert AI research assistant. Provide a comprehensive overview of: "{topic}"

Include:
1. **Definition & Core Concepts**: What is this topic about?
2. **Key Methods & Techniques**: Main approaches used in this area
3. **Important Milestones**: Historical development and breakthroughs
4. **Current State**: What's the current state of research/practice?
5. **Applications**: Real-world use cases and impact
6. **Challenges**: Open problems and limitations
7. **Future Directions**: Where is this field heading?

Be specific, technical yet accessible. Aim for 600-800 words."""

    try:
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 1200
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("response", "Error generating summary").strip()
            
            # REWARD
            tracker.add_reward(calc.task_completion(True), "Summary generated")
            tracker.add_reward(calc.response_time(duration, 20.0),
                             f"Time: {duration:.2f}s")
            
            return result
        else:
            tracker.add_reward(calc.error_penalty(), 
                             f"API error: {response.status_code}")
            return f"Error: API returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout}s")
        tracker.add_reward(calc.error_penalty(), "Timeout")
        
        return f"""Error: Request timed out after {timeout} seconds.

Suggestions:
- Try a shorter or simpler query
- Pre-load the model: ollama run llama3 "test"
- Use a smaller/faster model: ollama pull llama3.2:1b
- Check system resources (CPU/RAM usage)
- Increase timeout in the code if needed

Ollama might be loading the model for the first time, which can take 1-3 minutes."""
        
    except requests.exceptions.ConnectionError:
        tracker.add_reward(calc.error_penalty(), "Connection error")
        return """Error: Cannot connect to Ollama.

Please ensure:
1. Ollama is installed (https://ollama.com/download)
2. Ollama is running (check system tray for Ollama icon)
3. The API is accessible at http://localhost:11434

To start Ollama manually, try:
- Open Ollama from Start Menu
- Or run: ollama serve (if in PATH)"""
        
    except Exception as e:
        tracker.add_reward(calc.error_penalty(), str(e))
        return f"Error generating summary: {str(e)}"


def _fallback_summary(papers: List[ResearchPaper]) -> str:
    """Fallback summary when LLM fails"""
    
    summary = "# Research Paper Summary\n\n"
    summary += f"Retrieved {len(papers)} relevant research papers:\n\n"
    summary += "---\n\n"
    
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        
        summary += f"## {i}. {paper.title}\n\n"
        summary += f"**Authors:** {authors}\n"
        summary += f"**Year:** {paper.year} | **Source:** {paper.source}"
        
        if paper.citations > 0:
            summary += f" | **Citations:** {paper.citations}"
        
        summary += f"\n\n**Abstract:**\n{paper.abstract}\n\n"
        summary += f"**Links:** [View Paper]({paper.url})"
        
        if paper.pdf_url:
            summary += f" | [PDF]({paper.pdf_url})"
        
        summary += "\n\n---\n\n"
    
    return summary


# Test and utility functions
def test_ollama_connection():
    """Test if Ollama is running and llama3 is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            print("Ollama is running")
            print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
            
            if any('llama3' in name for name in model_names):
                print("llama3 model is available")
                return True
            else:
                print("llama3 model not found. Run: ollama pull llama3")
                return False
        else:
            print(f"Ollama returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama - is it running?")
        print("Check system tray for Ollama icon or start from Start Menu")
        return False
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print(" TESTING ENHANCED RESEARCH FUNCTION WITH TRACKING")
    print("=" * 70)
    
    # First test Ollama connection
    print("\n Testing Ollama connection...\n")
    if not test_ollama_connection():
        print("\n Please start Ollama before running tests")
        exit(1)
    
    print("\n" + "=" * 70)
    
    test_topic = "transformer attention mechanisms in NLP"
    
    print(f"\n Testing with: '{test_topic}'\n")
    
    result = research_topic(
        topic=test_topic,
        fetch_papers=True,
        max_papers=5,
        timeout=300  # 5 minutes for testing
    )
    
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    print(result[:1000] + "...\n")
    
    # Show tracker state
    tracker = get_tracker()
    tracker.display_state()
    
    print("\n TEST COMPLETE!")