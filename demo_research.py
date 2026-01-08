# demo_research.py - Demonstration of Enhanced Research Capabilities

import sys
from paper_fetcher import PaperFetcher
from main import research_topic


def demo_paper_fetching():
    """Demonstrate paper fetching"""
    print("=" * 80)
    print("🔬 DEMO: RESEARCH PAPER FETCHING")
    print("=" * 80)
    
    # Example topics
    topics = [
        "transformer attention mechanisms",
        "BERT language model",
        "computer vision CNN"
    ]
    
    print("\n📚 Available demo topics:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")
    
    # Get user choice
    print("\n" + "-" * 80)
    choice = input("Choose a topic (1-3) or enter your own: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(topics):
        selected_topic = topics[int(choice) - 1]
    elif choice:
        selected_topic = choice
    else:
        selected_topic = topics[0]
    
    print(f"\n✅ Selected: '{selected_topic}'")
    print("-" * 80)
    
    # Initialize fetcher
    fetcher = PaperFetcher()
    
    # Fetch papers
    print(f"\n🔍 Fetching papers on: '{selected_topic}'...")
    print("⏱️  This may take 5-15 seconds...\n")
    
    try:
        papers = fetcher.search_papers(
            query=selected_topic,
            max_results=5,
            sources=['arxiv', 'semantic_scholar']
        )
        
        if not papers:
            print("❌ No papers found. Try a different topic.")
            return
        
        # Display results
        print("\n" + "=" * 80)
        print(f"📊 RESULTS: Found {len(papers)} papers")
        print("=" * 80)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   📅 Year: {paper.year}")
            print(f"   👥 Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   📚 Source: {paper.source}")
            
            if paper.citations > 0:
                print(f"   📈 Citations: {paper.citations}")
            
            if paper.venue:
                print(f"   📍 Venue: {paper.venue}")
            
            print(f"   🔗 URL: {paper.url}")
            
            if paper.pdf_url:
                print(f"   📄 PDF: {paper.pdf_url}")
            
            print(f"\n   📝 Abstract:")
            abstract_preview = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
            print(f"   {abstract_preview}")
            print("   " + "-" * 76)
        
        # Ask about synthesis
        print("\n" + "=" * 80)
        synthesize = input("\n🧠 Generate LLM synthesis of these papers? (y/n): ").strip().lower()
        
        if synthesize == 'y':
            print("\n⏱️  Generating synthesis... (this may take 30-60 seconds)")
            print("🤖 Using local Ollama LLM...\n")
            
            result = research_topic(
                topic=selected_topic,
                fetch_papers=False,  # We already have papers
                max_papers=5
            )
            
            # Actually, let's use the papers we fetched
            from main import _build_research_context, _generate_research_summary
            context = _build_research_context(papers)
            summary = _generate_research_summary(selected_topic, papers, context)
            
            print("\n" + "=" * 80)
            print("📄 COMPREHENSIVE RESEARCH SUMMARY")
            print("=" * 80)
            print(summary)
            
            # Save option
            save = input("\n💾 Save summary to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"research_{'_'.join(selected_topic.split()[:3])}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"✅ Saved to: {filename}")
        
        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_full_research():
    """Demonstrate full research workflow"""
    print("=" * 80)
    print("🎯 DEMO: FULL RESEARCH WORKFLOW")
    print("=" * 80)
    
    topic = input("\n🔬 Enter research topic: ").strip()
    
    if not topic:
        print("❌ No topic provided")
        return
    
    max_papers = input("📊 Max papers to fetch (default 5): ").strip()
    max_papers = int(max_papers) if max_papers.isdigit() else 5
    
    print(f"\n✅ Starting research on: '{topic}'")
    print(f"📚 Fetching up to {max_papers} papers...")
    print("⏱️  This will take 30-90 seconds...\n")
    
    try:
        result = research_topic(
            topic=topic,
            fetch_papers=True,
            max_papers=max_papers
        )
        
        print("\n" + "=" * 80)
        print("📊 RESEARCH RESULTS")
        print("=" * 80)
        print(result)
        
        # Save option
        save = input("\n💾 Save results? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"research_{'_'.join(topic.split()[:3])}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"✅ Saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo menu"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    🧠 ATHENA RESEARCH ASSISTANT                           ║
║                     Enhanced Research Demo                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

Choose a demo:

1. 📚 Paper Fetching Demo
   - See how papers are retrieved from arXiv and Semantic Scholar
   - View paper metadata, abstracts, and links
   - Optional: Generate LLM synthesis

2. 🎯 Full Research Workflow
   - Enter any research topic
   - Fetch papers + generate comprehensive summary
   - Save results to file

3. 🧪 Quick Test
   - Run predefined test query
   - Fast demo of complete system

4. ❌ Exit
""")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        demo_paper_fetching()
    elif choice == '2':
        demo_full_research()
    elif choice == '3':
        print("\n🧪 Running quick test...\n")
        result = research_topic(
            topic="attention mechanisms in transformers",
            fetch_papers=True,
            max_papers=3
        )
        print("\n" + "=" * 80)
        print("📊 QUICK TEST RESULTS")
        print("=" * 80)
        print(result[:1000] + "...\n[truncated]")
    elif choice == '4':
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice")
    
    # Ask to continue
    print("\n" + "=" * 80)
    again = input("Run another demo? (y/n): ").strip().lower()
    if again == 'y':
        print("\n")
        main()
    else:
        print("\n👋 Thanks for trying Athena!")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                          🧠 ATHENA                                        ║
║                   AI Research Assistant                                   ║
║                                                                           ║
║  NEW FEATURES:                                                            ║
║  ✅ Fetch real research papers from arXiv & Semantic Scholar             ║
║  ✅ Automatic paper synthesis using local LLM                            ║
║  ✅ Citation tracking and paper metadata                                 ║
║  ✅ Direct links to papers and PDFs                                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Ollama
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama: Running")
        else:
            print("⚠️  Ollama: Error")
    except:
        print("❌ Ollama: Not running")
        print("   Please start: ollama serve\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()