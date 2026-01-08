# advanced_rag.py - Advanced RAG with Multi-Document Reasoning

import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_class = HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings_class = SentenceTransformerEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """Document metadata and content"""
    id: str
    title: str
    content: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedRAG:
    """
    Advanced RAG system with:
    - Multi-document reasoning
    - Contextual chunk retrieval
    - Cross-document comparison
    - Source attribution
    - Confidence scoring
    """
    
    def __init__(self, model="llama3", chunk_size=800, chunk_overlap=100):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        try:
            self.embeddings = embeddings_class(model_name="all-MiniLM-L6-v2")
        except:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Document store
        self.documents: Dict[str, Document] = {}
        self.vectorstores: Dict[str, FAISS] = {}
        self.global_vectorstore = None
        
        print("✅ Advanced RAG initialized")
    
    def add_document(self, doc_id: str, title: str, content: str, metadata: Dict = None):
        """Add document to the RAG system"""
        doc = Document(
            id=doc_id,
            title=title,
            content=content,
            metadata=metadata or {}
        )
        
        self.documents[doc_id] = doc
        
        # Create document-specific vectorstore
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        chunks = splitter.split_text(content)
        
        # Add metadata to each chunk
        metadatas = [
            {
                'doc_id': doc_id,
                'doc_title': title,
                'chunk_index': i,
                **doc.metadata
            }
            for i in range(len(chunks))
        ]
        
        self.vectorstores[doc_id] = FAISS.from_texts(
            chunks,
            self.embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ Added document: {title} ({len(chunks)} chunks)")
        
        # Rebuild global index
        self._rebuild_global_index()
    
    def _rebuild_global_index(self):
        """Rebuild global index from all documents"""
        all_texts = []
        all_metadatas = []
        
        for doc_id, doc in self.documents.items():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_text(doc.content)
            
            all_texts.extend(chunks)
            all_metadatas.extend([
                {
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'chunk_index': i,
                    **doc.metadata
                }
                for i in range(len(chunks))
            ])
        
        if all_texts:
            self.global_vectorstore = FAISS.from_texts(
                all_texts,
                self.embeddings,
                metadatas=all_metadatas
            )
            print(f"✅ Global index rebuilt: {len(all_texts)} total chunks")
    
    def retrieve_context(self, query: str, k: int = 5, doc_ids: List[str] = None) -> List[Tuple]:
        """
        Retrieve relevant context with metadata
        Returns: [(text, metadata, similarity_score), ...]
        """
        if doc_ids:
            # Search specific documents
            all_results = []
            for doc_id in doc_ids:
                if doc_id in self.vectorstores:
                    results = self.vectorstores[doc_id].similarity_search_with_score(query, k=k)
                    all_results.extend(results)
            
            # Sort by score and take top k
            all_results.sort(key=lambda x: x[1])
            results = all_results[:k]
        else:
            # Search all documents
            if not self.global_vectorstore:
                return []
            results = self.global_vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results with similarity scores
        formatted = []
        for doc, distance in results:
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            formatted.append((
                doc.page_content,
                doc.metadata,
                similarity
            ))
        
        return formatted
    
    def answer_with_context(self, query: str, k: int = 5, doc_ids: List[str] = None) -> Dict:
        """
        Answer query with retrieved context and source attribution
        """
        # Retrieve context
        contexts = self.retrieve_context(query, k=k, doc_ids=doc_ids)
        
        if not contexts:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context string with sources
        context_parts = []
        sources_info = []
        
        for i, (text, metadata, similarity) in enumerate(contexts, 1):
            doc_title = metadata.get('doc_title', 'Unknown')
            context_parts.append(f"[Source {i} - {doc_title}]:\n{text}")
            
            sources_info.append({
                'source_id': i,
                'doc_id': metadata.get('doc_id'),
                'doc_title': doc_title,
                'similarity': similarity,
                'chunk_index': metadata.get('chunk_index', 0)
            })
        
        context_str = "\n\n".join(context_parts)
        
        # Generate answer with LLM
        prompt = f"""You are Athena, an AI research assistant. Answer the question based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
1. Use information ONLY from the provided sources
2. Cite sources using [Source X] notation
3. If sources don't contain the answer, say so clearly
4. Be specific and reference document titles when relevant

CONTEXT:
{context_str}

QUESTION: {query}

ANSWER (with source citations):"""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 600
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "").strip()
                
                # Calculate confidence based on context relevance
                avg_similarity = np.mean([s[2] for s in contexts])
                confidence = min(avg_similarity * 1.2, 1.0)  # Boosted slightly
                
                return {
                    'answer': answer,
                    'sources': sources_info,
                    'confidence': confidence,
                    'num_sources_used': len(contexts)
                }
            else:
                return {
                    'answer': f"Error: LLM returned status {response.status_code}",
                    'sources': sources_info,
                    'confidence': 0.0
                }
                
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': sources_info,
                'confidence': 0.0
            }
    
    def compare_documents(self, query: str, doc_ids: List[str], k: int = 3) -> Dict:
        """
        Compare multiple documents on a specific topic
        """
        if len(doc_ids) < 2:
            return {'error': 'Need at least 2 documents to compare'}
        
        # Retrieve context from each document
        doc_contexts = {}
        for doc_id in doc_ids:
            contexts = self.retrieve_context(query, k=k, doc_ids=[doc_id])
            doc_contexts[doc_id] = contexts
        
        # Build comparison prompt
        comparison_parts = []
        for doc_id, contexts in doc_contexts.items():
            doc = self.documents[doc_id]
            context_text = "\n".join([c[0] for c in contexts[:2]])  # Top 2 chunks
            comparison_parts.append(f"**{doc.title}**:\n{context_text}")
        
        comparison_str = "\n\n---\n\n".join(comparison_parts)
        
        prompt = f"""Compare how these research papers address the following topic: {query}

PAPERS:
{comparison_str}

Provide a structured comparison:

1. **Common Ground**: What approaches or findings do they share?
2. **Key Differences**: How do their methods or conclusions differ?
3. **Unique Contributions**: What does each paper uniquely contribute?
4. **Synthesis**: What can we learn by combining their insights?

Be specific and reference each paper by name."""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 800
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                comparison = data.get("response", "").strip()
                
                return {
                    'comparison': comparison,
                    'documents_compared': [self.documents[d].title for d in doc_ids],
                    'query': query
                }
            else:
                return {'error': f'LLM error: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def find_connections(self, concept: str, doc_ids: List[str] = None) -> Dict:
        """
        Find how a concept connects across documents
        """
        # Get relevant chunks from all documents
        contexts = self.retrieve_context(concept, k=10, doc_ids=doc_ids)
        
        # Group by document
        doc_mentions = defaultdict(list)
        for text, metadata, similarity in contexts:
            doc_id = metadata.get('doc_id')
            doc_title = metadata.get('doc_title', 'Unknown')
            doc_mentions[doc_title].append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'similarity': similarity
            })
        
        # Build connection map
        prompt = f"""Analyze how the concept "{concept}" appears across these research papers.

MENTIONS:
"""
        for doc_title, mentions in doc_mentions.items():
            prompt += f"\n**{doc_title}**:\n"
            for m in mentions[:2]:  # Top 2 per doc
                prompt += f"- {m['text']}\n"
        
        prompt += f"""

Provide:
1. **Core Definition**: How is "{concept}" fundamentally understood?
2. **Cross-Paper Analysis**: How do different papers approach this concept?
3. **Evolution/Trends**: Are there emerging patterns or shifts in understanding?
4. **Research Gaps**: What aspects need more investigation?"""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 700
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("response", "").strip()
                
                return {
                    'concept': concept,
                    'analysis': analysis,
                    'documents': list(doc_mentions.keys()),
                    'total_mentions': sum(len(m) for m in doc_mentions.values())
                }
            else:
                return {'error': f'LLM error: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_document_summary(self) -> Dict:
        """Get summary of loaded documents"""
        return {
            'total_documents': len(self.documents),
            'documents': [
                {
                    'id': doc_id,
                    'title': doc.title,
                    'length': len(doc.content),
                    'metadata': doc.metadata
                }
                for doc_id, doc in self.documents.items()
            ]
        }


# =====================================================================
# 🧪 TEST SUITE
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 ADVANCED RAG SYSTEM TEST")
    print("=" * 70)
    
    # Initialize
    rag = AdvancedRAG(chunk_size=600, chunk_overlap=100)
    
    # Sample documents
    doc1_content = """
    Title: Attention Is All You Need
    
    We propose the Transformer, a novel architecture based solely on attention 
    mechanisms, dispensing with recurrence entirely. The Transformer uses 
    multi-head self-attention to process sequential data. We evaluate on 
    machine translation tasks, achieving 28.4 BLEU on WMT 2014 English-German.
    
    The model consists of an encoder and decoder, each with 6 layers. Multi-head 
    attention allows the model to jointly attend to information from different 
    representation subspaces. We use positional encoding to inject sequence order.
    
    Results show the Transformer outperforms recurrent and convolutional models 
    on translation quality while being more parallelizable and requiring 
    significantly less time to train.
    """
    
    doc2_content = """
    Title: BERT: Pre-training of Deep Bidirectional Transformers
    
    We introduce BERT, a new language representation model using bidirectional 
    Transformers. Unlike previous models, BERT is designed to pre-train deep 
    bidirectional representations by jointly conditioning on both left and right 
    context in all layers.
    
    BERT uses masked language modeling (MLM) and next sentence prediction (NSP) 
    for pre-training. The model achieves state-of-the-art results on eleven NLP 
    tasks including question answering (SQuAD) and natural language inference.
    
    We demonstrate that pre-trained representations eliminate the need for many 
    heavily-engineered task-specific architectures. BERT obtains 93.2% accuracy 
    on SQuAD 1.1 and 83.1% F1 on SQuAD 2.0.
    """
    
    doc3_content = """
    Title: GPT-3: Language Models are Few-Shot Learners
    
    We train GPT-3, an autoregressive language model with 175 billion parameters, 
    and test its few-shot learning abilities. For all tasks, GPT-3 is applied 
    without any gradient updates or fine-tuning, with tasks and few-shot 
    demonstrations specified purely via text interaction.
    
    GPT-3 achieves strong performance on many NLP datasets, including translation, 
    question-answering, and cloze tasks. On some datasets it achieves competitive 
    performance with fine-tuned models despite using few-shot learning.
    
    We find that performance scales smoothly with model size. GPT-3 demonstrates 
    remarkable abilities for in-context learning, adapting to tasks from just a 
    few examples without parameter updates.
    """
    
    # Add documents
    print("\n📚 Adding documents...")
    rag.add_document("transformer", "Attention Is All You Need", doc1_content)
    rag.add_document("bert", "BERT Pre-training", doc2_content)
    rag.add_document("gpt3", "GPT-3 Few-Shot", doc3_content)
    
    # Test 1: Single query
    print("\n" + "=" * 70)
    print("TEST 1: Answer with Context")
    print("=" * 70)
    
    query = "How do these models handle sequential data?"
    print(f"\n❓ Query: {query}")
    
    result = rag.answer_with_context(query, k=4)
    
    print(f"\n💡 Answer (Confidence: {result['confidence']:.0%}):")
    print(result['answer'])
    
    print(f"\n📚 Sources ({result['num_sources_used']}):")
    for source in result['sources']:
        print(f"   [{source['source_id']}] {source['doc_title']} (similarity: {source['similarity']:.0%})")
    
    # Test 2: Document comparison
    print("\n" + "=" * 70)
    print("TEST 2: Document Comparison")
    print("=" * 70)
    
    comparison_query = "pre-training strategies"
    print(f"\n🔍 Comparing on: {comparison_query}")
    
    comp_result = rag.compare_documents(
        comparison_query,
        doc_ids=["transformer", "bert", "gpt3"],
        k=3
    )
    
    if 'comparison' in comp_result:
        print(f"\n📊 Comparison:")
        print(comp_result['comparison'][:500] + "...")
    
    # Test 3: Find connections
    print("\n" + "=" * 70)
    print("TEST 3: Concept Connections")
    print("=" * 70)
    
    concept = "attention mechanism"
    print(f"\n🔗 Tracing concept: {concept}")
    
    conn_result = rag.find_connections(concept)
    
    if 'analysis' in conn_result:
        print(f"\n🧠 Analysis:")
        print(f"   Found in {conn_result['total_mentions']} mentions across {len(conn_result['documents'])} papers")
        print(f"\n{conn_result['analysis'][:400]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SYSTEM SUMMARY")
    print("=" * 70)
    
    summary = rag.get_document_summary()
    print(f"\n📚 Loaded Documents: {summary['total_documents']}")
    for doc in summary['documents']:
        print(f"   - {doc['title']}: {doc['length']:,} characters")
    
    print("\n✅ ALL TESTS COMPLETED!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Multi-document retrieval")
    print("   ✅ Source attribution")
    print("   ✅ Confidence scoring")
    print("   ✅ Cross-document comparison")
    print("   ✅ Concept tracking")