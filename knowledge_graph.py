# knowledge_graph.py - FIXED: Proper Research Entity Extraction

import re
import requests
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_class = HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings_class = SentenceTransformerEmbeddings


class KnowledgeGraphBuilder:
    """
    FIXED: Extracts meaningful research entities and relationships
    
    Entities:
    - Paper: Title and main contributions
    - Methods: Algorithms, techniques, approaches
    - Datasets: Training/evaluation data, subjects, samples
    - Models: Architectures, classifiers, neural networks
    - Metrics: Performance measures, accuracy, rates
    - Results: Performance numbers, outcomes
    - Hardware: Physical components, sensors
    - Software: Frameworks, tools, libraries
    
    Relationships:
    - uses, proposes, evaluates-on, achieves, based-on, detects, compares
    """
    
    def __init__(self, model="llama3"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.graph = nx.DiGraph()
        
        # FIXED: Research-focused entity patterns
        self.patterns = {
            'methods': [
                # General methods
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:detection|verification|estimation|classification|recognition|tracking|segmentation|analysis)\b',
                r'\b(MACE|SVM|CNN|RNN|LSTM|GAN|VAE|transformer|BERT|GPT|ResNet)\b',
                r'\b([a-z]+\s+(?:filter|classifier|detector|estimator|algorithm|method|approach|technique|model))\b',
                # Specific techniques
                r'\b(face\s+(?:detection|verification|recognition))\b',
                r'\b(speech\s+(?:detection|recognition))\b',
                r'\b(gaze\s+estimation)\b',
                r'\b(text\s+detection)\b',
                r'\b(user\s+(?:verification|authentication))\b',
                r'\b(covariance\s+features?)\b',
                r'\b(temporal\s+(?:features?|window|segmentation))\b',
            ],
            
            'datasets': [
                # Subject counts
                r'\b(\d+)\s+(?:subjects?|participants?|test\s*takers?|students?|users?)\b',
                r'\b(\d+)\s+(?:samples?|instances?|examples?|cases?)\b',
                r'\b(\d+)\s+(?:videos?|images?|frames?)\b',
                # Named datasets
                r'\b(ImageNet|COCO|MNIST|CIFAR|SQuAD|GLUE|WMT|OEP\s+dataset)\b',
                # Duration/size
                r'\b(\d+[,\d]*)\s+(?:seconds?|minutes?|hours?)\s+(?:of\s+)?(?:data|video|audio|cheating)\b',
            ],
            
            'metrics': [
                # Abbreviations
                r'\b(TDR|FAR|MAP|IoU|BLEU|ROUGE|AUC|ROC|F1|mAP)\b',
                # Full names with values
                r'\b(accuracy|precision|recall|specificity|sensitivity)\s*[:=]?\s*(\d+\.?\d*)%?\b',
                r'\b(detection\s+rate|false\s+alarm\s+rate|error\s+rate)\s*[:=]?\s*(\d+\.?\d*)%?\b',
                # Standalone
                r'\b(true\s+detection\s+rate|false\s+alarm\s+rate|peak[- ]to[- ]sidelobe\s+ratio)\b',
            ],
            
            'models': [
                # Classifiers
                r'\b((?:binary|multi[- ]class|two[- ]class)\s+(?:SVM|classifier))\b',
                r'\b((?:linear|RBF|polynomial)\s+(?:kernel|SVM))\b',
                # Architectures
                r'\b(ResNet|VGG|AlexNet|Inception|MobileNet|EfficientNet)[- ]?\d*\b',
                r'\b((?:convolutional|recurrent|feedforward)\s+neural\s+network)\b',
                # Filters
                r'\b(MACE\s+filter|Kalman\s+filter|particle\s+filter)\b',
            ],
            
            'results': [
                # Performance with values
                r'(\d+\.?\d*)%\s+(?:TDR|accuracy|precision|detection\s+rate)\b',
                r'(?:achieves?|obtains?|reaches?)\s+(\d+\.?\d*)%',
                r'(\d+\.?\d*)%\s+(?:FAR|false\s+alarm)',
                # Comparative results
                r'(?:better|worse|higher|lower|superior|inferior)\s+(?:than|to)',
            ],
            
            'hardware': [
                r'\b(webcam|camera|wearcam|microphone|sensor|GPU|CPU)\b',
                r'\b(NVIDIA|Intel|AMD)\s+\w+\b',
            ],
            
            'software': [
                r'\b(TensorFlow|PyTorch|Keras|OpenCV|scikit[- ]learn|FAISS|Ollama|LangChain)\b',
                r'\b(Python|C\+\+|MATLAB|Java)\b',
            ],
        }
        
        # FIXED: Research-focused relationships
        self.relation_patterns = {
            'uses': [r'use[ds]?', r'utiliz[es]{2,4}', r'employ[s]?', r'apply|applied|applies', r'leverage[s]?'],
            'proposes': [r'propose[s]?', r'introduce[s]?', r'present[s]?', r'design[s]?'],
            'achieves': [r'achieve[s]?', r'obtain[s]?', r'reach[es]{2,4}', r'attain[s]?', r'get[s]?'],
            'evaluates_on': [r'evaluat[es]{2,4}\s+on', r'test[es]{2,4}\s+on', r'valid at[es]{2,4}\s+on', r'benchmark[es]{2,4}\s+on'],
            'detects': [r'detect[s]?', r'recognize[s]?', r'identify|identifies', r'find[s]?'],
            'compares': [r'compare[s]?', r'contrast[s]?', r'versus', r'vs\.?', r'outperform[s]?'],
            'based_on': [r'based\s+on', r'built\s+on', r'extend[s]?', r'derived\s+from', r'inspired\s+by'],
        }
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """FIXED: Extract meaningful research entities"""
        entities = defaultdict(set)
        
        text_lower = text.lower()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get the full match or first captured group
                    entity = match.group(1) if match.groups() else match.group(0)
                    entity = entity.strip()
                    
                    # Filter out too short or too long entities
                    if 2 < len(entity) < 100:
                        # Clean up common noise
                        if not re.match(r'^[\d\s\.,]+$', entity):  # Not just numbers
                            entities[entity_type].add(entity)
        
        return dict(entities)
    
    def extract_title(self, text: str) -> str:
        """Extract paper title from text"""
        # Look for title patterns
        lines = text.split('\n')
        
        # Title often in first few lines, all caps or title case
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            # Skip very short lines
            if len(line) < 10:
                continue
            # Skip lines with keywords that indicate it's not a title
            if any(kw in line.lower() for kw in ['abstract', 'introduction', 'author', 'university', 'email']):
                continue
            # Title often has specific characteristics
            if line.isupper() or (line.istitle() and len(line) > 20):
                return line
        
        # Fallback: return filename or placeholder
        return "Research Paper"
    
    def extract_metrics_values(self, text: str) -> List[Dict]:
        """FIXED: Extract performance metrics with their values"""
        metrics_data = []
        
        # Pattern 1: "metric: value%" or "metric = value%"
        pattern1 = r'(TDR|FAR|accuracy|precision|recall|F1|BLEU|detection\s+rate)\s*[:=]\s*([\d.]+)%?'
        
        # Pattern 2: "value% metric"
        pattern2 = r'([\d.]+)%\s+(TDR|FAR|accuracy|precision|recall|detection)'
        
        # Pattern 3: "achieves value%"
        pattern3 = r'(?:achieve[s]?|obtain[s]?|reach[es]{2,4})\s+([\d.]+)%'
        
        for pattern in [pattern1, pattern2, pattern3]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        if re.match(r'[\d.]+', match.group(1)):
                            # Pattern 2
                            value = float(match.group(1))
                            metric = match.group(2)
                        else:
                            # Pattern 1
                            metric = match.group(1)
                            value = float(match.group(2))
                    else:
                        # Pattern 3
                        metric = "performance"
                        value = float(match.group(1))
                    
                    metrics_data.append({
                        'metric': metric,
                        'value': value,
                        'context': match.group(0)
                    })
                except (ValueError, IndexError):
                    continue
        
        return metrics_data
    
    def extract_relationships(self, text: str, entities: Dict) -> List[Tuple]:
        """FIXED: Extract meaningful relationships between entities"""
        relationships = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Find entities in this sentence
            sentence_entities = []
            for entity_type, entity_set in entities.items():
                for entity in entity_set:
                    if entity.lower() in sentence_lower:
                        sentence_entities.append((entity_type, entity))
            
            # Need at least 2 entities for a relationship
            if len(sentence_entities) >= 2:
                for relation_type, patterns in self.relation_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, sentence_lower):
                            # Create relationship
                            source = sentence_entities[0]
                            target = sentence_entities[1]
                            
                            relationships.append({
                                'source': source[1],
                                'source_type': source[0],
                                'relation': relation_type,
                                'target': target[1],
                                'target_type': target[0],
                                'context': sentence.strip()[:150]
                            })
                            break
        
        return relationships
    
    def build_graph(self, text: str, title: str = None) -> nx.DiGraph:
        """FIXED: Build meaningful knowledge graph from research paper"""
        
        if not title:
            title = self.extract_title(text)
        
        print(f"🔬 Building knowledge graph for: {title}")
        
        # Extract entities
        entities = self.extract_entities(text)
        
        print(f"   📊 Extracted entities:")
        total_entities = 0
        for entity_type, entity_set in entities.items():
            count = len(entity_set)
            total_entities += count
            print(f"      {entity_type}: {count} items")
            # Show first few for verification
            if count > 0:
                examples = list(entity_set)[:3]
                print(f"         Examples: {', '.join(examples)}")
        
        # Extract metrics
        metrics = self.extract_metrics_values(text)
        print(f"   📈 Found {len(metrics)} performance metrics")
        
        # Extract relationships
        relationships = self.extract_relationships(text, entities)
        print(f"   🔗 Found {len(relationships)} relationships")
        
        # Build graph
        self.graph = nx.DiGraph()
        
        # Add central paper node
        self.graph.add_node(title, type='paper', color='#FF6B6B')
        
        # Add entity nodes with proper colors
        node_colors = {
            'methods': '#4ECDC4',     # Cyan
            'datasets': '#FFE66D',    # Yellow
            'metrics': '#A8E6CF',     # Green
            'models': '#FF8B94',      # Pink
            'results': '#F38181',     # Light red
            'hardware': '#95E1D3',    # Mint
            'software': '#C7CEEA',    # Lavender
        }
        
        for entity_type, entity_set in entities.items():
            for entity in list(entity_set)[:50]:  # Limit to avoid overcrowding
                self.graph.add_node(
                    entity,
                    type=entity_type,
                    color=node_colors.get(entity_type, '#95E1D3')
                )
                # Connect to paper
                self.graph.add_edge(title, entity, relation='contains')
        
        # Add metric nodes with values
        for metric_data in metrics[:20]:  # Limit to top 20 metrics
            metric_label = f"{metric_data['metric']}: {metric_data['value']}"
            self.graph.add_node(
                metric_label,
                type='result',
                color='#F38181',
                value=metric_data['value']
            )
            self.graph.add_edge(title, metric_label, relation='achieves')
        
        # Add relationships
        for rel in relationships[:30]:  # Limit to avoid overcrowding
            if self.graph.has_node(rel['source']) and self.graph.has_node(rel['target']):
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    relation=rel['relation'],
                    context=rel['context']
                )
        
        print(f"   ✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        if self.graph.number_of_nodes() < 5:
            print(f"   ⚠️ WARNING: Very few nodes extracted. Paper might need custom patterns.")
        
        return self.graph
    
    def get_graph_summary(self) -> Dict:
        """Get summary statistics of the knowledge graph"""
        if not self.graph:
            return {}
        
        # Node type distribution
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        
        # Relation type distribution
        relation_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relation_types[data.get('relation', 'unknown')] += 1
        
        # Central nodes (highest degree)
        if self.graph.number_of_nodes() > 0:
            degrees = dict(self.graph.degree())
            central_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            central_nodes = []
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'relation_types': dict(relation_types),
            'central_nodes': central_nodes,
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0
        }
    
    def query_graph(self, query: str, k: int = 5) -> List[Dict]:
        """Query the knowledge graph"""
        query_lower = query.lower()
        results = []
        
        # Search nodes
        for node, data in self.graph.nodes(data=True):
            if query_lower in node.lower():
                neighbors = list(self.graph.neighbors(node))
                predecessors = list(self.graph.predecessors(node))
                
                results.append({
                    'node': node,
                    'type': data.get('type', 'unknown'),
                    'neighbors': neighbors[:3],
                    'connected_to': predecessors[:3],
                    'degree': self.graph.degree(node)
                })
        
        # Search edges
        for source, target, data in self.graph.edges(data=True):
            if query_lower in source.lower() or query_lower in target.lower():
                results.append({
                    'type': 'relationship',
                    'source': source,
                    'relation': data.get('relation', 'related'),
                    'target': target,
                    'context': data.get('context', '')
                })
        
        return results[:k]
    
    def export_to_cytoscape(self) -> Dict:
        """Export graph in Cytoscape.js format"""
        elements = []
        
        # Nodes
        for node, data in self.graph.nodes(data=True):
            elements.append({
                'data': {
                    'id': node,
                    'label': node,
                    'type': data.get('type', 'unknown'),
                    'color': data.get('color', '#95E1D3')
                }
            })
        
        # Edges
        for source, target, data in self.graph.edges(data=True):
            elements.append({
                'data': {
                    'source': source,
                    'target': target,
                    'relation': data.get('relation', 'related')
                }
            })
        
        return {'elements': elements}
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find all paths between two entities"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source=source,
                target=target,
                cutoff=max_length
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_subgraph(self, node: str, depth: int = 1) -> nx.DiGraph:
        """Get subgraph around a specific node"""
        if node not in self.graph:
            return nx.DiGraph()
        
        nodes = {node}
        current_level = {node}
        
        for _ in range(depth):
            next_level = set()
            for n in current_level:
                next_level.update(self.graph.neighbors(n))
                next_level.update(self.graph.predecessors(n))
            nodes.update(next_level)
            current_level = next_level
        
        return self.graph.subgraph(nodes).copy()


# Test with the proctoring paper
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 FIXED KNOWLEDGE GRAPH BUILDER TEST")
    print("=" * 70)
    
    # Sample from the proctoring paper
    sample_text = """
    Automated Online Exam Proctoring
    
    We present a multimedia analytics system that performs automatic online
    exam proctoring. The system includes six basic components: user verification,
    text detection, voice detection, active window detection, gaze estimation,
    and phone detection.
    
    The system hardware includes one webcam, one wearcam, and a microphone.
    We use MACE filter for face verification and SVM classifier for cheating
    detection with covariance features.
    
    To evaluate our proposed system, we collect data from 24 subjects performing
    various types of cheating. The system achieves 87% TDR at 2% FAR in segment-based
    metric. Text detection achieves 85.8% accuracy, speech detection 89.3%, and
    phone detection 100%.
    
    We use multi-class SVM with three pair-wise binary classifiers. The temporal
    window is 5 seconds with 80% overlap.
    """
    
    kg = KnowledgeGraphBuilder()
    graph = kg.build_graph(sample_text, "Automated Online Exam Proctoring")
    
    print("\n" + "=" * 70)
    print("📊 GRAPH SUMMARY")
    print("=" * 70)
    
    summary = kg.get_graph_summary()
    print(f"\n📈 Statistics:")
    print(f"   Total Nodes: {summary['total_nodes']}")
    print(f"   Total Edges: {summary['total_edges']}")
    print(f"   Density: {summary['density']:.3f}")
    
    print(f"\n📦 Node Types:")
    for node_type, count in summary['node_types'].items():
        print(f"   {node_type}: {count}")
    
    print(f"\n🔗 Relationships:")
    for rel_type, count in summary['relation_types'].items():
        print(f"   {rel_type}: {count}")
    
    print(f"\n🌟 Most Connected Nodes:")
    for node, degree in summary['central_nodes'][:5]:
        print(f"   {node[:50]}: {degree} connections")
    
    print("\n✅ FIXED: Graph now shows research entities, not config parameters!")