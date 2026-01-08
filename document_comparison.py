# document_comparison.py - Deep Insights Document Comparison

import requests
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
import re

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_class = HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings_class = SentenceTransformerEmbeddings

from sklearn.metrics.pairwise import cosine_similarity


class DocumentComparison:
    """Advanced document comparison with deep insights"""
    
    def __init__(self, model="llama3"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.documents = {}
        
        try:
            self.embeddings_model = embeddings_class(model_name="all-MiniLM-L6-v2")
        except:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            self.embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Comprehensive technology taxonomy
        self.tech_categories = {
            'AI/ML': ['tensorflow', 'pytorch', 'keras', 'scikit', 'sklearn', 'xgboost', 
                     'machine learning', 'deep learning', 'neural network', 'nlp', 
                     'computer vision', 'opencv', 'cnn', 'rnn', 'lstm', 'transformer',
                     'bert', 'gpt', 'llm', 'generative', 'reinforcement'],
            
            'Web Development': ['react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt',
                               'html', 'css', 'javascript', 'typescript', 'jquery',
                               'bootstrap', 'tailwind', 'sass', 'webpack', 'babel'],
            
            'Backend': ['node.js', 'express', 'django', 'flask', 'fastapi', 'spring',
                       'spring boot', 'laravel', 'ruby on rails', 'asp.net', 'graphql',
                       'rest', 'api', 'microservices', 'serverless'],
            
            'Cloud & DevOps': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                              'jenkins', 'gitlab', 'github actions', 'terraform', 'ansible',
                              'ci/cd', 'ec2', 's3', 'lambda', 'cloudformation'],
            
            'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                         'dynamodb', 'cassandra', 'oracle', 'sqlite', 'firebase'],
            
            'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'c++',
                                     'c', 'go', 'rust', 'kotlin', 'swift', 'php', 'ruby',
                                     'scala', 'r', 'matlab'],
            
            'Data Science': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
                           'jupyter', 'data analysis', 'statistics', 'visualization',
                           'tableau', 'power bi', 'excel'],
            
            'Mobile Development': ['android', 'ios', 'react native', 'flutter', 'swift',
                                  'kotlin', 'xamarin', 'ionic'],
            
            'Tools & Others': ['git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
                             'agile', 'scrum', 'linux', 'unix', 'bash', 'powershell']
        }
    
    def add_document(self, name: str, text: str):
        """Add document to comparison pool"""
        self.documents[name] = text
        print(f"✅ Added: {name} ({len(text)} chars)")
    
    def clear_documents(self):
        """Clear all documents"""
        self.documents = {}
    
    def _check_ollama(self) -> bool:
        """Check Ollama availability"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                has_llama = any('llama3' in m.get('name', '') for m in models)
                if has_llama:
                    print("✅ Ollama with llama3 available")
                    return True
            return False
        except:
            return False
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity"""
        try:
            emb1 = self.embeddings_model.embed_query(text1[:8000])
            emb2 = self.embeddings_model.embed_query(text2[:8000])
            return float(cosine_similarity([emb1], [emb2])[0][0])
        except:
            return 0.0
    
    def _categorize_technologies(self, text: str) -> Dict[str, List[str]]:
        """Categorize technologies found in text"""
        text_lower = text.lower()
        found_tech = {}
        
        for category, techs in self.tech_categories.items():
            found = [tech for tech in techs if tech in text_lower]
            if found:
                found_tech[category] = list(set(found))
        
        return found_tech
    
    def _extract_experience_level(self, text: str) -> str:
        """Determine experience level from text"""
        text_lower = text.lower()
        
        # Look for explicit years of experience
        years_match = re.search(r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', text_lower)
        if years_match:
            years = int(years_match.group(1))
            if years >= 5:
                return f"Senior ({years}+ years)"
            elif years >= 2:
                return f"Mid-level ({years} years)"
            else:
                return f"Junior ({years} year{'s' if years > 1 else ''})"
        
        # Infer from job titles
        if 'senior' in text_lower or 'lead' in text_lower or 'principal' in text_lower:
            return "Senior (3+ years)"
        elif 'junior' in text_lower or 'intern' in text_lower or 'trainee' in text_lower:
            return "Entry-level/Junior"
        else:
            return "Mid-level (estimated)"
    
    def _extract_projects(self, text: str) -> List[str]:
        """Extract project mentions"""
        # Look for project sections
        projects = []
        lines = text.split('\n')
        
        in_project_section = False
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect project section
            if 'project' in line_lower and len(line_lower) < 30:
                in_project_section = True
                continue
            
            # Detect end of project section
            if in_project_section and any(keyword in line_lower for keyword in 
                                         ['education', 'experience', 'skill', 'achievement']):
                in_project_section = False
            
            # Collect project lines
            if in_project_section and line.strip() and len(line.strip()) > 10:
                # Remove bullet points and clean
                clean_line = re.sub(r'^[\-\*•]\s*', '', line.strip())
                if clean_line:
                    projects.append(clean_line[:100])  # Limit length
        
        return projects[:5]  # Return top 5
    
    def _analyze_strengths(self, text: str, tech_categories: Dict) -> List[str]:
        """Identify key strengths based on text analysis"""
        strengths = []
        
        # Analyze technology depth
        for category, techs in tech_categories.items():
            if len(techs) >= 3:
                strengths.append(f"Strong {category} expertise ({len(techs)} technologies)")
        
        # Look for achievement indicators
        text_lower = text.lower()
        achievement_keywords = {
            'competition': 'Competitive programming experience',
            'published': 'Research publications',
            'award': 'Award winner',
            'certified': 'Professional certifications',
            'led team': 'Team leadership experience',
            'managed': 'Project management skills'
        }
        
        for keyword, strength in achievement_keywords.items():
            if keyword in text_lower:
                strengths.append(strength)
        
        return strengths
    
    def compare_documents(self, doc1_name: str, doc2_name: str) -> Dict:
        """Perform deep comparison with insights"""
        
        if doc1_name not in self.documents or doc2_name not in self.documents:
            raise ValueError("Documents not found")
        
        text1 = self.documents[doc1_name]
        text2 = self.documents[doc2_name]
        
        print(f"\n📊 Deep Analysis: {doc1_name} vs {doc2_name}")
        
        # Calculate similarity
        similarity = self.get_semantic_similarity(text1, text2)
        print(f"   Similarity: {similarity:.2%}")
        
        # Try Ollama first, fall back to intelligent analysis
        if self._check_ollama():
            try:
                return self._llm_comparison(doc1_name, text1, doc2_name, text2, similarity)
            except:
                print("   Ollama failed, using deep analysis")
        
        return self._deep_analysis(doc1_name, text1, doc2_name, text2, similarity)
    
    def _deep_analysis(self, name1: str, text1: str, name2: str, text2: str, 
                       similarity: float) -> Dict:
        """Generate deep, insightful comparison"""
        
        # Categorize technologies
        tech1 = self._categorize_technologies(text1)
        tech2 = self._categorize_technologies(text2)
        
        # Experience levels
        exp1 = self._extract_experience_level(text1)
        exp2 = self._extract_experience_level(text2)
        
        # Projects
        proj1 = self._extract_projects(text1)
        proj2 = self._extract_projects(text2)
        
        # Strengths
        strengths1 = self._analyze_strengths(text1, tech1)
        strengths2 = self._analyze_strengths(text2, tech2)
        
        # Find common and unique technology categories
        common_cats = set(tech1.keys()) & set(tech2.keys())
        unique_cats1 = set(tech1.keys()) - set(tech2.keys())
        unique_cats2 = set(tech2.keys()) - set(tech1.keys())
        
        # Generate insights
        
        # SUMMARY
        profile_type1 = self._infer_profile_type(tech1, text1)
        profile_type2 = self._infer_profile_type(tech2, text2)
        
        summary = f"""**🎯 Executive Summary**

**Candidate 1** ({name1}) - {profile_type1}
- Experience Level: {exp1}
- Technology Stack: {len(tech1)} categories, {sum(len(t) for t in tech1.values())} technologies
- Specialization: {', '.join(list(tech1.keys())[:3]) if tech1 else 'Generalist'}

**Candidate 2** ({name2}) - {profile_type2}
- Experience Level: {exp2}
- Technology Stack: {len(tech2)} categories, {sum(len(t) for t in tech2.values())} technologies
- Specialization: {', '.join(list(tech2.keys())[:3]) if tech2 else 'Generalist'}

**Match Analysis:** {similarity:.1%} similarity
{self._get_similarity_interpretation(similarity)}

**Quick Verdict:** {'Both candidates have comparable skill sets but different focuses.' if similarity > 0.5 else 'Candidates have distinctly different skill sets and specializations.'}"""

        # SIMILARITIES
        similarities_parts = []
        
        if common_cats:
            similarities_parts.append(f"**🔗 Shared Technical Domains ({len(common_cats)} areas):**")
            for cat in sorted(common_cats):
                common_tech = set(tech1[cat]) & set(tech2[cat])
                if common_tech:
                    similarities_parts.append(f"  • **{cat}:** {', '.join(sorted(common_tech))}")
                else:
                    similarities_parts.append(f"  • **{cat}:** Both have exposure but different tools")
        
        # Check for similar experience trajectory
        if exp1 == exp2:
            similarities_parts.append(f"\n**📊 Experience Level:** Both candidates at {exp1} level")
        
        # Similar project counts
        if abs(len(proj1) - len(proj2)) <= 1:
            similarities_parts.append(f"\n**🛠️ Project Experience:** Both have {len(proj1)}-{len(proj2)} documented projects")
        
        similarities = "\n".join(similarities_parts) if similarities_parts else "Limited technical overlap detected."
        
        # DIFFERENCES
        differences_parts = []
        
        differences_parts.append(f"**📋 {name1} - Unique Strengths:**")
        
        if unique_cats1:
            for cat in sorted(unique_cats1):
                differences_parts.append(f"  • **{cat}:** {', '.join(tech1[cat][:5])}")
        
        if strengths1:
            differences_parts.append(f"  • **Highlights:** {' | '.join(strengths1[:3])}")
        
        if proj1:
            differences_parts.append(f"  • **Key Projects:** {len(proj1)} documented ({proj1[0][:60]}...)")
        
        differences_parts.append(f"\n**📋 {name2} - Unique Strengths:**")
        
        if unique_cats2:
            for cat in sorted(unique_cats2):
                differences_parts.append(f"  • **{cat}:** {', '.join(tech2[cat][:5])}")
        
        if strengths2:
            differences_parts.append(f"  • **Highlights:** {' | '.join(strengths2[:3])}")
        
        if proj2:
            differences_parts.append(f"  • **Key Projects:** {len(proj2)} documented ({proj2[0][:60]}...)")
        
        # Experience gap
        if exp1 != exp2:
            differences_parts.append(f"\n**⚖️ Experience Gap:** {name1} ({exp1}) vs {name2} ({exp2})")
        
        differences = "\n".join(differences_parts)
        
        # RECOMMENDATIONS
        recommendations_parts = []
        
        recommendations_parts.append("**🎯 Hiring Recommendations:**\n")
        
        # Specific role recommendations
        if 'AI/ML' in tech1 and 'AI/ML' not in tech2:
            recommendations_parts.append(f"✅ **Choose {name1} for:** AI/ML roles, data science, computer vision, NLP projects")
        elif 'AI/ML' in tech2 and 'AI/ML' not in tech1:
            recommendations_parts.append(f"✅ **Choose {name2} for:** AI/ML roles, data science, computer vision, NLP projects")
        
        if 'Web Development' in tech1 and 'Web Development' not in tech2:
            recommendations_parts.append(f"✅ **Choose {name1} for:** Frontend/fullstack roles, web application development")
        elif 'Web Development' in tech2 and 'Web Development' not in tech1:
            recommendations_parts.append(f"✅ **Choose {name2} for:** Frontend/fullstack roles, web application development")
        
        if 'Cloud & DevOps' in tech1 and 'Cloud & DevOps' not in tech2:
            recommendations_parts.append(f"✅ **Choose {name1} for:** DevOps roles, cloud infrastructure, deployment automation")
        elif 'Cloud & DevOps' in tech2 and 'Cloud & DevOps' not in tech1:
            recommendations_parts.append(f"✅ **Choose {name2} for:** DevOps roles, cloud infrastructure, deployment automation")
        
        # Overall recommendation
        if similarity > 0.6:
            recommendations_parts.append(f"\n**⚖️ Overall:** Both candidates are closely matched. Decision should be based on:")
            recommendations_parts.append("  • Cultural fit and communication skills")
            recommendations_parts.append("  • Specific project requirements")
            recommendations_parts.append("  • Team composition needs")
        else:
            recommendations_parts.append(f"\n**⚖️ Overall:** Clear differentiation between candidates.")
            recommendations_parts.append(f"  • {name1}: Better for {profile_type1} roles")
            recommendations_parts.append(f"  • {name2}: Better for {profile_type2} roles")
        
        recommendations_parts.append("\n**💡 Next Steps:** Conduct technical interviews focusing on hands-on problem solving in their respective domains.")
        
        recommendations = "\n".join(recommendations_parts)
        
        return {
            'summary': summary,
            'similarities': similarities,
            'differences': differences,
            'recommendations': recommendations,
            'similarity_score': similarity
        }
    
    def _infer_profile_type(self, tech_cats: Dict, text: str) -> str:
        """Infer the profile type from technologies"""
        if 'AI/ML' in tech_cats:
            return "AI/ML Engineer"
        elif 'Web Development' in tech_cats and 'Backend' in tech_cats:
            return "Full-Stack Developer"
        elif 'Web Development' in tech_cats:
            return "Frontend Developer"
        elif 'Backend' in tech_cats:
            return "Backend Developer"
        elif 'Cloud & DevOps' in tech_cats:
            return "DevOps Engineer"
        elif 'Mobile Development' in tech_cats:
            return "Mobile Developer"
        elif 'Data Science' in tech_cats:
            return "Data Scientist"
        else:
            return "Software Engineer"
    
    def _get_similarity_interpretation(self, similarity: float) -> str:
        """Get human-readable similarity interpretation"""
        if similarity > 0.75:
            return "🟢 **Very High Similarity** - Candidates are nearly interchangeable for most roles"
        elif similarity > 0.60:
            return "🟡 **High Similarity** - Substantial overlap with some specialization differences"
        elif similarity > 0.45:
            return "🟠 **Moderate Similarity** - Some common ground but distinct specializations"
        elif similarity > 0.30:
            return "🔴 **Low Similarity** - Different skill sets, suited for different roles"
        else:
            return "⚫ **Very Low Similarity** - Completely different profiles and expertise"
    
    def _llm_comparison(self, name1: str, text1: str, name2: str, text2: str,
                       similarity: float) -> Dict:
        """LLM-based comparison (fallback to deep analysis on error)"""
        try:
            prompt = f"""Compare these resumes professionally. Be specific and insightful.

RESUME 1: {name1}
{text1[:2500]}

RESUME 2: {name2}
{text2[:2500]}

Provide:

1. SUMMARY: What each candidate excels at, experience level, main specialization (3-4 sentences)

2. SIMILARITIES: Specific shared skills, technologies, qualifications (4-5 bullet points)

3. DIFFERENCES: Unique strengths of each candidate with examples (4-5 bullet points per candidate)

4. RECOMMENDATIONS: Which candidate for which roles, with reasoning (3-4 sentences)

Be specific with technologies, projects, and qualifications mentioned."""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 800, "num_ctx": 4096}
            }
            
            print("   Waiting for AI analysis...")
            response = requests.post(self.ollama_url, json=payload, timeout=90)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("response", "").strip()
                
                if len(analysis) > 50:
                    sections = self._parse_response(analysis)
                    sections['similarity_score'] = similarity
                    print("   ✅ AI analysis complete")
                    return sections
            
            raise Exception("Invalid response")
            
        except Exception as e:
            print(f"   Using deep analysis (Ollama: {e})")
            return self._deep_analysis(name1, text1, name2, text2, similarity)
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response"""
        sections = {'summary': '', 'similarities': '', 'differences': '', 'recommendations': ''}
        
        lines = response.split('\n')
        current = None
        
        for line in lines:
            lower = line.lower().strip()
            
            if 'summary' in lower and (':' in line or lower.startswith('1')):
                current = 'summary'
                continue
            elif 'similar' in lower and (':' in line or lower.startswith('2')):
                current = 'similarities'
                continue
            elif 'differ' in lower and (':' in line or lower.startswith('3')):
                current = 'differences'
                continue
            elif 'recommend' in lower and (':' in line or lower.startswith('4')):
                current = 'recommendations'
                continue
            
            if current and line.strip():
                sections[current] += line + '\n'
        
        for key in sections:
            sections[key] = sections[key].strip() or f"Details for {key} not available"
        
        return sections


# Test
if __name__ == "__main__":
    print("🧪 Testing Deep Insights Comparison\n")
    
    comp = DocumentComparison()
    
    resume1 = """
    John Smith - AI/ML Engineer
    
    Experience: 3 years in machine learning and computer vision
    
    Skills: Python, TensorFlow, PyTorch, OpenCV, Scikit-learn, CNN, RNN, 
    Deep Learning, Computer Vision, NLP, AWS, Docker, Git
    
    Projects:
    - Built real-time object detection system using YOLO
    - Developed sentiment analysis model with 95% accuracy
    - Created image classification pipeline for medical diagnosis
    
    Education: MS in AI, Stanford University, GPA 3.9
    
    Achievements: Published 2 papers at CVPR, Winner of Kaggle competition
    """
    
    resume2 = """
    Jane Doe - Full Stack Developer
    
    Experience: 2.5 years in web development
    
    Skills: JavaScript, TypeScript, React, Node.js, Express, MongoDB, 
    PostgreSQL, HTML, CSS, REST API, Docker, Git, AWS
    
    Projects:
    - Developed e-commerce platform handling 10K users
    - Built real-time chat application with WebSockets
    - Created admin dashboard with data visualization
    
    Education: BS in Computer Science, MIT, GPA 3.8
    
    Achievements: Led team of 3 developers, Certified AWS Developer
    """
    
    comp.add_document("AI_Engineer.pdf", resume1)
    comp.add_document("FullStack_Dev.pdf", resume2)
    
    result = comp.compare_documents("AI_Engineer.pdf", "FullStack_Dev.pdf")
    
    print("\n" + "="*70)
    print(f"\n🎯 Similarity: {result['similarity_score']:.2%}\n")
    print("="*70)
    print("\n" + result['summary'])
    print("\n" + "="*70)
    print("\n🤝 SIMILARITIES:\n" + result['similarities'])
    print("\n" + "="*70)
    print("\n⚡ DIFFERENCES:\n" + result['differences'])
    print("\n" + "="*70)
    print("\n💡 RECOMMENDATIONS:\n" + result['recommendations'])