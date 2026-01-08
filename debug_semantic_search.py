#!/usr/bin/env python3
"""
Debug script for semantic search issues
Run this to see what's happening with your PDF
"""

import PyPDF2
from semantic_search import build_semantic_index, search_semantic
from pdf_utils import extract_text_from_pdf  # ✅ NEW IMPORT

# Test with your resume
pdf_path = "Sagar Prajapati Resume Final Nov .pdf (2).pdf"

print("=" * 60)
print("🔍 SEMANTIC SEARCH DEBUG")
print("=" * 60)

# Step 1: Extract PDF text
print("\n1️⃣ Extracting PDF text...")
try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    print(f"✅ Extracted {len(text)} characters")
    print(f"📄 First 200 chars: {text[:200]}...")
    
except Exception as e:
    print(f"❌ Error reading PDF: {e}")
    exit(1)

# Step 2: Build semantic index
print("\n2️⃣ Building semantic index...")
try:
    vectordb = build_semantic_index(text, chunk_size=500, chunk_overlap=100)
    print("✅ Index built successfully")
    
except Exception as e:
    print(f"❌ Error building index: {e}")
    exit(1)

# Step 3: Test search
print("\n3️⃣ Testing semantic search...")
test_queries = [
    "DSA problems solved",
    "GeeksforGeeks",
    "work experience",
    "Python programming",
    "540 problems"
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    try:
        results = search_semantic(vectordb, query, k=5)
        
        if not results:
            print("   ⚠️ No results returned (empty list)")
        else:
            print(f"   ✅ Found {len(results)} results:")
            for i, (text_chunk, distance) in enumerate(results[:3], 1):
                print(f"\n   Result {i} (distance: {distance:.4f}):")
                print(f"   {text_chunk[:150]}...")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("🏁 DEBUG COMPLETE")
print("=" * 60)