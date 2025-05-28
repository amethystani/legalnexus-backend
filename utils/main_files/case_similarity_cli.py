#!/usr/bin/env python3
# case_similarity_cli.py - A command line tool for finding similar legal cases
# Usage: python case_similarity_cli.py "your legal query here"
# For help: python case_similarity_cli.py --help
import os
import json
import sys
import glob
import argparse
import pickle
import re
import difflib
import time
from dotenv import load_dotenv
# Replace OpenAI imports with Google imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# Constants
EMBEDDINGS_CACHE_FILE = "case_embeddings_gemini.pkl"
MAX_RESULTS = 5
EMBEDDING_CHUNK_SIZE = 5  # Process embeddings in smaller chunks to avoid rate limits
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds
# Set Google Gemini API key
GOOGLE_API_KEY = "AIzaSyA0dLTfkzxcZYP6KidlFClAyMLl6mea1y8"

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

def compute_text_similarity(query_text, document_text):
    """Compute simple text similarity when embeddings are not available"""
    # Convert to lowercase for better matching
    query_lower = query_text.lower()
    doc_lower = document_text.lower()
    
    # Extract key words from query (remove common stop words)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'has', 'have', 'had', 'been', 'of', 'for', 'by', 'with', 'to', 'in', 
                 'on', 'at', 'from', 'as', 'it', 'its'}
    
    # Extract words from query and document
    query_words = set([word.strip('.,;:()[]{}""\'') for word in query_lower.split() 
                     if word.strip('.,;:()[]{}""\'') and word.strip('.,;:()[]{}""\'') not in stop_words])
    
    # Count matches
    match_count = sum(1 for word in query_words if word in doc_lower)
    
    # Calculate similarity score
    # Base score from word matching
    base_score = match_count / max(len(query_words), 1)
    
    # Add bonus from sequence matching using difflib
    similarity_bonus = difflib.SequenceMatcher(None, query_lower, doc_lower[:min(len(doc_lower), 1000)]).ratio() * 0.2
    
    # Final score combines direct word matches and sequence similarity
    return min(base_score + similarity_bonus, 1.0)  # Cap at 1.0

def load_all_legal_data(data_path="data"):
    """Load all the legal data from both test cases and scraped data"""
    print(f"Loading all legal data from: {data_path}")
    
    # Adjust the path if running from Backend directory
    if os.path.basename(os.getcwd()) == "Backend":
        data_path = os.path.join("..", data_path)
        
    all_docs = []
    json_files = glob.glob(os.path.join(data_path, "**/*.json"), recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract the relevant fields from the JSON structure
                content = data.get('content', '')
                if not content:
                    continue
                    
                # Create metadata from the available fields
                metadata = {
                    'source': data.get('source', os.path.basename(json_file)),
                    'title': data.get('title', os.path.basename(json_file)),
                    'court': data.get('court', 'Unknown Court'),
                    'judgment_date': data.get('judgment_date', 'Unknown Date'),
                    'id': data.get('id', f"doc_{len(all_docs)}"),
                }
                
                # Add judges if available
                if 'entities' in data and 'judges' in data['entities']:
                    metadata['judges'] = data['entities']['judges']
                elif 'metadata' in data and 'judges' in data['metadata']:
                    metadata['judges'] = data['metadata']['judges']
                
                # Add entities if available
                if 'entities' in data:
                    for entity_type, entities in data['entities'].items():
                        if entities:
                            metadata[entity_type] = entities
                
                # Create a Document object for this legal case
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)
                print(f"Loaded: {metadata.get('title', 'Unnamed case')}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Successfully loaded {len(all_docs)} legal documents")
    return all_docs

def compute_embeddings_with_gemini(docs, cache_file=EMBEDDINGS_CACHE_FILE, force_recompute=False):
    """Compute embeddings for a list of documents using Gemini with retries and caching"""
    # Check if we have cached embeddings
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading cached embeddings from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Verify the cache matches our current documents
            if len(cached_data['docs']) == len(docs) and all(
                cached_doc.page_content == doc.page_content 
                for cached_doc, doc in zip(cached_data['docs'], docs)
            ):
                print("Using cached embeddings")
                return cached_data['embeddings'], cached_data['docs']
            else:
                print("Cache mismatch - recomputing embeddings")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    print("\nComputing embeddings for documents with Gemini...")
    
    # Initialize Google Gemini embeddings
    try:
        # Create the embedding model with the Google API key
        embeddings_model = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/embedding-001",
            task_type="retrieval_document",
            title="Legal case document"
        )
        
        # Compute embeddings for each document in chunks to avoid rate limits
        doc_embeddings = []
        for i in range(0, len(docs), EMBEDDING_CHUNK_SIZE):
            chunk_end = min(i + EMBEDDING_CHUNK_SIZE, len(docs))
            chunk_docs = docs[i:chunk_end]
            print(f"Computing embeddings for documents {i+1}-{chunk_end} of {len(docs)}")
            
            # Process each document with retries
            for j, doc in enumerate(chunk_docs, i):
                retry_count = 0
                while retry_count < MAX_RETRY_ATTEMPTS:
                    try:
                        # Use Gemini to embed the document
                        embed = embeddings_model.embed_query(doc.page_content[:8000])  # Ensure text is within limits
                        doc_embeddings.append(embed)
                        print(f"  - Embedded document: {doc.metadata.get('title', f'Doc {j}')}")
                        break  # Success, exit retry loop
                    except Exception as e:
                        retry_count += 1
                        if retry_count < MAX_RETRY_ATTEMPTS:
                            print(f"Error embedding document {j} (attempt {retry_count}): {e}")
                            print(f"Retrying in {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"Failed to embed document {j} after {MAX_RETRY_ATTEMPTS} attempts: {e}")
                            # Add a placeholder embedding to maintain alignment
                            doc_embeddings.append([0] * 768)  # Gemini embedding dimension
            
            # Pause between chunks to avoid rate limits
            if i + EMBEDDING_CHUNK_SIZE < len(docs):
                print(f"Pausing for {RETRY_DELAY} seconds to avoid rate limits...")
                time.sleep(RETRY_DELAY)
        
        # Cache the embeddings
        print(f"Saving embeddings to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': doc_embeddings,
                'docs': docs
            }, f)
        
        return doc_embeddings, docs
    except Exception as e:
        print(f"Error initializing Gemini embeddings: {e}")
        print("Could not compute embeddings.")
        return None, docs

def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def find_similar_cases_with_embeddings(query_text, docs, doc_embeddings, embeddings_model, n=MAX_RESULTS):
    """Find cases similar to the query text using embeddings"""
    print(f"\nFinding cases similar to: {query_text[:100]}...")
    print("Using Gemini embeddings for precise similarity matching...")
    
    # Compute embedding for the query with retries
    query_embedding = None
    retry_count = 0
    while retry_count < MAX_RETRY_ATTEMPTS:
        try:
            query_embedding = embeddings_model.embed_query(query_text[:8000])  # Ensure text is within limits
            break
        except Exception as e:
            retry_count += 1
            if retry_count < MAX_RETRY_ATTEMPTS:
                print(f"Error embedding query (attempt {retry_count}): {e}")
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to embed query after {MAX_RETRY_ATTEMPTS} attempts: {e}")
                return None
    
    # Compute similarities
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        try:
            # Skip documents with placeholder embeddings
            if all(x == 0 for x in doc_embedding):
                continue
                
            similarity = compute_cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        except Exception as e:
            print(f"Error computing similarity for doc {i}: {e}")
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n similar documents
    results = []
    for i, similarity in similarities[:n]:
        doc = docs[i]
        results.append((doc, similarity))
    
    return results

def find_similar_cases_with_text(query_text, docs, n=MAX_RESULTS):
    """Find cases similar to the query text using text-based similarity"""
    print(f"\nFinding cases similar to: {query_text[:100]}...")
    print("Using text-based similarity search...")
    
    similarities = []
    for i, doc in enumerate(docs):
        try:
            similarity = compute_text_similarity(query_text, doc.page_content)
            similarities.append((i, similarity))
        except Exception as e:
            print(f"Error computing text similarity for doc {i}: {e}")
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n similar documents
    results = []
    for i, similarity in similarities[:n]:
        doc = docs[i]
        results.append((doc, similarity))
    
    return results

def print_results(results, verbose=False):
    """Print search results in a user-friendly format"""
    if not results:
        print("\nNo matching cases found.")
        return
        
    print("\n" + "=" * 80)
    print(f"Found {len(results)} similar cases:")
    print("=" * 80)
    
    for i, (doc, similarity) in enumerate(results):
        title = doc.metadata.get('title', 'Untitled Case')
        court = doc.metadata.get('court', 'Unknown Court')
        date = doc.metadata.get('judgment_date', 'Unknown Date')
        
        print(f"\n{i+1}. {title}")
        print(f"   Court: {court}")
        print(f"   Date: {date}")
        print(f"   Similarity: {similarity:.4f}")
        
        if verbose:
            # Print a more substantial excerpt in verbose mode
            excerpt_length = 500 if verbose else 200
            excerpt = doc.page_content[:excerpt_length] + "..." if len(doc.page_content) > excerpt_length else doc.page_content
            print(f"\n   Excerpt:\n   {excerpt}\n")
            
            # Print additional metadata if available
            if 'judges' in doc.metadata:
                judges = doc.metadata['judges']
                if isinstance(judges, list):
                    judges = ", ".join(judges)
                print(f"   Judges: {judges}")
            
            if 'statutes' in doc.metadata and doc.metadata['statutes']:
                statutes = doc.metadata['statutes']
                if isinstance(statutes, list):
                    statutes = ", ".join(statutes)
                print(f"   Statutes: {statutes}")
        
        print("-" * 80)

def filter_cases_by_criteria(docs, court=None, date_from=None, date_to=None):
    """Filter cases by court and date criteria"""
    filtered_docs = []
    
    for doc in docs:
        # Apply court filter if specified
        if court and court.lower() not in doc.metadata.get('court', '').lower():
            continue
            
        # Filter logic for dates would go here (requires date parsing)
        # This is a placeholder for future implementation
        
        filtered_docs.append(doc)
    
    return filtered_docs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Legal Case Similarity Search Tool')
    parser.add_argument('query', nargs='?', default=None, help='Search query text')
    parser.add_argument('--recompute', action='store_true', help='Force recomputation of embeddings')
    parser.add_argument('--court', type=str, help='Filter by court (e.g., "Supreme Court")')
    parser.add_argument('--results', type=int, default=MAX_RESULTS, help=f'Number of results to return (default: {MAX_RESULTS})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed case information')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--text-only', '-t', action='store_true', help='Use only text-based matching (no embeddings)')
    parser.add_argument('--force-embeddings', '-f', action='store_true', help='Force use of embeddings, fail if not available')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Legal Case Similarity Search (Using Gemini AI)")
    print("=" * 80)
    
    # Load all legal documents
    docs = load_all_legal_data()
    
    if not docs:
        print("No legal documents found. Exiting.")
        sys.exit(1)
    
    # Apply any filters before computing embeddings
    if args.court:
        print(f"Filtering by court: {args.court}")
        docs = filter_cases_by_criteria(docs, court=args.court)
        if not docs:
            print(f"No cases found matching court: {args.court}")
            sys.exit(1)
        print(f"Found {len(docs)} cases matching criteria")
    
    # Initialize Gemini embeddings if not in text-only mode
    embeddings_model = None
    doc_embeddings = None
    use_embeddings = not args.text_only
    
    if use_embeddings:
        try:
            # Initialize Gemini embedding model
            embeddings_model = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY,
                model="models/embedding-001"
            )
            print("Gemini embeddings initialized")
            
            # Compute or load embeddings
            doc_embeddings, docs = compute_embeddings_with_gemini(docs, force_recompute=args.recompute)
            if not doc_embeddings and args.force_embeddings:
                print("Failed to compute embeddings and --force-embeddings was specified. Exiting.")
                sys.exit(1)
        except Exception as e:
            print(f"Error initializing Gemini embeddings: {e}")
            if args.force_embeddings:
                print("Failed to initialize embeddings and --force-embeddings was specified. Exiting.")
                sys.exit(1)
            print("Falling back to text-based similarity.")
            use_embeddings = False
    else:
        print("Using text-based similarity (embeddings disabled)")
    
    # Interactive mode
    if args.interactive:
        while True:
            query = input("\nEnter your legal query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            # Search based on available methods
            results = None
            if use_embeddings and doc_embeddings and embeddings_model:
                results = find_similar_cases_with_embeddings(query, docs, doc_embeddings, embeddings_model, n=args.results)
                
            # Fall back to text search if embeddings failed or were disabled
            if not results:
                results = find_similar_cases_with_text(query, docs, n=args.results)
                
            print_results(results, verbose=args.verbose)
    
    # Direct query mode
    elif args.query:
        # Search based on available methods
        results = None
        if use_embeddings and doc_embeddings and embeddings_model:
            results = find_similar_cases_with_embeddings(args.query, docs, doc_embeddings, embeddings_model, n=args.results)
            
        # Fall back to text search if embeddings failed or were disabled
        if not results:
            results = find_similar_cases_with_text(args.query, docs, n=args.results)
            
        print_results(results, verbose=args.verbose)
    
    # No query provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python case_similarity_cli.py 'Section 65B electronic evidence whatsapp'")
        print("  python case_similarity_cli.py 'pension rights government employees' --court 'Supreme Court' --verbose")
        print("  python case_similarity_cli.py --interactive")
        print("  python case_similarity_cli.py --force-embeddings 'electronic evidence admissibility'")

if __name__ == "__main__":
    main() 