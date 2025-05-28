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
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# Constants
EMBEDDINGS_CACHE_FILE = "case_embeddings.pkl"
MAX_RESULTS = 5
EMBEDDING_CHUNK_SIZE = 10  # Process embeddings in chunks to avoid rate limits

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

def compute_embeddings(docs, cache_file=EMBEDDINGS_CACHE_FILE, force_recompute=False):
    """Compute embeddings for a list of documents, with caching support"""
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
    
    print("\nComputing embeddings for documents...")
    
    load_dotenv()  # Load environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    embeddings_model = OpenAIEmbeddings()
    
    # Compute embeddings for each document in chunks to avoid rate limits
    doc_embeddings = []
    for i in range(0, len(docs), EMBEDDING_CHUNK_SIZE):
        chunk_end = min(i + EMBEDDING_CHUNK_SIZE, len(docs))
        print(f"Computing embeddings for documents {i+1}-{chunk_end} of {len(docs)}")
        
        for j in range(i, chunk_end):
            try:
                embed = embeddings_model.embed_query(docs[j].page_content)
                doc_embeddings.append(embed)
                print(f"  - Embedded document: {docs[j].metadata.get('title', f'Doc {j}')}")
            except Exception as e:
                print(f"Error embedding document {j}: {e}")
                # Add a placeholder embedding to maintain alignment
                doc_embeddings.append([0] * 1536)  # Standard OpenAI embedding dimension
    
    # Cache the embeddings
    print(f"Saving embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'embeddings': doc_embeddings,
            'docs': docs
        }, f)
    
    return doc_embeddings, docs

def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def find_similar_cases(query_text, docs, doc_embeddings, embeddings_model, n=MAX_RESULTS):
    """Find cases similar to the query text using embeddings"""
    print(f"\nFinding cases similar to: {query_text[:100]}...")
    
    # Compute embedding for the query
    query_embedding = embeddings_model.embed_query(query_text)
    
    # Compute similarities
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        try:
            similarity = compute_cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        except Exception as e:
            print(f"Error computing similarity for doc {i}: {e}")
            similarities.append((i, 0.0))
    
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
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Legal Case Similarity Search")
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
    
    # Initialize OpenAI embeddings
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    embeddings_model = OpenAIEmbeddings()
    print("OpenAI embeddings initialized")
    
    # Compute or load embeddings for all documents
    doc_embeddings, docs = compute_embeddings(docs, force_recompute=args.recompute)
    
    # Interactive mode
    if args.interactive:
        while True:
            query = input("\nEnter your legal query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            results = find_similar_cases(query, docs, doc_embeddings, embeddings_model, n=args.results)
            print_results(results, verbose=args.verbose)
    
    # Direct query mode
    elif args.query:
        results = find_similar_cases(args.query, docs, doc_embeddings, embeddings_model, n=args.results)
        print_results(results, verbose=args.verbose)
    
    # No query provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python case_similarity.py 'Section 65B electronic evidence whatsapp'")
        print("  python case_similarity.py 'pension rights government employees' --court 'Supreme Court' --verbose")
        print("  python case_similarity.py --interactive")

if __name__ == "__main__":
    main() 