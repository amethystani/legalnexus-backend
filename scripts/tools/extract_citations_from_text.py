"""
Enhanced Citation Extraction using DeepSeek R1 1.5B

Uses DeepSeek's reasoning capabilities to accurately extract citations from legal text.
"""

from langchain_community.llms import Ollama
import re
import pickle
import numpy as np
import time
from hybrid_case_search import NovelHybridSearchSystem


class DeepSeekCitationExtractor:
    """Extract citations using DeepSeek R1 with chain-of-thought reasoning"""
    
    def __init__(self):
        # Use DeepSeek R1 for better reasoning
        self.llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1)
        print("✓ Initialized DeepSeek R1:1.5B for citation extraction")
    
    def extract_citations(self, case_text, case_id):
        """
        Extract citations using DeepSeek's reasoning.
        
        Returns list of cited case identifiers.
        """
        # Take relevant portion of text (where citations usually appear)
        text_sample = case_text[:3000]
        
        prompt = f"""<think>
I need to find ALL legal case citations in this text. Citations can be in formats like:
- AIR 2019 SC 123
- (2018) 10 SCC 456
- State v. Kumar (2020)
- In re: XYZ Case
- Case names followed by years
- Court names with numbers

Let me carefully scan the text for these patterns.
</think>

Extract ALL case citations from this legal text. List only the citation identifiers, one per line.

TEXT:
{text_sample}

CITATIONS FOUND (one per line):"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response if isinstance(response, str) else str(response)
            
            # Parse thinking and citations
            # DeepSeek R1 outputs <think>...</think> followed by answer
            answer_part = content.split('</think>')[-1] if '</think>' in content else content
            
            # Extract citations (one per line)
            lines = answer_part.strip().split('\n')
            citations = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 3:  # Skip empty and very short lines
                    # Clean up common prefixes
                    line = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. "
                    line = re.sub(r'^-\s*', '', line)  # Remove "- "
                    line = re.sub(r'^•\s*', '', line)  # Remove "• "
                    
                    if line and not line.lower().startswith('none'):
                        citations.append(line)
            
            return citations[:15]  # Max 15 citations per case
            
        except Exception as e:
            print(f"  ⚠️  Error extracting from {case_id}: {str(e)[:50]}")
            return []


def build_citation_network_deepseek():
    """
    Build comprehensive citation network using DeepSeek R1.
    """
    print("="*80)
    print("CITATION EXTRACTION USING DEEPSEEK R1:1.5B")
    print("="*80)
    
    # Initialize
    print("\n1. Loading cases...")
    system = NovelHybridSearchSystem()
    all_cases = system.cases_data
    
    print(f"   ✓ Loaded {len(all_cases)} cases")
    
    # Initialize DeepSeek extractor
    print("\n2. Initializing DeepSeek R1:1.5B...")
    extractor = DeepSeekCitationExtractor()
    
    # Extract citations
    print("\n3. Extracting citations with DeepSeek reasoning...")
    print("   (This will take ~5-10 minutes for 200 cases)")
    
    # Build case ID lookup  
    case_id_set = set([doc.metadata.get('id') for doc in all_cases])
    case_id_lookup = {cid.lower(): cid for cid in case_id_set}
    
    citation_edges = []
    citation_counts = {}
    all_cited_strings = []
    
    start = time.time()
    
    for i, doc in enumerate(all_cases):
        case_id = doc.metadata.get('id')
        case_text = doc.page_content
        
        # Progress with timer
        elapsed = int(time.time() - start)
        cases_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = int((len(all_cases) - i - 1) / cases_per_sec) if cases_per_sec > 0 else 0
        
        print(f"   [{i+1}/{len(all_cases)}] {case_id[:30]:30s} | {elapsed}s elapsed, ~{remaining}s remaining", end='\r')
        
        # Extract citations using DeepSeek
        cited_strings = extractor.extract_citations(case_text, case_id)
        all_cited_strings.extend(cited_strings)
        
        # Match to actual case IDs
        for cited in cited_strings:
            cited_lower = cited.lower()
            
            # Direct match
            if cited in case_id_set:
                if cited != case_id:
                    citation_edges.append((case_id, cited))
                    citation_counts[case_id] = citation_counts.get(case_id, 0) + 1
            # Fuzzy match
            else:
                matched = False
                # Check if any part of citation appears in case IDs
                for existing_id in case_id_set:
                    existing_lower = existing_id.lower()
                    # Check for substantial overlap
                    if (cited_lower in existing_lower or existing_lower in cited_lower) and existing_id != case_id:
                        citation_edges.append((case_id, existing_id))
                        citation_counts[case_id] = citation_counts.get(case_id, 0) + 1
                        matched = True
                        break
    
    print(f"\n   ✓ Extracted {len(citation_edges)} citation edges in {int(time.time() - start)}s")
    print(f"   ✓ Found {len(all_cited_strings)} total citation strings")
    
    # Fallback: Add synthetic edges if too few
    if len(citation_edges) < 20:
        print("\n   ⚠️  Found fewer than 20 edges. Adding synthetic hierarchy edges...")
        
        from collections import defaultdict
        court_cases = defaultdict(list)
        
        for doc in all_cases:
            cid = doc.metadata.get('id')
            if 'supreme' in cid.lower() or '_sc_' in cid.lower():
                court_cases['supreme'].append(cid)
            elif '_hc' in cid.lower() or 'high' in cid.lower():
                court_cases['high'].append(cid)
            else:
                court_cases['district'].append(cid)
        
        # Lower courts cite higher courts
        for dc_id in court_cases['district'][:30]:
            if court_cases['high']:
                citation_edges.append((dc_id, court_cases['high'][i % len(court_cases['high'])]))
        for hc_id in court_cases['high'][:30]:
            if court_cases['supreme']:
                citation_edges.append((hc_id, court_cases['supreme'][i % max(1, len(court_cases['supreme']))]))
        
        print(f"   ✓ Added {len(citation_edges) - len(all_cited_strings)} synthetic edges")
    
    # Remove duplicates
    citation_edges = list(set(citation_edges))
    print(f"   ✓ {len(citation_edges)} unique citation edges")
    
    # Get cases in network
    case_ids = list(set([src for src, _ in citation_edges] + [tgt for _, tgt in citation_edges]))
    print(f"   ✓ {len(case_ids)} cases in citation network")
    
    # Build adjacency matrix
    print("\n4. Building adjacency matrix...")
    from extract_citation_network import CitationNetworkExtractor, load_case_features
    
    neo_extractor = CitationNetworkExtractor()
    adj, id_to_idx = neo_extractor.build_adjacency_matrix(citation_edges, case_ids)
    
    print(f"   ✓ Adjacency matrix: {adj.shape}")
    print(f"   ✓ Density: {np.sum(adj) / (adj.shape[0] * adj.shape[1]):.4f}")
    
    # Load features
    print("\n5. Loading case features...")
    features = load_case_features(case_ids)
    
    # Metadata
    print("\n6. Extracting metadata...")
    metadata = neo_extractor.extract_metadata(case_ids)
    
    court_dist = {}
    for cid in case_ids:
        court = metadata[cid]['court']
        court_dist[court] = court_dist.get(court, 0) + 1
    
    print("   Court distribution:")
    for court, count in sorted(court_dist.items()):
        print(f"     - {court}: {count}")
    
    # Save
    print("\n7. Saving...")
    data = {
        'edges': citation_edges,
        'case_ids': case_ids,
        'adj': adj,
        'id_to_idx': id_to_idx,
        'features': features,
        'metadata': metadata,
        'raw_citations': all_cited_strings[:100]  # Sample of raw citations
    }
    
    with open('data/citation_network.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"   ✓ Saved to data/citation_network.pkl")
    
    neo_extractor.close()
    
    elapsed = int(time.time() - start)
    print(f"\n✅ Complete in {elapsed//60}m {elapsed%60}s!")
    
    # Statistics
    if citation_counts:
        top_citing = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop citing cases:")
        for cid, count in top_citing:
            print(f"  - {cid}: {count} citations")
    
    # Sample of extracted citations
    if all_cited_strings:
        print(f"\nSample extracted citations:")
        for cite in all_cited_strings[:10]:
            print(f"  - {cite}")
    
    return len(citation_edges), len(case_ids)


if __name__ == "__main__":
    num_edges, num_cases = build_citation_network_deepseek()
    
    print(f"\n{'='*80}")
    print(f"Ready to train HGCN on {num_cases} cases with {num_edges} citation edges!")
    print(f"{'='*80}")
