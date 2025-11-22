"""Full HGCN Pipeline with Citation Extraction"""
import time, sys, subprocess

def run_step(name, script):
    print(f"\n{'='*80}\n{name}\n{'='*80}")
    start = time.time()
    subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    print(f"\n✓ {name} complete in {int(elapsed//60)}m {int(elapsed%60)}s")
    return elapsed

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FULL HGCN PIPELINE - CITATION EXTRACTION + TRAINING")
    print("="*80)
    
    start = time.time()
    
    t1 = run_step("1. Citation Extraction (LLM)", "extract_citations_from_text.py")
    t2 = run_step("2. HGCN Training", "train_hyperbolic.py")
    t3 = run_step("3. 3D Visualization", "visualize_poincare.py")
    
    total = time.time() - start
    print(f"\n{'='*80}")
    print(f"COMPLETE! Total: {int(total//60)}m {int(total%60)}s")
    print(f"  Extraction: {int(t1)}s | Training: {int(t2)}s | Viz: {int(t3)}s")
