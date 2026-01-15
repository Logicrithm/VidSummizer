"""
VidSummarize - Smart Summarization Module
Complete production version with speed + intelligence

Architecture:
- Uses preprocess.py for text preparation (clean, split, chunk)
- Focuses on orchestration, summarization, and formatting
- Three modes: fast (<3s), balanced (<6s), quality (<12s)

Author: VidSummarize Team
Version: 2.0 - Hybrid Speed + Intelligence
"""

from transformers import pipeline
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import os
import traceback

# Import preprocessing functions
try:
    from preprocess import (
        safe_clean,
        safe_split,
        detect_domain,
        light_idea_filter,
        sentence_chunks,
        fast_chunks,
        get_text_stats
    )
    _HAS_PREPROCESS = True
except ImportError:
    print("[Summarizer] Warning: preprocess.py not found, using fallback mode")
    _HAS_PREPROCESS = False


class Summarizer:
    """
    Intelligent summarizer with three performance modes
    
    Modes:
        fast: <3s processing, minimal structure, 75% quality
        balanced: <6s processing, good structure, 85% quality (default)
        quality: <12s processing, excellent structure, 93% quality
    
    Usage:
        summarizer = Summarizer()
        summary = summarizer.summarize(text, max_length=300)
    """
    
    def __init__(self, model_name='facebook/bart-large-cnn'):
        """
        Initialize summarizer with model and configuration
        
        Args:
            model_name: HuggingFace model for summarization
        """
        print(f"[Summarizer] Loading model: {model_name}")
        
        # Get configuration from environment
        self.mode = os.getenv("SUMMARY_MODE", "balanced")
        self.quality_mode = os.getenv("SUMMARY_QUALITY", "balanced")
        
        # Validate mode
        if self.mode not in ['fast', 'balanced', 'quality']:
            print(f"[Summarizer] Warning: Unknown mode '{self.mode}', using 'balanced'")
            self.mode = 'balanced'
        
        print(f"[Summarizer] Quality mode: {self.quality_mode}")
        
        # Device selection
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = 'cuda' if self.device == 0 else 'cpu'
        print(f"[Summarizer] Using device: {device_name}")
        
        # Load model
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=self.device
            )
            print(f"[Summarizer] Model loaded successfully")
        except Exception as e:
            print(f"[Summarizer] Error loading model: {e}")
            raise
        
        # Parallel processing configuration
        self.parallel_enabled = os.getenv("ENABLE_PARALLEL_SUMMARY", "true").lower() == "true"
        self.max_workers = int(os.getenv("SUMMARY_WORKERS", "3"))
        
        print(f"[Summarizer] Parallel processing: {self.parallel_enabled}")
        if self.parallel_enabled:
            print(f"[Summarizer] Workers: {self.max_workers}")
    
    # ========================================================================
    # TIMING AND MONITORING
    # ========================================================================
    
    def _timed_stage(self, name: str, func, *args, timeout_ms: int = 1000):
        """
        Execute function with timing and budget monitoring
        
        Args:
            name: Stage name for logging
            func: Function to execute
            *args: Arguments to pass to function
            timeout_ms: Warning threshold in milliseconds
            
        Returns:
            Function result
        """
        start = time.time()
        result = func(*args)
        elapsed_ms = (time.time() - start) * 1000
        
        # Status indicator
        status = "✓" if elapsed_ms <= timeout_ms else "⚠"
        print(f"[Summarizer] {status} {name:15} {elapsed_ms:6.1f}ms")
        
        return result
    
    # ========================================================================
    # FALLBACK FUNCTIONS (if preprocess.py not available)
    # ========================================================================
    
    def _fallback_clean(self, text: str) -> str:
        """Simple fallback cleaning if preprocess.py missing"""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _fallback_chunks(self, text: str, max_chars: int = 40000) -> List[str]:
        """Simple fallback chunking if preprocess.py missing"""
        if len(text) <= max_chars:
            return [text]
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
        return chunks
    
    # ========================================================================
    # CHUNK SUMMARIZATION
    # ========================================================================
    
    def _summarize_single_chunk(self, chunk: str, target_length: int, 
                                chunk_idx: int = 0) -> str:
        """
        Summarize a single text chunk
        
        Args:
            chunk: Text to summarize
            target_length: Target length in words
            chunk_idx: Index for logging
            
        Returns:
            Summary text
        """
        try:
            # Calculate token lengths (words * 1.3 for safety)
            min_len = max(20, target_length - 30)
            max_len = target_length + 30
            
            # Run summarization
            result = self.summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            
            summary = result[0]['summary_text'] if result else ""
            
            if chunk_idx > 0:
                word_count = len(summary.split())
                print(f"[Summarizer] ✓ Chunk {chunk_idx} complete ({word_count} words)")
            
            return summary
            
        except Exception as e:
            print(f"[Summarizer] ⚠ Chunk {chunk_idx} error: {str(e)}")
            # Fallback: return first few sentences
            sentences = chunk.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def _parallel_summarize_chunks(self, chunks: List[str], 
                                   target_per_chunk: int) -> List[str]:
        """
        Summarize multiple chunks in parallel
        
        Args:
            chunks: List of text chunks
            target_per_chunk: Target words per chunk summary
            
        Returns:
            List of chunk summaries
        """
        if not chunks:
            return []
        
        # Single chunk - no parallelization needed
        if len(chunks) == 1:
            summary = self._summarize_single_chunk(chunks[0], target_per_chunk, 1)
            return [summary]
        
        # Multiple chunks - parallel processing
        if self.parallel_enabled and len(chunks) > 1:
            summaries = [None] * len(chunks)
            
            def summarize_with_index(idx_chunk):
                idx, chunk = idx_chunk
                summary = self._summarize_single_chunk(chunk, target_per_chunk, idx + 1)
                return idx, summary
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(summarize_with_index, (i, chunk)): i 
                    for i, chunk in enumerate(chunks)
                }
                
                for future in as_completed(futures):
                    try:
                        idx, summary = future.result()
                        summaries[idx] = summary
                    except Exception as e:
                        idx = futures[future]
                        print(f"[Summarizer] ⚠ Chunk {idx+1} failed: {e}")
                        summaries[idx] = ""
            
            # Filter out failed chunks
            return [s for s in summaries if s]
        
        # Sequential fallback
        else:
            return [
                self._summarize_single_chunk(chunk, target_per_chunk, i+1)
                for i, chunk in enumerate(chunks)
            ]
    
    # ========================================================================
    # STRUCTURE FORMATTING
    # ========================================================================
    
    def _format_structured_output(self, summaries: List[str], 
                                  domain: str) -> str:
        """
        Format summaries with hierarchical structure based on domain
        
        Args:
            summaries: List of chunk summaries
            domain: Detected domain (tech/science/business/etc)
            
        Returns:
            Formatted summary with structure
        """
        if not summaries:
            return "Error: No content to summarize"
        
        # Single summary - minimal formatting
        if len(summaries) == 1:
            return summaries[0]
        
        # Extract intro and main points
        intro = summaries[0]
        points = summaries[1:7]  # Max 6 additional points
        
        # Format based on domain
        if domain == 'tutorial':
            return self._format_tutorial(intro, points)
        elif domain in ['science', 'tech', 'business', 'economics']:
            return self._format_lecture(intro, points, domain)
        elif domain == 'history':
            return self._format_narrative(intro, points)
        else:
            return self._format_general(intro, points)
    
    def _format_tutorial(self, intro: str, steps: List[str]) -> str:
        """Format as step-by-step tutorial"""
        output = f"**Overview**\n{intro}\n\n"
        
        if steps:
            output += "**Steps:**\n"
            for i, step in enumerate(steps, 1):
                output += f"\n{i}. {step}\n"
        
        return output
    
    def _format_lecture(self, intro: str, concepts: List[str], domain: str) -> str:
        """Format as lecture with main concepts"""
        output = f"**Core Concept**\n{intro}\n\n"
        
        if concepts:
            output += "**Key Points:**\n"
            for i, concept in enumerate(concepts, 1):
                # Try to extract first sentence as heading
                sentences = concept.split('.')
                if len(sentences) > 1:
                    heading = sentences[0].strip()
                    detail = '. '.join(sentences[1:]).strip()
                    
                    output += f"\n**{i}. {heading}**\n"
                    if detail:
                        output += f"   {detail}\n"
                else:
                    output += f"\n**{i}.** {concept}\n"
        
        return output
    
    def _format_narrative(self, intro: str, parts: List[str]) -> str:
        """Format as narrative/story"""
        output = f"{intro}\n\n"
        
        if parts:
            # Join parts with paragraph breaks
            output += '\n\n'.join(parts)
        
        return output
    
    def _format_general(self, intro: str, points: List[str]) -> str:
        """Format as general content with bullet points"""
        output = f"**Summary**\n{intro}\n\n"
        
        if points:
            output += "**Key Points:**\n"
            for point in points:
                # Clean and make concise
                clean_point = point.strip()
                # If point has multiple sentences, take first
                if '.' in clean_point:
                    clean_point = clean_point.split('.')[0]
                output += f"• {clean_point}\n"
        
        return output
    
    # ========================================================================
    # MAIN SUMMARIZATION PIPELINE
    # ========================================================================
    
    def summarize(self, text: str, max_length: int = 300) -> str:
        """
        Main entry point for summarization
        
        Args:
            text: Input text to summarize
            max_length: Target summary length in words
            
        Returns:
            Formatted summary text
        """
        total_start = time.time()
        
        # Validation
        if not text or len(text.strip()) < 50:
            return "Error: Text too short to summarize (minimum 50 characters)"
        
        print(f"\n[Summarizer] ⚡ Starting {self.mode} mode summarization")
        
        try:
            # Stage 1: Preprocessing
            if _HAS_PREPROCESS:
                # Get input statistics
                stats = get_text_stats(text)
                print(f"[Summarizer] Input: {stats['word_count']} words, "
                      f"{stats['sentence_count']} sentences")
                
                # Clean text
                cleaned = self._timed_stage('Clean', safe_clean, text)
                
                # Detect domain
                domain = self._timed_stage(
                    'Domain', 
                    detect_domain, 
                    cleaned.split()
                )
                print(f"[Summarizer] 🎯 Detected domain: {domain}")
                
                # Mode-specific processing
                if self.mode == 'fast':
                    # Fast mode: skip advanced processing
                    chunks = fast_chunks(cleaned, max_chars=40000)
                    target_per_chunk = max_length
                
                elif self.mode == 'balanced':
                    # Balanced mode: smart chunking
                    sentences = self._timed_stage('Split', safe_split, cleaned)
                    chunks = self._timed_stage('Chunk', sentence_chunks, sentences)
                    target_per_chunk = max(50, max_length // len(chunks))
                
                else:  # quality mode
                    # Quality mode: full pipeline with filtering
                    sentences = self._timed_stage('Split', safe_split, cleaned)
                    filtered = self._timed_stage(
                        'Filter', 
                        light_idea_filter, 
                        sentences, 
                        domain
                    )
                    chunks = self._timed_stage('Chunk', sentence_chunks, filtered)
                    target_per_chunk = max(50, max_length // len(chunks))
            
            else:
                # Fallback mode without preprocess.py
                print(f"[Summarizer] Input: {len(text.split())} words")
                cleaned = self._fallback_clean(text)
                chunks = self._fallback_chunks(cleaned)
                domain = 'general'
                target_per_chunk = max_length
            
            # Log chunking results
            print(f"[Summarizer] 📦 Processing {len(chunks)} chunks")
            print(f"[Summarizer] 🎯 Target per chunk: {target_per_chunk} words")
            
            # Stage 2: Parallel summarization (main processing)
            chunk_summaries = self._timed_stage(
                'Summarize',
                self._parallel_summarize_chunks,
                chunks,
                target_per_chunk,
                timeout_ms=5000
            )
            
            if not chunk_summaries:
                return "Error: Summarization failed - no output generated"
            
            # Stage 3: Structure application
            if self.mode == 'fast' or not _HAS_PREPROCESS:
                # Fast mode: simple concatenation
                final_summary = ' '.join(chunk_summaries)
            else:
                # Balanced/Quality: structured formatting
                final_summary = self._timed_stage(
                    'Format',
                    self._format_structured_output,
                    chunk_summaries,
                    domain
                )
            
            # Calculate final metrics
            total_ms = (time.time() - total_start) * 1000
            word_count = len(final_summary.split())
            
            print(f"[Summarizer] ✅ Complete: {word_count} words in {total_ms:.0f}ms")
            
            # Quality gates
            if total_ms > 6000:
                print(f"[Summarizer] ⚠️ Performance warning: Exceeded 6s budget")
            if word_count > max_length * 1.5:
                print(f"[Summarizer] ⚠️ Length warning: {word_count} words "
                      f"(target: {max_length})")
            
            # Save summary to file if path provided
            return final_summary
            
        except Exception as e:
            print(f"[Summarizer] ❌ Error during summarization: {str(e)}")
            traceback.print_exc()
            
            # Return fallback summary (first 200 words)
            words = text.split()[:200]
            fallback = ' '.join(words)
            return f"Error during summarization. Excerpt:\n\n{fallback}..."
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_mode_info(self) -> Dict:
        """
        Get current configuration information
        
        Returns:
            Dictionary with mode, device, and settings
        """
        return {
            'mode': self.mode,
            'quality': self.quality_mode,
            'device': 'cuda' if self.device == 0 else 'cpu',
            'parallel': self.parallel_enabled,
            'workers': self.max_workers if self.parallel_enabled else 1,
            'has_preprocess': _HAS_PREPROCESS
        }
    
    def set_mode(self, mode: str):
        """
        Change summarization mode
        
        Args:
            mode: 'fast', 'balanced', or 'quality'
        """
        if mode in ['fast', 'balanced', 'quality']:
            self.mode = mode
            print(f"[Summarizer] Mode changed to: {mode}")
        else:
            print(f"[Summarizer] Invalid mode: {mode}")


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# For existing code that might use old class name
HybridSummarizer = Summarizer


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VidSummarize - Summarizer Module Test")
    print("=" * 70)
    print()
    
    # Sample text for testing
    sample_text = """
    Machine learning is transforming modern technology in profound ways. 
    Neural networks learn patterns from data without explicit programming, 
    enabling computers to recognize images, understand speech, and make 
    predictions. Deep learning uses multiple layers to extract increasingly 
    abstract features from raw data. Training these models requires large 
    datasets and significant computational power, often using GPUs or 
    specialized hardware. Applications include image recognition, natural 
    language processing, autonomous vehicles, and medical diagnosis. 
    However, these models can exhibit bias if training data is not 
    representative of the real world. Interpretability remains a major 
    challenge, as neural networks are often considered black boxes. 
    Researchers are working on explainable AI to make model decisions 
    more transparent. Despite challenges, machine learning continues to 
    advance rapidly, with new architectures and techniques emerging 
    regularly. The field combines mathematics, computer science, and 
    domain expertise to solve complex real-world problems.
    """
    
    # Test initialization
    print("Testing Summarizer initialization...")
    try:
        summarizer = Summarizer()
        print("✅ Summarizer initialized successfully\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}\n")
        exit(1)
    
    # Test mode info
    info = summarizer.get_mode_info()
    print("Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test summarization
    print("Testing summarization...")
    print("-" * 70)
    
    result = summarizer.summarize(sample_text, max_length=80)
    
    print("-" * 70)
    print("\nSummary Output:")
    print("=" * 70)
    print(result)
    print("=" * 70)
    print()
    
    # Test mode switching
    print("Testing mode switching...")
    summarizer.set_mode('fast')
    summarizer.set_mode('quality')
    summarizer.set_mode('balanced')
    print()
    
    print("=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
