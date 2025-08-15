#!/usr/bin/env python3
"""
Script to analyze .el files and determine nv (number of vertices) and ne (number of edges)
for DGAP simulation parameters.

Usage:
    python analyze_el_files.py [base_file] [dynamic_file]
    
If no arguments provided, uses default files:
    - sx-mathoverflow-unique-undir.base.el
    - sx-mathoverflow-unique-undir.dynamic.el
"""

import sys
import os
from typing import Set, Tuple, Dict
from collections import defaultdict


def analyze_el_file(filepath: str) -> Tuple[int, int, Dict[int, int]]:
    """
    Analyze an .el file to determine number of vertices and edges.
    
    Args:
        filepath: Path to the .el file
        
    Returns:
        Tuple of (num_vertices, num_edges, vertex_degrees)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    vertices: Set[int] = set()
    edges: int = 0
    vertex_degrees: Dict[int, int] = defaultdict(int)
    
    print(f"Analyzing file: {filepath}")
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Parse edge (u, v)
                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Line {line_num} has unexpected format: '{line}'")
                    continue
                    
                u, v = int(parts[0]), int(parts[1])
                vertices.add(u)
                vertices.add(v)
                edges += 1
                
                # Count degrees (in-degree and out-degree for directed graphs)
                vertex_degrees[u] += 1
                vertex_degrees[v] += 1
                
            except ValueError as e:
                print(f"Warning: Line {line_num} contains invalid data: '{line}' - {e}")
                continue
    
    num_vertices = len(vertices)
    num_edges = edges
    
    return num_vertices, num_edges, dict(vertex_degrees)


def print_analysis_results(filename: str, num_vertices: int, num_edges: int, 
                          vertex_degrees: Dict[int, int]) -> None:
    """Print analysis results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Analysis Results for: {filename}")
    print(f"{'='*60}")
    print(f"Number of vertices (nv): {num_vertices:,}")
    print(f"Number of edges (ne): {num_edges:,}")
    print(f"Average degree: {num_edges * 2 / num_vertices:.2f}")
    
    # Show degree distribution
    if vertex_degrees:
        degree_counts = defaultdict(int)
        for degree in vertex_degrees.values():
            degree_counts[degree] += 1
        
        print(f"\nDegree distribution:")
        for degree in sorted(degree_counts.keys()):
            count = degree_counts[degree]
            percentage = (count / len(vertex_degrees)) * 100
            print(f"  Degree {degree}: {count:,} vertices ({percentage:.1f}%)")
    
    print(f"{'='*60}")


def main():
    """Main function to analyze .el files."""
    # Default files
    default_base = "rl4sys/examples/dgap/sx-mathoverflow-unique-undir.base.el"
    default_dynamic = "rl4sys/examples/dgap/sx-mathoverflow-unique-undir.dynamic.el"
    
    # Parse command line arguments
    if len(sys.argv) == 3:
        base_file = sys.argv[1]
        dynamic_file = sys.argv[2]
    elif len(sys.argv) == 1:
        base_file = default_base
        dynamic_file = default_dynamic
    else:
        print("Usage: python analyze_el_files.py [base_file] [dynamic_file]")
        print("If no arguments provided, uses default files:")
        print(f"  Base: {default_base}")
        print(f"  Dynamic: {default_dynamic}")
        sys.exit(1)
    
    print("DGAP .el File Analyzer")
    print("=" * 60)
    
    try:
        # Analyze base file
        print(f"\nAnalyzing base graph file...")
        base_nv, base_ne, base_degrees = analyze_el_file(base_file)
        print_analysis_results(base_file, base_nv, base_ne, base_degrees)
        
        # Analyze dynamic file
        print(f"\nAnalyzing dynamic graph file...")
        dynamic_nv, dynamic_ne, dynamic_degrees = analyze_el_file(dynamic_file)
        print_analysis_results(dynamic_file, dynamic_nv, dynamic_ne, dynamic_degrees)
        
        # Combined analysis
        print(f"\n{'='*60}")
        print("COMBINED ANALYSIS")
        print(f"{'='*60}")
        print(f"Total unique vertices across both files: {len(set(list(base_degrees.keys()) + list(dynamic_degrees.keys()))):,}")
        print(f"Total edges across both files: {base_ne + dynamic_ne:,}")
        
        # DGAP parameters recommendation
        print(f"\n{'='*60}")
        print("DGAP PARAMETERS RECOMMENDATION")
        print(f"{'='*60}")
        print(f"For base graph: --nv {base_nv} --ne {base_ne}")
        print(f"For dynamic graph: --nv {dynamic_nv} --ne {dynamic_ne}")
        print(f"For combined simulation: --nv {max(base_nv, dynamic_nv)} --ne {base_ne + dynamic_ne}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 