"""
Hierarchy Structure Analysis Module

This module provides functionality to analyze XML hierarchy structures,
extract user dimensions, and find optimal paths through hierarchical data.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging
import os
from datetime import datetime
from constants import IDEAL_SPINES
import pandas as pd

# Global variable to store mismatched dimensions (similar to hierarchy.py)
mismatched_dimensions: Dict[str, Dict[str, List[str]]] = {}


def setup_logging(log_level: str = "INFO") -> str:
    """
    Setup professional logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Path to the created log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"hierarchy_analysis_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()  # Console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    logger.info(f"Log level set to: {log_level.upper()}")
    
    return log_filepath


@dataclass
class HierarchyInfo:
    """Data class to store hierarchy information."""
    name: str
    rpas_name: str
    virthier: Optional[str]
    dimensions: List[str]


@dataclass
class DimensionInfo:
    """Data class to store dimension information."""
    name: str
    rpas_name: str
    userdim: bool
    aggs: List[str]


@dataclass
class LabeledIntersectionInfo:
    """Data class to store labeled intersection information."""
    name: str
    definition: str


class HierarchyAnalyzer:
    """Main class for analyzing hierarchy structures from XML files."""
    
    def __init__(self, xml_file_path: str, log_level: str = "INFO"):
        """
        Initialize the analyzer with an XML file path.
        
        Args:
            xml_file_path: Path to the XML file containing hierarchy data
            log_level: Logging level for the analyzer
        """
        self.xml_file_path = xml_file_path
        self._tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None
        self._dimension_cache: Dict[str, DimensionInfo] = {}
        self._hierarchy_cache: Dict[str, HierarchyInfo] = {}
        self._labeled_intersections_cache: Dict[str, LabeledIntersectionInfo] = {}
        self._analysis_results: Optional[Dict[str, Dict]] = None
        
        # Setup logging
        self.log_filepath = setup_logging(log_level)
        self.logger = logging.getLogger(f"{__name__}.HierarchyAnalyzer")
        self.logger.info(f"Initializing HierarchyAnalyzer with XML file: {xml_file_path}")
        
    def _load_xml(self) -> None:
        """Load and parse the XML file."""
        self.logger.info("Loading XML file...")
        try:
            self._tree = ET.parse(self.xml_file_path)
            self._root = self._tree.getroot()
            self.logger.info(f"Successfully loaded XML file. Root tag: {self._root.tag}")
        except ET.ParseError as e:
            self.logger.error(f"Invalid XML file: {e}")
            raise ValueError(f"Invalid XML file: {e}")
        except FileNotFoundError:
            self.logger.error(f"XML file not found: {self.xml_file_path}")
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading XML: {e}")
            raise
    
    def _extract_dimension_info(self, dim_element: ET.Element) -> DimensionInfo:
        """
        Extract dimension information from a DimClass element.
        
        Args:
            dim_element: XML element representing a dimension
            
        Returns:
            DimensionInfo object containing dimension details
        """
        attrs = {}
        
        # Extract all attributes efficiently
        for attr_elem in dim_element.findall("./attribute"):
            name = attr_elem.get("name")
            value_elem = attr_elem.find("value")
            if name and value_elem is not None and value_elem.text is not None:
                attrs[name] = value_elem.text.strip()
        
        # Extract required fields with defaults
        name = attrs.get("name", "")
        rpas_name = attrs.get("rpas_name", name)  # Don't modify case
        userdim = attrs.get("userdim", "false").lower() == "true"
        aggs_text = attrs.get("aggs", "")
        aggs = [agg.strip() for agg in aggs_text.split()] if aggs_text else []
        
        return DimensionInfo(name=name, rpas_name=rpas_name, userdim=userdim, aggs=aggs)
    
    def _build_dimension_cache(self) -> None:
        """Build a cache of all dimension information for efficient access."""
        if self._root is None:
            self._load_xml()
        
        self.logger.info("Building dimension cache...")
        dimension_count = 0
        userdim_count = 0
        
        for dim_element in self._root.findall(".//data_model[@type='DimClass']"):
            dim_info = self._extract_dimension_info(dim_element)
            if dim_info.name:
                self._dimension_cache[dim_info.name] = dim_info
                dimension_count += 1
                if dim_info.userdim:
                    userdim_count += 1
                    self.logger.debug(f"Found userdim dimension: {dim_info.name} (rpas_name: {dim_info.rpas_name})")
        
        self.logger.info(f"Dimension cache built: {dimension_count} total dimensions, {userdim_count} userdim dimensions")
    
    def _build_hierarchy_cache(self) -> None:
        """Build a cache of all hierarchy information."""
        if self._root is None:
            self._load_xml()
        
        self.logger.info("Building hierarchy cache...")
        hierarchy_count = 0
        
        for hier_element in self._root.findall(".//data_model[@type='HierClass']"):
            name_elem = hier_element.find("./attribute[@name='name']/value")
            if name_elem is None or name_elem.text is None:
                continue
                
            hier_name = name_elem.text.strip()
            
            # Extract rpas_name attribute
            rpas_name_elem = hier_element.find("./attribute[@name='rpas_name']/value")
            hier_rpas_name = rpas_name_elem.text.strip() if rpas_name_elem is not None and rpas_name_elem.text else hier_name
            
            # Extract virthier attribute
            virthier_elem = hier_element.find("./attribute[@name='virthier']/value")
            virthier = virthier_elem.text.strip() if virthier_elem is not None and virthier_elem.text else None
            
            # Extract dimensions
            dimensions = []
            for dim_elem in hier_element.findall(".//data_model[@type='DimClass']"):
                name_elem = dim_elem.find("./attribute[@name='name']/value")
                if name_elem is not None and name_elem.text:
                    dimensions.append(name_elem.text.strip())
            
            self._hierarchy_cache[hier_name] = HierarchyInfo(
                name=hier_name, rpas_name=hier_rpas_name, virthier=virthier, dimensions=dimensions
            )
            hierarchy_count += 1
            self.logger.debug(f"Found hierarchy: {hier_name} (rpas_name: {hier_rpas_name}, dimensions: {len(dimensions)})")
        
        self.logger.info(f"Hierarchy cache built: {hierarchy_count} hierarchies")
    
    def _build_labeled_intersections_cache(self) -> None:
        """Build a cache of all labeled intersection information."""
        if self._root is None:
            self._load_xml()
        
        self.logger.info("Building labeled intersections cache...")
        intersection_count = 0
        
        for labeled_intx in self._root.findall('.//data_model[@type="LabeledIntx"]'):
            try:
                name_elem = labeled_intx.find('.//attribute[@name="name"]/value')
                definition_elem = labeled_intx.find('.//attribute[@name="LabeledIntxDefinition"]/value')
                
                if name_elem is not None and name_elem.text and definition_elem is not None and definition_elem.text:
                    name = name_elem.text.strip()
                    definition = definition_elem.text.strip()
                    
                    self._labeled_intersections_cache[name] = LabeledIntersectionInfo(
                        name=name, definition=definition
                    )
                    intersection_count += 1
                    self.logger.debug(f"Found labeled intersection: {name}")
            except AttributeError as e:
                self.logger.warning(f"Error processing labeled intersection: {e}")
                continue
        
        self.logger.info(f"Labeled intersections cache built: {intersection_count} labeled intersections")

    def get_labeled_intersections(self) -> Dict[str, str]:
        """
        Get all labeled intersections as a dictionary.
        
        Returns:
            Dictionary mapping labeled intersection names to their definitions
        """
        if not self._labeled_intersections_cache:
            self._build_labeled_intersections_cache()
        
        labeled_intersections = {
            name: info.definition 
            for name, info in self._labeled_intersections_cache.items()
        }
        
        self.logger.info(f"Retrieved {len(labeled_intersections)} labeled intersections")
        return labeled_intersections

    def get_labeled_intersections_info(self) -> Dict[str, LabeledIntersectionInfo]:
        """
        Get all labeled intersections with full information.
        
        Returns:
            Dictionary mapping labeled intersection names to LabeledIntersectionInfo objects
        """
        if not self._labeled_intersections_cache:
            self._build_labeled_intersections_cache()
        
        return self._labeled_intersections_cache.copy()
    
    def get_userdim_dimensions(self) -> Set[str]:
        """
        Get all dimensions with userdim=true.
        
        Returns:
            Set of rpas_name values for dimensions with userdim=true
        """
        if not self._dimension_cache:
            self._build_dimension_cache()
        
        userdim_dimensions = {
            dim_info.rpas_name 
            for dim_info in self._dimension_cache.values() 
            if dim_info.userdim
        }
        
        self.logger.info(f"Retrieved {len(userdim_dimensions)} userdim dimensions")
        return userdim_dimensions
    
    def get_hierarchy_virthier_values(self) -> Dict[str, Optional[str]]:
        """
        Get virthier values for all hierarchies.
        
        Returns:
            Dictionary mapping hierarchy names to their virthier values
        """
        if not self._hierarchy_cache:
            self._build_hierarchy_cache()
        
        hierarchy_virthier = {
            hier_name: hier_info.virthier 
            for hier_name, hier_info in self._hierarchy_cache.items()
        }
        
        self.logger.info(f"Retrieved virthier values for {len(hierarchy_virthier)} hierarchies")
        return hierarchy_virthier
    
    def get_userdim_dimensions_by_hierarchy(self) -> Dict[str, Set[str]]:
        """
        Get all dimensions with userdim=true grouped by their parent hierarchy.
        
        Returns:
            Dictionary mapping hierarchy rpas_name to set of dimension rpas_name values
        """
        if not self._dimension_cache:
            self._build_dimension_cache()
        if not self._hierarchy_cache:
            self._build_hierarchy_cache()
        
        # Build mapping from dimension name to hierarchy rpas_name
        dim_to_hierarchy_rpas = {}
        for hier_name, hier_info in self._hierarchy_cache.items():
            for dim_name in hier_info.dimensions:
                dim_to_hierarchy_rpas[dim_name] = hier_info.rpas_name
        
        # Group userdim dimensions by hierarchy
        hierarchy_userdims = defaultdict(set)
        for dim_info in self._dimension_cache.values():
            if dim_info.userdim and dim_info.name in dim_to_hierarchy_rpas:
                hierarchy_rpas_name = dim_to_hierarchy_rpas[dim_info.name]
                hierarchy_userdims[hierarchy_rpas_name].add(dim_info.rpas_name)
        
        self.logger.info(f"Grouped userdim dimensions by {len(hierarchy_userdims)} hierarchies")
        for hierarchy, dimensions in hierarchy_userdims.items():
            self.logger.debug(f"Hierarchy {hierarchy}: {len(dimensions)} userdim dimensions")
        
        return dict(hierarchy_userdims)
    
    def generate_hierarchy_edges(self) -> List[Tuple[str, str]]:
        """
        Generate parent-child edges from the hierarchy structure.
        
        Returns:
            List of (parent, child) tuples representing hierarchy edges
        """
        if not self._dimension_cache:
            self._build_dimension_cache()
        
        self.logger.info("Generating hierarchy edges...")
        edges = []
        dim_to_hierarchy = {}
        
        # Build dimension to hierarchy mapping
        for hier_name, hier_info in self._hierarchy_cache.items():
            for dim_name in hier_info.dimensions:
                dim_to_hierarchy[dim_name] = hier_name
        
        # Generate edges from aggregation relationships
        for dim_info in self._dimension_cache.values():
            if dim_info.aggs:
                # Add edges from aggregation parents to this dimension
                for parent in dim_info.aggs:
                    edges.append((parent, dim_info.name))
            else:
                # If no aggregations, connect to hierarchy root
                if dim_info.name in dim_to_hierarchy:
                    edges.append((dim_to_hierarchy[dim_info.name], dim_info.name))
        
        self.logger.info(f"Generated {len(edges)} hierarchy edges")
        return edges
    
    def build_graph(self, edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """
        Build an adjacency list representation of the hierarchy graph.
        
        Args:
            edges: List of (parent, child) tuples
            
        Returns:
            Dictionary representing the graph as adjacency lists
        """
        self.logger.info("Building graph from edges...")
        graph = defaultdict(list)
        for parent, child in edges:
            graph[parent].append(child)
        
        graph_dict = dict(graph)
        self.logger.info(f"Graph built with {len(graph_dict)} nodes")
        return graph_dict
    
    def find_all_paths(self, graph: Dict[str, List[str]], start_node: str) -> List[List[str]]:
        """
        Find all possible paths from a starting node using DFS.
        
        Args:
            graph: Adjacency list representation of the graph
            start_node: Starting node for path finding
            
        Returns:
            List of all possible paths from start_node
        """
        self.logger.debug(f"Finding all paths from {start_node}")
        all_paths = []
        
        def dfs(node: str, current_path: List[str]) -> None:
            current_path.append(node)
            
            # If leaf node (no children), add current path
            if node not in graph or not graph[node]:
                all_paths.append(current_path[:])
            else:
                # Explore all children
                for neighbor in graph[node]:
                    dfs(neighbor, current_path)
            
            current_path.pop()
        
        dfs(start_node, [])
        self.logger.debug(f"Found {len(all_paths)} paths from {start_node}")
        return all_paths
    
    def count_consecutive_matches(self, path: List[str], ideal_spine: List[str]) -> int:
        """
        Count consecutive matches between a path and ideal spine.
        
        Args:
            path: Actual path to compare
            ideal_spine: Ideal spine to match against
            
        Returns:
            Number of consecutive matches from the beginning
        """
        matches = 0
        for actual, ideal in zip(path, ideal_spine):
            if actual.strip().lower() == ideal.strip().lower():
                matches += 1
            else:
                break
        return matches
    
    def find_best_matching_path(
        self, 
        graph: Dict[str, List[str]], 
        start_node: str, 
        ideal_spine: List[str]
    ) -> Tuple[List[str], int, List[str]]:
        """
        Find the best matching path for a given ideal spine.
        
        Args:
            graph: Adjacency list representation of the graph
            start_node: Starting node for path finding
            ideal_spine: Ideal spine to match against (without hierarchy name)
            
        Returns:
            Tuple of (best_path, match_score, non_matching_dimensions)
        """
        self.logger.debug(f"Finding best matching path for {start_node} with ideal spine: {ideal_spine}")
        
        # Get all possible paths first
        all_paths = self.find_all_paths(graph, start_node)
        
        if not all_paths:
            self.logger.debug(f"No paths found for {start_node}")
            return [], -1, []
        
        # Get the first path as default (will be used if no matches or first element doesn't match)
        first_path = all_paths[0]
        first_path_dimensions = first_path[1:] if len(first_path) > 1 else []
        
        # Check if first element from ideal spine matches first dimension from XML
        if not ideal_spine or not first_path_dimensions:
            self.logger.debug(f"Empty ideal spine or first path for {start_node}, using first path")
            return first_path, 0, []
        
        first_ideal = ideal_spine[0].strip().lower()
        first_actual = first_path_dimensions[0].strip().lower()
        
        # If first element doesn't match, take the first spine as main spine
        if first_ideal != first_actual:
            self.logger.debug(f"First element mismatch for {start_node}: expected '{first_ideal}', got '{first_actual}', using first path")
            non_matching = self.find_non_matching_dimensions(first_path_dimensions, ideal_spine)
            return first_path, 0, non_matching
        
        # Find best matching path with original logic
        best_path = []
        best_score = -1
        
        def dfs(node: str, current_path: List[str], ideal_idx: int) -> None:
            nonlocal best_path, best_score
            
            current_path.append(node)
            
            # Check if current node matches ideal spine (skip hierarchy name in comparison)
            if (ideal_idx < len(ideal_spine) and 
                node.strip().lower() == ideal_spine[ideal_idx].strip().lower()):
                ideal_idx += 1
            
            # If leaf node, evaluate path
            if node not in graph or not graph[node]:
                # Compare only the dimensions (skip hierarchy name)
                path_dimensions = current_path[1:] if len(current_path) > 1 else []
                score = self.count_consecutive_matches(path_dimensions, ideal_spine)
                if (score > best_score or 
                    (score == best_score and len(current_path) > len(best_path))):
                    best_score = score
                    best_path = current_path[:]
            else:
                # Explore all children
                for neighbor in graph[node]:
                    dfs(neighbor, current_path, ideal_idx)
            
            current_path.pop()
        
        dfs(start_node, [], 0)
        
        # If nothing matches at all, take the first spine
        if best_score == 0:
            self.logger.debug(f"No matches found for {start_node}, using first path")
            best_path = first_path
            best_score = 0
        
        # Find non-matching dimensions for the best path
        best_path_dimensions = best_path[1:] if len(best_path) > 1 else []
        non_matching_dimensions = self.find_non_matching_dimensions(best_path_dimensions, ideal_spine)
        
        self.logger.debug(f"Best path for {start_node}: score={best_score}, path={best_path}")
        return best_path, best_score, non_matching_dimensions
    
    def find_non_matching_dimensions(self, path_dimensions: List[str], ideal_spine: List[str]) -> List[str]:
        """
        Find dimensions that exist in path but don't match with ideal spine at all.
        
        Args:
            path_dimensions: Actual path dimensions (without hierarchy name)
            ideal_spine: Ideal spine to match against
            
        Returns:
            List of dimensions that don't exist in ideal spine
        """
        if not path_dimensions or not ideal_spine:
            return []
        
        # Convert to lowercase for comparison
        path_lower = [dim.strip().lower() for dim in path_dimensions]
        ideal_lower = [dim.strip().lower() for dim in ideal_spine]
        ideal_set = set(ideal_lower)
        
        # Find dimensions that don't exist in ideal spine at all
        non_matching = []
        for i, dim in enumerate(path_lower):
            if dim not in ideal_set:
                # This dimension doesn't exist in ideal spine at all
                non_matching.append(path_dimensions[i])  # Use original case
        
        return non_matching
    
    def analyze_hierarchies(self) -> Dict[str, Dict]:
        """
        Perform complete hierarchy analysis and return structured results.
        
        Returns:
            Dictionary containing analysis results for each hierarchy
        """
        self.logger.info("Starting hierarchy analysis...")
        
        # Extract userdim dimensions grouped by hierarchy
        userdim_by_hierarchy = self.get_userdim_dimensions_by_hierarchy()
        hierarchy_virthier = self.get_hierarchy_virthier_values()
        labeled_intersections = self.get_labeled_intersections()
        
        # Print userdim dimensions grouped by hierarchy
        print("\nDimensions (rpas_name) with userdim=true grouped by hierarchy:")
        for hierarchy_rpas, dimensions in sorted(userdim_by_hierarchy.items()):
            print(f"  {hierarchy_rpas}:")
            for dim_name in sorted(dimensions):
                print(f"    {dim_name}")
        
        # Print hierarchy virthier values
        print("\nHierarchy virthier attribute values:")
        for hier_name, virthier in sorted(hierarchy_virthier.items()):
            print(f"  {hier_name}: {virthier}")
        
        # Print labeled intersections
        print(f"\nLabeled intersections found: {len(labeled_intersections)}")
        for name, definition in sorted(labeled_intersections.items()):
            print(f"  {name}: {definition}")
        
        # Generate edges and build graph
        edges = self.generate_hierarchy_edges()
        graph = self.build_graph(edges)
        
        # Store results for each hierarchy
        hierarchy_results = {}
        
        # Analyze each hierarchy
        for hierarchy, ideal_spine in IDEAL_SPINES.items():
            self.logger.info(f"Analyzing hierarchy: {hierarchy}")
            print(f"\n--- Hierarchy: {hierarchy} ---")
            
            # Find all paths for this hierarchy
            all_paths = self.find_all_paths(graph, hierarchy)
            
            print("All possible paths:")
            for path in all_paths:
                # Format path: show hierarchy name with colon, then the rest
                formatted_path = f"{hierarchy}: {' -> '.join(path[1:])}" if len(path) > 1 else f"{hierarchy}:"
                print(f"  {formatted_path}")
            
            # Find best matching path - compare only from first dimension (skip hierarchy name)
            best_path, match_score, non_matching_dimensions = self.find_best_matching_path(
                graph, hierarchy, ideal_spine  # Use ideal_spine directly, not full_ideal_spine
            )
            
            # Compare with ideal spine and populate global mismatched_dimensions
            if best_path:
                actual_spine = best_path[1:] if len(best_path) > 1 else []
                self.compare_with_ideal_spine(hierarchy, actual_spine, ideal_spine)
            
            # Store results for this hierarchy
            hierarchy_results[hierarchy] = {
                'userdim_dimensions': list(userdim_by_hierarchy.get(hierarchy, set())),
                'main_spine': best_path[1:] if len(best_path) > 1 else [],  # Remove hierarchy name
                'virtual_hierarchy': hierarchy_virthier.get(hierarchy),
                'non_matching_dimensions': non_matching_dimensions,
                'match_score': match_score,
                'all_paths': [path[1:] for path in all_paths]  # Remove hierarchy names
            }
            
            if best_path:
                # Format best path: show hierarchy name with colon, then the rest
                formatted_best_path = f"{hierarchy}: {' -> '.join(best_path[1:])}" if len(best_path) > 1 else f"{hierarchy}:"
                print(f"\nMain Spine (best match): {formatted_best_path}")
                print(f"Consecutive matches with ideal spine: {match_score}")
                
                # Display non-matching dimensions if any
                if non_matching_dimensions:
                    print(f"Non-matching dimensions (match with GA but not in order): {', '.join(non_matching_dimensions)}")
                else:
                    print("Non-matching dimensions: None")
            else:
                print("\nMain Spine: No paths found for this hierarchy.")
        
        self.logger.info("Hierarchy analysis completed successfully")
        self._analysis_results = hierarchy_results
        return hierarchy_results
    
    def get_hierarchy_variables(self, hierarchy_name: str) -> Optional[Dict]:
        """
        Get stored variables for a specific hierarchy.
        
        Args:
            hierarchy_name: Name of the hierarchy (e.g., 'CLND', 'PROD', 'LOC')
            
        Returns:
            Dictionary containing hierarchy variables or None if not found
        """
        if self._analysis_results is None:
            self.logger.warning("Analysis not performed yet. Call analyze_hierarchies() first.")
            return None
        
        return self._analysis_results.get(hierarchy_name)
    
    def get_all_hierarchy_variables(self) -> Optional[Dict[str, Dict]]:
        """
        Get stored variables for all hierarchies.
        
        Returns:
            Dictionary containing all hierarchy variables or None if analysis not performed
        """
        if self._analysis_results is None:
            self.logger.warning("Analysis not performed yet. Call analyze_hierarchies() first.")
            return None
        
        return self._analysis_results
    
    def get_mismatched_dimensions(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the dictionary containing mismatched dimensions.
        
        Returns:
            Dictionary with structure:
            {
                'hierarchy_name': {
                    'ga_only': [dimensions only in ideal spine],
                    'file2_only': [dimensions only in XML file]
                }
            }
        """
        return mismatched_dimensions
    
    def compare_with_ideal_spine(self, hierarchy_name: str, actual_spine: List[str], ideal_spine: List[str]) -> None:
        """
        Compare actual spine with ideal spine and populate global mismatched_dimensions.
        
        Args:
            hierarchy_name: Name of the hierarchy
            actual_spine: Actual spine from XML file
            ideal_spine: Ideal spine from constants
        """
        global mismatched_dimensions
        
        if hierarchy_name not in mismatched_dimensions:
            mismatched_dimensions[hierarchy_name] = {
                'ga_only': [],      # only in ideal spine
                'file2_only': []   # only in XML file
            }
        
        # Convert to lowercase for comparison
        actual_lower = [dim.strip().lower() for dim in actual_spine]
        ideal_lower = [dim.strip().lower() for dim in ideal_spine]
        
        # Find dimensions only in ideal spine
        actual_set = set(actual_lower)
        ideal_set = set(ideal_lower)
        
        ga_only = [ideal_spine[i] for i, dim in enumerate(ideal_lower) if dim not in actual_set]
        file2_only = [actual_spine[i] for i, dim in enumerate(actual_lower) if dim not in ideal_set]
        
        mismatched_dimensions[hierarchy_name]['ga_only'] = ga_only
        mismatched_dimensions[hierarchy_name]['file2_only'] = file2_only
        
        # Print messages for dimensions that match but not in order
        matching_out_of_order = []
        for i, dim in enumerate(actual_lower):
            if dim in ideal_set:
                expected_pos = ideal_lower.index(dim)
                if i != expected_pos:
                    matching_out_of_order.append(actual_spine[i])
        
        if matching_out_of_order:
            self.logger.info(f"Dimensions in {hierarchy_name} that match with ideal spine but not in order: {matching_out_of_order}")
            print(f"⚠️  Dimensions in {hierarchy_name} that match with ideal spine but not in order: {', '.join(matching_out_of_order)}")
    
    def write_hierarchy_data_to_excel(self, output_dir: Optional[str] = None) -> None:
        """
        Write detailed hierarchy data to Excel file with one sheet per hierarchy containing multiple tables.
        
        Args:
            output_dir: Optional directory to save the output Excel file
        """
        if self._analysis_results is None:
            self.logger.warning("Analysis not performed yet. Call analyze_hierarchies() first.")
            return
        
        output_path = 'hierarchy_analysis_results.xlsx'
        if output_dir:
            output_path = os.path.join(output_dir, output_path)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write labeled intersections sheet first
            labeled_intersections = self.get_labeled_intersections()
            if labeled_intersections:
                labeled_data = [
                    {'Labeled Intersection': name, 'Definition': definition}
                    for name, definition in labeled_intersections.items()
                ]
                labeled_df = pd.DataFrame(labeled_data)
                labeled_df.to_excel(writer, sheet_name='Labeled_Intersections', index=False)
                self.logger.info(f"Written {len(labeled_intersections)} labeled intersections to Excel")
            
            for hierarchy, results in self._analysis_results.items():
                # Get ideal spine for this hierarchy
                ideal_spine = IDEAL_SPINES.get(hierarchy, [])
                main_spine = results['main_spine']
                
                # Create one sheet per hierarchy with multiple tables
                sheet_name = hierarchy
                
                # 1. Spine Comparison Table
                spine_comparison_data = []
                max_length = max(len(ideal_spine), len(main_spine))
                
                for i in range(max_length):
                    ideal_dim = ideal_spine[i] if i < len(ideal_spine) else ''
                    customer_dim = main_spine[i] if i < len(main_spine) else ''
                    
                    # Check if dimensions match (case-insensitive)
                    matches = (ideal_dim.lower() == customer_dim.lower()) if ideal_dim and customer_dim else False
                    match_indicator = '✓' if matches else '❌'
                    
                    spine_comparison_data.append({
                        'Level': i + 1,
                        'Ideal Spine': ideal_dim,
                        'Customer Spine': customer_dim,
                        'Matches': match_indicator
                    })
                
                spine_df = pd.DataFrame(spine_comparison_data)
                spine_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
                
                # 2. User Defined Dimensions Table (below spine comparison)
                userdim_data = [{
                    'Hierarchy': hierarchy,
                    'User Defined Dimensions': ', '.join(results['userdim_dimensions']),
                    'Match Score': results['match_score'],
                    'Virtual Hierarchy': results['virtual_hierarchy'] or 'None'
                }]
                
                userdim_df = pd.DataFrame(userdim_data)
                userdim_df.to_excel(writer, sheet_name=sheet_name, startrow=len(spine_comparison_data) + 3, startcol=0, index=False)
                
                # 3. All Dimensions Table (below user defined)
                all_dimensions = []
                for i, dim in enumerate(main_spine):
                    all_dimensions.append({
                        'Position': i + 1,
                        'Dimension Name': dim,
                        'In Ideal Spine': 'Yes' if dim.lower() in [d.lower() for d in ideal_spine] else 'No'
                    })
                
                all_dim_df = pd.DataFrame(all_dimensions)
                all_dim_df.to_excel(writer, sheet_name=sheet_name, startrow=len(spine_comparison_data) + len(userdim_data) + 6, startcol=0, index=False)
                
                # 4. All Customer Paths Table (below all dimensions)
                all_paths_data = []
                
                # Get all paths for this hierarchy
                all_paths = results['all_paths']
                
                for path_idx, path in enumerate(all_paths):
                    # Determine if this is the main spine
                    is_main_spine = path == main_spine
                    spine_type = "Main Spine" if is_main_spine else f"Alternate Path {path_idx + 1}"
                    
                    # Create a row for each path
                    path_str = ' -> '.join(path) if path else 'None'
                    all_paths_data.append({
                        'Path Type': spine_type,
                        'Complete Path': path_str,
                        'Length': len(path),
                        'Is Main Spine': 'Yes' if is_main_spine else 'No'
                    })
                
                all_paths_df = pd.DataFrame(all_paths_data)
                all_paths_df.to_excel(writer, sheet_name=sheet_name, startrow=len(spine_comparison_data) + len(userdim_data) + len(all_dimensions) + 9, startcol=0, index=False)
        
        self.logger.info(f"Detailed hierarchy data written to: {output_path}")
        print(f"📊 Detailed hierarchy data written to: {output_path}")
        print(f"   - Created one sheet per hierarchy: {', '.join(self._analysis_results.keys())}")
        print(f"   - Each sheet contains: Spine Comparison, User Defined, All Dimensions, All Customer Paths tables")
        if labeled_intersections:
            print(f"   - Added Labeled_Intersections sheet with {len(labeled_intersections)} intersections")


def extract_info(xml_file: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Extract hierarchy and labeled intersection information from XML file.
    This function maintains compatibility with the old hierarchy.py interface.
    
    Args:
        xml_file: Path to the XML file
        
    Returns:
        Tuple containing:
        - Dictionary of hierarchies with their dimensions
        - Dictionary of labeled intersections
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file {xml_file}: {e}")
        raise
    except FileNotFoundError:
        logging.error(f"XML file not found: {xml_file}")
        raise

    hierarchies: Dict[str, List[str]] = {}
    labeled_intersections: Dict[str, str] = {}

    # Extract hierarchies
    for hier in root.findall('.//data_model[@type="HierClass"]'):
        try:
            hier_name = hier.find('.//attribute[@name="rpas_name"]/value').text
            if hier_name not in ['CLND', 'PROD', 'LOC']:
                continue

            dimensions: List[str] = []
            dim_aggs: Dict[str, str] = {}

            for dim in hier.findall('.//data_model[@type="DimClass"]'):
                dim_name = dim.find('.//attribute[@name="rpas_name"]/value').text
                aggs = dim.find('.//attribute[@name="aggs"]/value').text
                dimensions.append(dim_name)
                dim_aggs[dim_name] = aggs

            main_spine: List[str] = []
            current_dim = dimensions[0] if dimensions else None

            while current_dim:
                main_spine.append(current_dim)
                current_dim = next((dim for dim, aggs in dim_aggs.items() if aggs == current_dim), None)

            hierarchies[hier_name.lower()] = main_spine
        except AttributeError as e:
            logging.warning(f"Error processing hierarchy: {e}")
            continue

    # Extract labeled intersections
    for labeled_intx in root.findall('.//data_model[@type="LabeledIntx"]'):
        try:
            name = labeled_intx.find('.//attribute[@name="name"]/value').text
            value = labeled_intx.find('.//attribute[@name="LabeledIntxDefinition"]/value').text
            labeled_intersections[name] = value
        except AttributeError as e:
            logging.warning(f"Error processing labeled intersection: {e}")
            continue

    return hierarchies, labeled_intersections


def compare_hierarchies(hier1: Dict[str, List[str]], hier2: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Compare two hierarchies and identify matches and mismatches.
    This function maintains compatibility with the old hierarchy.py interface.
    
    Args:
        hier1: First hierarchy dictionary
        hier2: Second hierarchy dictionary
        
    Returns:
        Dictionary containing comparison results with match indicators
    """
    global mismatched_dimensions
    comparison: Dict[str, List[Tuple[str, str, str]]] = {}
    mismatched_dimensions = {}

    for hier_name in set(hier1.keys()) & set(hier2.keys()):
        comparison[hier_name] = []
        mismatched_dimensions[hier_name] = {
            'ga_only': [],
            'file2_only': []
        }
        i, j = 0, 0

        while i < len(hier1[hier_name]) or j < len(hier2[hier_name]):
            if i < len(hier1[hier_name]) and j < len(hier2[hier_name]):
                if hier1[hier_name][i] == hier2[hier_name][j]:
                    comparison[hier_name].append((hier1[hier_name][i], hier2[hier_name][j], '✓'))
                    i += 1
                    j += 1
                else:
                    if hier1[hier_name][i] in hier2[hier_name][j:]:
                        comparison[hier_name].append(('', hier2[hier_name][j], '❌'))
                        mismatched_dimensions[hier_name]['file2_only'].append(hier2[hier_name][j])
                        j += 1
                    elif hier2[hier_name][j] in hier1[hier_name][i:]:
                        comparison[hier_name].append((hier1[hier_name][i], '', '❌'))
                        mismatched_dimensions[hier_name]['ga_only'].append(hier1[hier_name][i])
                        i += 1
                    else:
                        comparison[hier_name].append((hier1[hier_name][i], hier2[hier_name][j], '❌'))
                        mismatched_dimensions[hier_name]['ga_only'].append(hier1[hier_name][i])
                        mismatched_dimensions[hier_name]['file2_only'].append(hier2[hier_name][j])
                        i += 1
                        j += 1
            elif i < len(hier1[hier_name]):
                comparison[hier_name].append((hier1[hier_name][i], '', '❌'))
                mismatched_dimensions[hier_name]['ga_only'].append(hier1[hier_name][i])
                i += 1
            else:
                comparison[hier_name].append(('', hier2[hier_name][j], '❌'))
                mismatched_dimensions[hier_name]['file2_only'].append(hier2[hier_name][j])
                j += 1

    return comparison


def get_mismatched_dimensions() -> Dict[str, Dict[str, List[str]]]:
    """
    Get the dictionary containing mismatched dimensions.
    This function maintains compatibility with the old hierarchy.py interface.
    
    Returns:
        Dictionary with structure:
        {
            'hierarchy_name': {
                'ga_only': [dimensions only in ideal spine],
                'file2_only': [dimensions only in XML file]
            }
        }
    """
def main():
    """Main function to run the hierarchy analysis."""
    try:
        # Setup logging first
        log_filepath = setup_logging("INFO")
        logger = logging.getLogger(__name__)
        
        logger.info("Starting hierarchy structure analysis")
        logger.info(f"IDEAL_SPINES configuration: {list(IDEAL_SPINES.keys())}")
        
        # Initialize analyzer
        analyzer = HierarchyAnalyzer("hierarchy.xml", "INFO")
        
        # Perform analysis
        hierarchy_results = analyzer.analyze_hierarchies()
        
        # Write hierarchy data to Excel
        analyzer.write_hierarchy_data_to_excel()
        
        # Display mismatched dimensions
        mismatched_dims = analyzer.get_mismatched_dimensions()
        print("\n" + "="*60)
        print("MISMATCHED DIMENSIONS (for Measures.py)")
        print("="*60)
        
        for hierarchy, data in mismatched_dims.items():
            print(f"\n{hierarchy}:")
            print(f"  Dimensions only in ideal spine: {data['ga_only']}")
            print(f"  Dimensions only in XML file: {data['file2_only']}")
        
        # Demonstrate accessing stored variables for each hierarchy
        print("\n" + "="*60)
        print("STORED VARIABLES FOR EACH HIERARCHY")
        print("="*60)
        
        for hierarchy, results in hierarchy_results.items():
            print(f"\n--- {hierarchy} Hierarchy Variables ---")
            
            # Access userdim dimensions
            userdim_dims = results['userdim_dimensions']
            print(f"Userdim dimensions: {userdim_dims}")
            
            # Access main spine
            main_spine = results['main_spine']
            print(f"Main spine: {' -> '.join(main_spine) if main_spine else 'None'}")
            
            # Access virtual hierarchy
            virtual_hier = results['virtual_hierarchy']
            print(f"Virtual hierarchy: {virtual_hier}")
            
            # Access non-matching dimensions
            non_matching = results['non_matching_dimensions']
            print(f"Non-matching dimensions: {non_matching}")
            
            # Access match score
            match_score = results['match_score']
            print(f"Match score: {match_score}")
            
            # Access all paths
            all_paths = results['all_paths']
            print(f"Total paths found: {len(all_paths)}")
        
        logger.info("Analysis completed successfully")
        logger.info(f"Log file saved to: {log_filepath}")
        
        return 0
        
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
