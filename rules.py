import os
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Optional, Set
import logging
from hierarchy_paths import get_mismatched_dimensions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
KEYWORDS = ["tssum", "locsecurityusr", "forecast150", "java", "javaexpression", "[CLND].top"]

def check_keywords(formula: str, keywords: List[str]) -> str:
    """
    Check for invalid keywords in formula.
    
    Args:
        formula: Formula to check
        keywords: List of keywords to check for
        
    Returns:
        Comma-separated string of found keywords
    """
    return ', '.join(keyword for keyword in keywords if keyword.lower() in formula.lower())

def check_mismatched_dimensions(formula: str, mismatched_dims: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    """
    Check for mismatched dimensions in formula.
    
    Args:
        formula: Formula to check
        mismatched_dims: Dictionary of mismatched dimensions
        
    Returns:
        Comma-separated string of found dimensions or None
    """
    found_dims = []
    for hier_name, dims in mismatched_dims.items():
        for dim in dims['file2_only']:
            # Look for dimension in square brackets format
            if f"[{dim}]" in formula or f"[{dim.lower()}]" in formula:
                found_dims.append(f"{dim} (from {hier_name})")
    return ', '.join(found_dims) if found_dims else None

def process_rule_file(file_path: str, mismatched_dims: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, str]]:
    """
    Process a single rule XML file.
    
    Args:
        file_path: Path to the rule XML file
        mismatched_dims: Dictionary of mismatched dimensions
        
    Returns:
        List of dictionaries containing invalid rule data
    """
    invalid_data = []
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        rule_group_name = root.find("attribute[@name='name']/value").text or "Unknown"
        
        for rule in root.findall("data_model[@type='com.retek.ride.rule.newmodel.Rule']"):
            try:
                rule_name = rule.find("attribute[@name='name']/value").text or "Unknown"
                
                for data_model in rule.findall("data_model[@type='java.lang.String']"):
                    formula_element = data_model.find("attribute[@name='formula']/value")
                    if formula_element is not None:
                        formula = formula_element.text
                        invalid_functions = check_keywords(formula, KEYWORDS)
                        mismatched_dims_found = check_mismatched_dimensions(formula, mismatched_dims)
                        
                        if invalid_functions or mismatched_dims_found:
                            invalid_data.append({
                                "Rule Set": os.path.basename(os.path.dirname(file_path)),
                                "Rule Group": rule_group_name,
                                "Rule Name": rule_name,
                                "Rule Expression": formula,
                                "Invalid keywords": invalid_functions,
                                "Mismatched Dimensions": mismatched_dims_found
                            })
            except AttributeError as e:
                logging.warning(f"Error processing rule in {file_path}: {e}")
                continue
                
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file {file_path}: {e}")
    except FileNotFoundError:
        logging.error(f"Rule file not found: {file_path}")
    except Exception as e:
        logging.error(f"Unexpected error processing rule file {file_path}: {e}")
    
    return invalid_data

def parse_ruleset_folder(folder_path: str, keywords: List[str], output_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Parse ruleset folder and return results.
    
    Args:
        folder_path: Path to the ruleset folder
        keywords: List of keywords to check for
        output_dir: Optional directory to save the output Excel file.
        
    Returns:
        DataFrame containing invalid rule data or None if no data
    """
    invalid_data = []
    rulesets_path = os.path.join(folder_path, "rulesets")
    
    if not os.path.isdir(rulesets_path):
        raise ValueError("The 'rulesets' folder does not exist under the selected main folder.")

    # Get mismatched dimensions from hierarchy
    xml_file = 'hierarchy.xml'
    ga_file = 'ga_hierarchy.txt'
    mismatched_dims = get_mismatched_dimensions(xml_file, ga_file)

    try:
        for subfolder in os.listdir(rulesets_path):
            subfolder_path = os.path.join(rulesets_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            for file in os.listdir(subfolder_path):
                if file.endswith(".xml") and file != "rulesets.xml":
                    file_path = os.path.join(subfolder_path, file)
                    invalid_data.extend(process_rule_file(file_path, mismatched_dims))
    except Exception as e:
        logging.error(f"Error processing ruleset folder: {e}")
        raise

    if invalid_data:
        df = pd.DataFrame(invalid_data)
        
        output_path = 'invalid_rules.xlsx'
        if output_dir:
            output_path = os.path.join(output_dir, output_path)
            
        df.to_excel(output_path, sheet_name='rules', index=False)
        logging.info(f"Found {len(invalid_data)} invalid rules")
        return df
    else:
        logging.info("No invalid rules found")
        return None

if __name__ == "__main__":
    try:
        # For standalone execution, save in current directory
        parse_ruleset_folder(os.getcwd(), KEYWORDS, os.getcwd())
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise