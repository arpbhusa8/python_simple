import os
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Optional
import logging
from hierarchy_paths import get_mismatched_dimensions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_attributes(xml_file: str) -> Dict[str, str]:
    """
    Extract attributes from workbook XML file.
    
    Args:
        xml_file: Path to the XML file
        
    Returns:
        Dictionary containing workbook attributes
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

    attributes = {"CLNDDim": "", "LOCDim": "", "PRODDim": ""}
    
    try:
        for data_model in root.findall(".//data_model[@type='Wizard']"):
            name = data_model.find(".//attribute[@name='name']/value").text
            if name == "CLNDWizard":
                attributes["CLNDDim"] = data_model.find(".//attribute[@name='CLNDDim']/value").text if data_model.find(".//attribute[@name='CLNDDim']/value") is not None else ""
            elif name == "LOCWizard":
                attributes["LOCDim"] = data_model.find(".//attribute[@name='LOCDim']/value").text if data_model.find(".//attribute[@name='LOCDim']/value") is not None else ""
            elif name == "PRODWizard":
                attributes["PRODDim"] = data_model.find(".//attribute[@name='PRODDim']/value").text if data_model.find(".//attribute[@name='PRODDim']/value") is not None else ""
    except AttributeError as e:
        logging.warning(f"Error processing workbook attributes: {e}")
    
    return attributes

def process_workbooks(workbooks_dir: str, mismatched_dimensions: set) -> pd.DataFrame:
    """
    Process all workbook XML files in the specified directory.
    
    Args:
        workbooks_dir: Directory containing workbook XML files
        mismatched_dimensions: Set of mismatched dimensions to check against
        
    Returns:
        DataFrame containing workbook data
    """
    data: List[Dict[str, str]] = []
    
    try:
        for filename in os.listdir(workbooks_dir):
            if not filename.endswith(".xml"):
                continue
                
            xml_file = os.path.join(workbooks_dir, filename)
            attributes = extract_attributes(xml_file)
            attributes["xmlfilename"] = os.path.splitext(filename)[0]
            
            # Only include rows where at least one dimension matches a mismatched dimension
            if (attributes["CLNDDim"] in mismatched_dimensions or 
                attributes["LOCDim"] in mismatched_dimensions or 
                attributes["PRODDim"] in mismatched_dimensions):
                data.append(attributes)
    except Exception as e:
        logging.error(f"Error processing workbooks directory: {e}")
        raise
    
    return pd.DataFrame(data)

def main() -> Optional[pd.DataFrame]:
    """
    Main function to process workbooks and return results.
    
    Returns:
        DataFrame containing workbook analysis results or None if no data
    """
    try:
        # Get mismatched dimensions from hierarchy comparison
        xml_file2 = 'hierarchy.xml'
        ga_file = 'ga_hierarchy.txt'
        
        # Use hierarchy_paths.py functions instead of hierarchy.py
        mismatched_dims = get_mismatched_dimensions(xml_file2, ga_file)
        
        # Collect all mismatched dimensions from file2_only
        mismatched_dimensions = set()
        for hier_name, data in mismatched_dims.items():
            mismatched_dimensions.update(data['file2_only'])
        
        # Process workbooks
        df = process_workbooks("workbooks", mismatched_dimensions)
        
        if not df.empty:
            df = df[["xmlfilename", "CLNDDim", "LOCDim", "PRODDim"]]
            df.to_excel("output.xlsx", sheet_name="workbook", index=False)
            logging.info(f"Found {len(df)} workbooks with mismatched dimensions")
            return df
        else:
            logging.info("No matching dimensions found in workbooks")
            return None
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
    