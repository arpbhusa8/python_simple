import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Optional, Set
import logging
from hierarchy_paths import get_mismatched_dimensions
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
INVALID_VALUES = {'Scalar', '#MP_OUT#', 'user', 'system'}
REQUIRED_FIELDS = [
    'measure_name', 'measure_label', 'description', 'baseint', 
    'database', 'creator', 'clearint', 'loadint', 'defagg', 
    'defspread', 'filename', 'sharedfactname', 'loadagg', 
    'sharedfactbaseintx'
]

def is_valid_dimension_value(value: str) -> bool:
    """
    Check if the value is a valid dimension and not a special value.
    
    Args:
        value: Dimension value to check
        
    Returns:
        True if value is valid, False otherwise
    """
    return bool(value and value not in INVALID_VALUES)

def check_dimension_in_fields(measure_data: Dict[str, str], file2_only_dimensions: Set[str]) -> bool:
    """
    Check if any file2_only dimension appears in baseint, loadint, or clearint fields.
    
    Args:
        measure_data: Dictionary containing measure data
        file2_only_dimensions: Set of dimensions to check against
        
    Returns:
        True if any dimension is found, False otherwise
    """
    for field in ['baseint', 'loadint', 'clearint']:
        field_value = measure_data.get(field, '')
        if field_value and is_valid_dimension_value(field_value):
            dimensions = [dim.strip().lower() for dim in field_value.split(',')]
            if any(any(file2_dim.lower() in dimension for file2_dim in file2_only_dimensions) 
                  for dimension in dimensions):
                return True
    return False

def extract_measure_data(data_model: ET.Element) -> Optional[Dict[str, str]]:
    """
    Extract measure data from XML element.
    
    Args:
        data_model: XML element containing measure data
        
    Returns:
        Dictionary containing measure data or None if invalid
    """
    measure_data = {}
    
    try:
        # Extract all attributes
        for attribute in data_model.findall('.//attribute'):
            name = attribute.get('name')
            value = attribute.find('value')
            if value is not None:
                measure_data[name] = value.text.strip()
        
        # Process measure if it has a name
        if 'name' not in measure_data:
            return None
            
        measure_name = measure_data['name'].replace('#INHERITED#', '')
        if not measure_name:
            return None
            
        # Add all required fields with default empty string
        measure_data.update({
            'measure_name': measure_name,
            'measure_label': measure_data.get('label', ''),
            'description': measure_data.get('description', ''),
            'baseint': measure_data.get('baseint', ''),
            'database': measure_data.get('db', ''),
            'creator': measure_data.get('creator', ''),
            'clearint': measure_data.get('clearint', ''),
            'loadint': measure_data.get('loadint', ''),
            'defagg': measure_data.get('defagg', ''),
            'defspread': measure_data.get('defspread', ''),
            'filename': measure_data.get('filename', ''),
            'sharedfactname': measure_data.get('sharedfactname', ''),
            'loadagg': measure_data.get('loadagg', ''),
            'sharedfactbaseintx': measure_data.get('sharedfactbaseintx', '')
        })
        
        return measure_data
    except Exception as e:
        logging.warning(f"Error extracting measure data: {e}")
        return None

def parse_xml(xml_file: str, output_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Parse measures XML file and return results.
    
    Args:
        xml_file: Path to the XML file
        output_dir: Optional directory to save the output Excel file.
        
    Returns:
        DataFrame containing measure data or None if no data
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

    # Get mismatched dimensions from hierarchy_paths.py
    hierarchy_xml = 'hierarchy.xml'
    ga_file = 'ga_hierarchy.txt'
    mismatched_dimensions = get_mismatched_dimensions(hierarchy_xml, ga_file)
    file2_only_dimensions = {dim for hierarchy_data in mismatched_dimensions.values() 
                           for dim in hierarchy_data['file2_only']}

    data: List[Dict[str, str]] = []
    
    for data_model in root.findall('.//data_model'):
        measure_data = extract_measure_data(data_model)
        if measure_data and check_dimension_in_fields(measure_data, file2_only_dimensions):
            data.append(measure_data)

    if data:
        df = pd.DataFrame(data)
        df = df[REQUIRED_FIELDS]
        
        output_path = 'measures_with_mismatched_dimensions.xlsx'
        if output_dir:
            output_path = os.path.join(output_dir, output_path)
            
        df.to_excel(output_path, sheet_name='measures', index=False)
        logging.info(f"Found {len(data)} measures containing mismatched dimensions")
        return df
    else:
        logging.info("No measures found containing mismatched dimensions")
        return None

if __name__ == "__main__":
    try:
        # For standalone execution, save in current directory
        parse_xml('realmeasures.xml', os.getcwd())
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise