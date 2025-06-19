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

def extract_worksheet_intersection(xml_file: str) -> Optional[str]:
    """
    Extract WorksheetIntersection value from workbook XML file.
    Args:
        xml_file: Path to the XML file
    Returns:
        The WorksheetIntersection value or None if not found
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        ws_intx_elem = root.find(".//attribute[@name='WorksheetIntersection']/value")
        if ws_intx_elem is not None and ws_intx_elem.text:
            return ws_intx_elem.text.strip()
    except Exception as e:
        logging.warning(f"Error extracting WorksheetIntersection from {xml_file}: {e}")
    return None

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

def extract_worksheet_intersections(xml_file: str) -> List[Dict[str, str]]:
    """
    Extract all worksheet-level WorksheetIntersection values from a workbook XML file.
    Returns a list of dicts with worksheet_name and intersection value.
    """
    intersections = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Find all worksheet nodes
        for ws in root.findall(".//data_model[@type='java.lang.Object']"):
            ws_name_elem = ws.find("attribute[@name='name']/value")
            ws_intx_elem = ws.find("attribute[@name='WorksheetIntersection']/value")
            if ws_intx_elem is not None and ws_intx_elem.text:
                intersections.append({
                    "worksheet_name": ws_name_elem.text if ws_name_elem is not None else '',
                    "WorksheetIntersection": ws_intx_elem.text.strip()
                })
    except Exception as e:
        logging.warning(f"Error extracting worksheet intersections from {xml_file}: {e}")
    return intersections

def process_worksheet_intersections(workbooks_dir: str, mismatched_dimensions: set) -> pd.DataFrame:
    """
    Process all workbook XML files for WorksheetIntersection values (per worksheet).
    Only include those where the intersection contains a mismatched dimension as a whole word or substring (e.g., _str_, str_, or str).
    Returns DataFrame with xmlfilename, worksheet_name, WorksheetIntersection
    """
    data: List[Dict[str, str]] = []
    try:
        for filename in os.listdir(workbooks_dir):
            if not filename.endswith(".xml"):
                continue
            xml_file = os.path.join(workbooks_dir, filename)
            intersections = extract_worksheet_intersections(xml_file)
            for entry in intersections:
                ws_intx = entry["WorksheetIntersection"]
                # Check for any mismatched dimension as substring or whole word
                for mismatched in mismatched_dimensions:
                    if (
                        f"_{mismatched}_" in f"_{ws_intx}_" or
                        ws_intx.startswith(f"{mismatched}_") or
                        ws_intx.endswith(f"_{mismatched}") or
                        ws_intx == mismatched
                    ):
                        data.append({
                            "xmlfilename": os.path.splitext(filename)[0],
                            "worksheet_name": entry["worksheet_name"],
                            "WorksheetIntersection": ws_intx
                        })
                        break  # Only add once per intersection
    except Exception as e:
        logging.error(f"Error processing worksheet intersections: {e}")
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
        
        # Process workbooks (Wizard dims)
        df = process_workbooks("workbooks", mismatched_dimensions)
        
        # Process worksheet intersections (always include all, with match status)
        df_ws = process_worksheet_intersections("workbooks", mismatched_dimensions)
        
        with pd.ExcelWriter("output.xlsx") as writer:
            if not df.empty:
                df = df[["xmlfilename", "CLNDDim", "LOCDim", "PRODDim"]]
                df.to_excel(writer, sheet_name="workbook", index=False)
                logging.info(f"Found {len(df)} workbooks with mismatched dimensions")
            else:
                logging.info("No matching dimensions found in workbooks")
            # Always write worksheetintersection sheet
            if not df_ws.empty:
                df_ws = df_ws[["xmlfilename", "worksheet_name", "WorksheetIntersection"]]
                df_ws.to_excel(writer, sheet_name="worksheetintersection", index=False)
                logging.info(f"Found {len(df_ws)} worksheet intersections with mismatched dimensions")
            else:
                pd.DataFrame(columns=["xmlfilename", "worksheet_name", "WorksheetIntersection"]).to_excel(writer, sheet_name="worksheetintersection", index=False)
                logging.info("No WorksheetIntersection values found in workbooks")
        # Return both DataFrames for reference (optional)
        if not df.empty or not df_ws.empty:
            return df if not df.empty else df_ws
        else:
            logging.info("No matching dimensions or WorksheetIntersection found in workbooks. No Excel file created.")
            return None
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
    