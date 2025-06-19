import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from hierarchy_paths import get_mismatched_dimensions, extract_labeled_intersections, output_hierarchy_analysis, read_ga_hierarchy_from_file
import workbook
import Measures
import rules

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
XML_FILES = {
    'hierarchy': 'hierarchy.xml',
    'ga': 'ga_hierarchy.txt'
}

def run_hierarchy_comparison() -> Tuple[Dict, Dict]:
    """
    Run hierarchy comparison and return results.
    
    Returns:
        Tuple containing comparison results and labeled intersections
    """
    try:
        # Get detailed hierarchy analysis using hierarchy_paths.py
        ga_hierarchies = read_ga_hierarchy_from_file(XML_FILES['ga'])
        
        # This will generate the detailed analysis and save to hierarchy_analysis.xlsx
        # We'll read from that file to get the detailed data
        output_hierarchy_analysis(XML_FILES['hierarchy'], ga_hierarchies)
        
        # Get labeled intersections
        labeled_intersections = extract_labeled_intersections(XML_FILES['hierarchy'])
        
        # Return a placeholder for comparison (we'll use the Excel file directly)
        comparison = {'detailed_analysis': True}
        
        return comparison, labeled_intersections
    except Exception as e:
        logging.error(f"Error in hierarchy comparison: {e}")
        raise

def combine_excel_files(comparison: Dict, labeled_intersections: Dict) -> None:
    """
    Combine all analysis results into a single Excel file.
    
    Args:
        comparison: Hierarchy comparison results
        labeled_intersections: Labeled intersections data
    """
    try:
        sheets_written = 0
        with pd.ExcelWriter('combined_analysis.xlsx') as writer:
            # Read and write workbook data
            if os.path.exists('output.xlsx'):
                try:
                    workbook_excel = pd.ExcelFile('output.xlsx')
                    for sheet_name in workbook_excel.sheet_names:
                        df_workbook = pd.read_excel('output.xlsx', sheet_name=sheet_name)
                        if not df_workbook.empty:
                            df_workbook.to_excel(writer, sheet_name=f'workbook_{sheet_name.lower()}', index=False)
                            sheets_written += 1
                    os.remove('output.xlsx')
                except Exception as e:
                    logging.warning(f"Skipping output.xlsx due to error: {e}")
            
            # Read and write measures data
            if os.path.exists('measures_with_mismatched_dimensions.xlsx'):
                try:
                    df_measures = pd.read_excel('measures_with_mismatched_dimensions.xlsx')
                    if not df_measures.empty:
                        df_measures.to_excel(writer, sheet_name='measures_analysis', index=False)
                        sheets_written += 1
                    os.remove('measures_with_mismatched_dimensions.xlsx')
                except Exception as e:
                    logging.warning(f"Skipping measures_with_mismatched_dimensions.xlsx due to error: {e}")
            
            # Read and write rules data
            if os.path.exists('invalid_rules.xlsx'):
                try:
                    df_rules = pd.read_excel('invalid_rules.xlsx')
                    if not df_rules.empty:
                        df_rules.to_excel(writer, sheet_name='rules_analysis', index=False)
                        sheets_written += 1
                    os.remove('invalid_rules.xlsx')
                except Exception as e:
                    logging.warning(f"Skipping invalid_rules.xlsx due to error: {e}")
            
            # Read and write detailed hierarchy analysis from hierarchy_paths.py
            if os.path.exists('hierarchy_analysis.xlsx'):
                try:
                    hierarchy_excel = pd.ExcelFile('hierarchy_analysis.xlsx')
                    for sheet_name in hierarchy_excel.sheet_names:
                        df_hier = pd.read_excel('hierarchy_analysis.xlsx', sheet_name=sheet_name)
                        if not df_hier.empty:
                            df_hier.to_excel(writer, sheet_name=f'hierarchy_{sheet_name.lower()}', index=False)
                            sheets_written += 1
                    os.remove('hierarchy_analysis.xlsx')
                except Exception as e:
                    logging.warning(f"Skipping hierarchy_analysis.xlsx due to error: {e}")
            
            # Write labeled intersections if there is anything to write
            if labeled_intersections and len(labeled_intersections) > 0:
                df_labeled = pd.DataFrame(list(labeled_intersections.items()), 
                                        columns=['Labeled Intersection', 'Value'])
                if not df_labeled.empty:
                    df_labeled.to_excel(writer, sheet_name='labeled_intersections', index=False)
                    sheets_written += 1
            
            if sheets_written == 0:
                logging.info("No analysis results to write. No Excel file created.")
                return
        logging.info("Successfully created combined_analysis.xlsx")
    except Exception as e:
        logging.error(f"Error combining Excel files: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate the analysis process.
    """
    try:
        # Step 1: Run hierarchy comparison
        logging.info("Starting hierarchy comparison...")
        comparison, labeled_intersections = run_hierarchy_comparison()
        
        # Step 2: Run workbook analysis
        logging.info("Starting workbook analysis...")
        workbook.main()
        
        # Step 3: Run measures analysis
        logging.info("Starting measures analysis...")
        Measures.parse_xml('realmeasures.xml')
        
        # Step 4: Run rules analysis
        logging.info("Starting rules analysis...")
        rules.parse_ruleset_folder(os.getcwd(), rules.KEYWORDS)
        
        # Step 5: Combine all Excel files
        logging.info("Combining results into single Excel file...")
        combine_excel_files(comparison, labeled_intersections)
        
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 