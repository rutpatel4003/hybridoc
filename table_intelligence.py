from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
import json

@dataclass
class TableCell:
    """
    Represents a single table cell
    """
    content: str
    row: int
    col: int

@dataclass 
class StructuredTable:
    """
    Structured representation of a table
    """
    headers: List[str]
    rows: List[Dict[str, str]]
    raw_markdown: str
    num_rows: int
    num_cols: int

    def to_dict(self) -> dict:
        """
        Convert to dictionary for metadata storage
        """
        return {
            'headers': self.headers,
            'rows': self.rows,
            'raw_markdown': self.raw_markdown,
            'num_rows': self.num_rows,
            'num_cols': self.num_cols
        }
    
    def to_searchable_text(self) -> str:
        """
        Convert table to searchable plain text
        """
        lines = []
        lines.append('TABLE HEADERS: ' + ' | '.join(self.headers)) # add headers
        for i, row in enumerate(self.rows):
            row_text = f'Row {i+1}: '
            row_parts = []
            for header, value in row.items():
                if value.strip():
                    row_parts.append(f'{header}={value}')
            lines.append(row_text + ', '.join(row_parts))

        return '\n'.join(lines)
    

class TableExtractor:
    """
    Extracts and parses tables from OCR text
    """
    @staticmethod
    def detect_table(text: str) -> bool:
        """
        Detect if text contains a markdown table
        """
        lines = text.split('\n') # check for makrdown table patterns
        pipe_lines = sum(1 for line in lines if '|' in line) # count lines with pipe separator
        has_separator = any(
            re.search(r'\|[\s]*[-:]+[\s]*\|', line) or  #
            re.match(r'^\s*\|(\s*-+\s*\|)+\s*$', line)   
            for line in lines
        )
        return pipe_lines >= 3 and has_separator
    
    @staticmethod
    def extract_markdown_table(text: str) -> Optional[str]:
        """
        Extract markdown table from text containing other content
        """
        lines = text.split('\n')
        table_lines = []
        in_table = False
        for l in lines:
            stripped = l.strip()
            
            # skip empty lines at the start
            if not stripped and not in_table:
                continue
                
            # detect table start (line with pipes)
            if '|' in l and not in_table:
                in_table = True
                table_lines.append(l)
            elif '|' in l and in_table:
                table_lines.append(l)
            elif in_table and '|' not in l:
                # empty line or non-table content - end of table
                if stripped:  # if there's content without pipes, table has ended
                    break
                # allow empty lines within table
                    
        if len(table_lines) >= 3:
            return '\n'.join(table_lines)
        
        return None
    
    @staticmethod
    def parse_markdown_table(md_table: str) -> Optional[StructuredTable]:
        """
        Parse markdown table into structured format
        """
        try:
            lines = [l.strip() for l in md_table.split('\n') if l.strip() and '|' in l]
            
            if len(lines) < 2:  # need at least header + separator (data optional)
                return None
            
            # Pparse header line
            header_line = lines[0]
            headers = [h.strip() for h in header_line.split('|') if h.strip()]
            
            if not headers:
                return None
            
            # find separator line and skip it
            data_start = 1
            for i, line in enumerate(lines[1:], 1):
                # check if this is a separator line (contains only |, -, :, spaces)
                cleaned = line.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')
                if not cleaned:  
                    data_start = i + 1
                    break
            
            # parse data rows
            rows = []
            for line in lines[data_start:]:
                cells = [c.strip() for c in line.split('|')]
                
                # remove empty strings from start/end (from leading/trailing |)
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]

                if len(cells) == len(headers):
                    row_dict = dict(zip(headers, cells))
                    rows.append(row_dict)
                elif len(cells) > 0:
                    # pad or truncate to match headers
                    while len(cells) < len(headers):
                        cells.append('')
                    cells = cells[:len(headers)]
                    row_dict = dict(zip(headers, cells))
                    rows.append(row_dict)
            
            return StructuredTable(
                headers=headers,
                rows=rows,
                raw_markdown=md_table,
                num_rows=len(rows),
                num_cols=len(headers)
            )
        
        except Exception as e:
            print(f"Failed to parse markdown table: {e}")
            return None
        
    @staticmethod
    def extract_table_from_ocr(ocr_text: str) -> Optional[Tuple[StructuredTable, str]]:
        """
        Extract and parse table from OCR output
        """
        if not TableExtractor.detect_table(ocr_text):
            return None
        
        md_table = TableExtractor.extract_markdown_table(ocr_text)

        if not md_table:
            return None
    
        structured = TableExtractor.parse_markdown_table(md_table=md_table)

        if structured:
            return (structured, ocr_text)
        
        return None
    
    @staticmethod
    def format_table_for_llm(table: StructuredTable) -> str:
        """
        Format structured table for LLM consumption
        """
        output = []
        output.append('TABLE DATA')
        output.append(f"Columns: {', '.join(table.headers)}")
        output.append(f'Rows: {table.num_rows}')
        output.append('')

        # add structured data
        for i, row in enumerate(table.rows):
            output.append(f'Row {i+1}:')
            for header, value in row.items():
                if value.strip():
                    output.append(f"  - {header}: {value}")
        output.append("END TABLE")

        return "\n".join(output)

    @staticmethod
    def detect_figure(text: str) -> bool:
        """
        Detect if OCR text represents a figure/diagram (not a table).
        """
        text_lower = text.lower()
        
        # positive indicators of figure content
        figure_indicators = [
            r'figure\s+\d+',
            r'fig\.\s*\d+',
            r'diagram',
            r'illustration',
            r'architecture',
            r'→|←|↑|↓',  # arrows common in diagrams
            r'input.*output',
            r'layer\s*\d+',
            r'\bx_\d+\b|\by_\d+\b',  # mathematical notation
        ]
        
        for pattern in figure_indicators:
            if re.search(pattern, text_lower):
                return True
        
        # if text is short and doesn't look like a table, it's likely a figure
        lines = text.strip().split('\n')
        if len(lines) < 3 and len(text) < 200:
            return True
        
        return False
    
class TableQueryHelper:
    """
    Helper for querying structured tables
    """
    @staticmethod
    def find_value_in_table(table: StructuredTable, column: str, row_match: Dict[str, str]) -> Optional[str]:
        """
        Find a specific value in table
        """
        for row in table.rows:
            # check if all row matches all conditions
            match = all(
                row.get(k, "").lower() == v.lower() for k, v in row_match.items()
            )
            if match:
                return row.get(column, None)
            
        return None
    
    @staticmethod
    def filter_rows(table: StructuredTable, conditions: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Filter table rows based on conditions
        """
        matching_rows = []
        for row in table.rows:
            match = all(
                conditions.get(k, '').lower() in row.get(k, '').lower() for k in conditions.keys()
            )

            if match:
                matching_rows.append(row)

        return matching_rows
    
    @staticmethod
    def get_column_values(table: StructuredTable, column: str) -> List[str]:
        """
        Get all values from a specific column
        """
        return [row.get(column, "" ) for row in table.rows if row.get(column, '')]
    
    @staticmethod
    def search_table(table: StructuredTable, query: str) -> List[Dict[str, str]]:
        """
        Search entire table for query string
        """
        query_lower = query.lower()
        matching_rows = []
        for r in table.rows:
            if any(query_lower in str(value).lower() for value in r.values()):
                matching_rows.append(r)
        return matching_rows


            
    





