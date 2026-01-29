from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass

def normalize_sci(s: str) -> str:
        """Converts OCR-spaced scientific notation (e.g., '1 . 0 · 10 20') to standard '1.0e20'."""
        if not isinstance(s, str):
            return s
        # only collapse spaces around operators when there are digits nearby
        # this prevents "model, we" → "model,we" while still fixing "1 , 234" → "1,234"
        # 1. fix spaced multiplication/dot in scientific notation context
        s2 = re.sub(r'(\d)\s*([·x*X])\s*(\d)', r'\1\2\3', s)
        # 2. handle pattern: Coefficient [Symbol] 10 [Whitespace] Exponent
        pattern = r'(\d+(?:[.,]\d+)?)\s*[·x*X]?\s*10\s*([+-]?\d+)'
        def replace_match(m):
            coeff = m.group(1).replace(',', '.')
            exp = m.group(2)
            return f"{coeff}e{exp}"
        return re.sub(pattern, replace_match, s2)

def normalize_text(text: str) -> str:
    """
    Normalize OCR artifacts in text:
    - '0 . 9' → '0.9'
    - '10 -9' → '10^-9'
    - 'β 1' → 'β1'
    - 'ϵ ls' → 'ϵ_ls'
    - 'P drop' → 'P_drop'
    - 'warmup _ steps' → 'warmup_steps'
    Preserves intentional spacing in regular prose.
    """
    if not isinstance(text, str):
        return text

    # 1. fix spaced decimals: '0 . 9' → '0.9', '3 . 5' → '3.5'
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)

    # 2. fix scientific notation: '1 . 0 · 10 20' → '1.0e20'
    text = normalize_sci(text)

    # 3. fix negative exponents: '10 -9' → '10^-9'
    text = re.sub(r'(\d+)\s+-\s*(\d+)(?=\s|[.,;:]|$)', r'\1^-\2', text)

    # 4. fix Greek letters with numeric subscripts: 'β 1' → 'β1'
    text = re.sub(r'([α-ωΑ-Ωϐ-Ͽἀ-῾])\s+(\d+)', r'\1\2', text)

    # 5. fix Greek letters with word subscripts: 'ϵ ls' → 'ϵ_ls'
    text = re.sub(r'([α-ωΑ-Ωϐ-Ͽἀ-῾])\s+([a-z]+)(?=\s|[=.,;:]|$)', r'\1_\2', text)
    # 6. fix variable subscripts ONLY when preceded by '=' or at start of string
    # this prevents "a rate" → "a_rate" while still catching "P drop = " → "P_drop = "
    # patterns: "= P drop", "P drop =", or start with capital letter + space + lowercase
    text = re.sub(r'(=\s*)([A-Z])\s+([a-z]+)\b', r'\1\2_\3', text)  # "= P drop" → "= P_drop"
    text = re.sub(r'\b([A-Z])\s+([a-z]+)(?=\s*=)', r'\1_\2', text)  # "P drop =" → "P_drop ="
    # also catch single lowercase letter with underscore: "d model", "d ff"
    text = re.sub(r'\b([a-z])\s+([a-z]+)(?=\s*=)', r'\1_\2', text)  # "d model =" → "d_model ="

    # 7. fix spaced underscores: 'warmup _ steps' → 'warmup_steps'
    text = re.sub(r'(\w+)\s+_\s+(\w+)', r'\1_\2', text)
    return text   

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
        Convert table to searchable plain text with explicit column headers.
        """
        lines = []
        lines.append('Headers: ' + ', '.join(self.headers))

        for i, row in enumerate(self.rows):
            row_parts = [f'{h}={normalize_sci(str(v))}' for h, v in row.items() if str(v).strip()]
            lines.append(f'Row {i+1}: ' + ', '.join(row_parts))
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
            
            # parse header line
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
                    row_dict = {h: normalize_sci(c) for h, c in zip(headers, cells)}
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
