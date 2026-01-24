"""
Table Semantic Enrichment

Adds rich semantic descriptions to tables to improve retrieval.
Tables often have captions like "Table 3: Regularization" but the actual
table data doesn't contain searchable keywords. This module generates
natural language descriptions that make tables findable.

Example:
    Caption: "Table 3: Regularization"
    Markdown: "| Layer | Dropout | Label Smoothing |"

    Enriched: "Table 3: Regularization
               This table shows regularization hyperparameters including
               dropout rates and label smoothing values for different layers.

               | Layer | Dropout | Label Smoothing |"
"""

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import Config


TABLE_DESCRIPTION_PROMPT = ChatPromptTemplate.from_template(
    """You are analyzing a table from a technical document.

Given the table caption and headers, write a 1-2 sentence description of what this table contains.
Focus on WHAT KIND OF DATA it shows, not detailed values.

Rules:
- Start with "This table shows..." or "This table compares..."
- Mention key column names
- Be specific about the type of data (e.g., "hyperparameters", "performance metrics", "configurations")
- Keep it under 50 words
- Do NOT include values from the table, only describe what it contains

Caption: {caption}
Headers: {headers}

Description:"""
)


class TableEnricher:
    """
    Enriches table chunks with semantic descriptions for better retrieval.
    """

    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Args:
            llm: Optional LLM for description generation
        """
        if llm is None:
            self.llm = ChatOllama(
                model=Config.Model.NAME,
                temperature=0,
                num_ctx=1024,
                num_predict=100,
                keep_alive=-1,
            )
        else:
            self.llm = llm

        self.chain = TABLE_DESCRIPTION_PROMPT | self.llm | StrOutputParser()

    def _extract_caption_and_headers(self, table_content: str) -> tuple[str, str]:
        """
        Extract caption and headers from table content.

        Args:
            table_content: Full table text (caption + markdown)

        Returns:
            (caption, headers_str)
        """
        lines = table_content.strip().split('\n')

        # Find caption (usually first line before markdown table)
        caption = ""
        for i, line in enumerate(lines):
            if line.strip().startswith('|'):
                # Found table start
                if i > 0:
                    caption = lines[i-1].strip()
                break

        # Extract headers (first row of markdown table)
        headers = ""
        for line in lines:
            if line.strip().startswith('|') and '---' not in line:
                headers = line.strip()
                break

        # Clean headers
        headers = headers.replace('|', ',').strip(',').strip()

        return caption, headers

    def enrich_table(self, table_content: str) -> str:
        """
        Add semantic description to table content.

        Args:
            table_content: Original table text (caption + markdown)

        Returns:
            Enriched table text with semantic description
        """
        # Skip if table is too short or already enriched
        if len(table_content) < 20 or "This table shows" in table_content:
            return table_content

        caption, headers = self._extract_caption_and_headers(table_content)

        # If no caption or headers, return as-is
        if not caption and not headers:
            return table_content

        try:
            # Generate semantic description
            description = self.chain.invoke({
                'caption': caption if caption else "No caption",
                'headers': headers if headers else "No headers"
            })

            # Clean thinking tags
            if '</think>' in description:
                description = description.split('</think>')[-1].strip()

            description = description.strip()

            # Insert description after caption, before table
            lines = table_content.strip().split('\n')

            # Find where table starts
            table_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('|'):
                    table_start = i
                    break

            # Build enriched content
            if table_start > 0:
                # Caption exists, insert description after it
                enriched_lines = (
                    lines[:table_start] +
                    [description, ""] +
                    lines[table_start:]
                )
            else:
                # No caption, add description at top
                enriched_lines = [description, ""] + lines

            return '\n'.join(enriched_lines)

        except Exception as e:
            print(f"    Table enrichment failed: {e}")
            return table_content


# Global instance (lazy init)
_global_enricher: Optional[TableEnricher] = None


def get_table_enricher(llm=None) -> TableEnricher:
    """Get or create global table enricher"""
    global _global_enricher
    if _global_enricher is None:
        _global_enricher = TableEnricher(llm=llm)
    return _global_enricher


def enrich_table_content(table_content: str, llm=None) -> str:
    """
    Convenience function to enrich a single table.

    Args:
        table_content: Table text to enrich
        llm: Optional LLM instance

    Returns:
        Enriched table text
    """
    if not Config.Preprocessing.ENABLE_TABLE_SEMANTIC_ENRICHMENT:
        return table_content

    enricher = get_table_enricher(llm)
    return enricher.enrich_table(table_content)
