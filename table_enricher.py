from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
from llama_wrapper import ChatLlamaCppWrapper
from thinking_utils import strip_thinking_tags


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

    def __init__(self, llm: Optional[ChatLlamaCppWrapper] = None):
        self._owns_llm = (llm is None)  # Track if we created the LLM
        if llm is None:
            # create LLM for table enrichment using wrapper
            model_path = str(Config.Model.GGUF_PATH.resolve())
            self.llm = ChatLlamaCppWrapper(
                model_path=model_path,
                temperature=0,  # deterministic
                n_ctx=2048,   # small - just need table headers/caption
                max_tokens=100, # short descriptions only
                n_gpu_layers=0,  # CPU only to save VRAM
                n_batch=Config.Model.N_BATCH,
                n_threads=Config.Model.N_THREADS,
                streaming=False,
                verbose=False,
            )
        else:
            self.llm = llm

        self.chain = TABLE_DESCRIPTION_PROMPT | self.llm | StrOutputParser()

    def cleanup(self):
        """Free LLM resources (equivalent to keep_alive=0 in Ollama)"""
        if self._owns_llm and self.llm is not None:
            del self.llm
            self.llm = None
            import gc
            gc.collect()

    def __del__(self):
        """Cleanup on garbage collection"""
        self.cleanup()

    def _extract_caption_and_headers(self, table_content: str) -> tuple[str, str]:
        """
        Extract caption and headers from table content.
        """
        lines = table_content.strip().split('\n')

        # find caption (usually first line before markdown table)
        caption = ""
        for i, line in enumerate(lines):
            if line.strip().startswith('|'):
                # Found table start
                if i > 0:
                    caption = lines[i-1].strip()
                break

        # extract headers (first row of markdown table)
        headers = ""
        for line in lines:
            if line.strip().startswith('|') and '---' not in line:
                headers = line.strip()
                break

        # clean headers
        headers = headers.replace('|', ',').strip(',').strip()

        return caption, headers

    def enrich_table(self, table_content: str) -> str:
        """
        Add semantic description to table content.
        """
        # skip if table is too short or already enriched
        if len(table_content) < 20 or "This table shows" in table_content:
            return table_content

        caption, headers = self._extract_caption_and_headers(table_content)

        # if no caption or headers, return as-is
        if not caption and not headers:
            return table_content

        try:
            # generate semantic description
            description = self.chain.invoke({
                'caption': caption if caption else "No caption",
                'headers': headers if headers else "No headers"
            })

            # CRITICAL: Strip thinking tags using robust utility
            description = strip_thinking_tags(description, aggressive=True)

            # insert description after caption, before table
            lines = table_content.strip().split('\n')

            # find where table starts
            table_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('|'):
                    table_start = i
                    break

            # build enriched content
            if table_start > 0:
                # caption exists, insert description after it
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
# global instance (lazy init)
_global_enricher: Optional[TableEnricher] = None

def get_table_enricher(llm=None) -> TableEnricher:
    """Get or create global table enricher"""
    global _global_enricher
    if _global_enricher is None:
        _global_enricher = TableEnricher(llm=llm)
    return _global_enricher

def cleanup_table_enricher():
    """Cleanup global table enricher (free LLM resources)"""
    global _global_enricher
    if _global_enricher is not None:
        _global_enricher.cleanup()
        del _global_enricher
        _global_enricher = None

def enrich_table_content(table_content: str, llm=None) -> str:
    """
    Convenience function to enrich a single table.
    """
    if not Config.Preprocessing.ENABLE_TABLE_SEMANTIC_ENRICHMENT:
        return table_content

    enricher = get_table_enricher(llm)
    return enricher.enrich_table(table_content)
