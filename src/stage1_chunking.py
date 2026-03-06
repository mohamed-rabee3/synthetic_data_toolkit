import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.llm_client import UnifiedLLMClient
from src.utils import (
    count_tokens,
    load_prompt,
    read_json,
    render_prompt,
    write_json,
    write_jsonl,
)

logger = logging.getLogger("pipeline.stage1")


@dataclass
class Sentence:
    text: str
    source_file: str
    is_heading: bool = False
    heading_level: int | None = None
    heading_hierarchy: str = ""


@dataclass
class Chunk:
    chunk_id: str
    chunk_text: str
    source_file: str
    section_hierarchy: str
    token_count: int
    contains_table: bool
    chunk_index: int
    is_oversized: bool = False

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "source_file": self.source_file,
            "section_hierarchy": self.section_hierarchy,
            "token_count": self.token_count,
            "contains_table": self.contains_table,
            "chunk_index": self.chunk_index,
            "is_oversized": self.is_oversized,
        }


# ---------------------------------------------------------------------------
# 1a. Document Summary Generation
# ---------------------------------------------------------------------------


async def generate_summaries(
    md_files: list[Path],
    llm_client: UnifiedLLMClient,
    config: dict,
) -> dict[str, str]:
    summary_cfg = config["chunking"]["summary"]
    if not summary_cfg.get("enabled", True):
        logger.info("Summary generation disabled, skipping")
        return {}

    prompts_dir = config["paths"]["prompts_dir"]
    system_prompt = load_prompt(prompts_dir, summary_cfg["system_prompt_file"])
    user_template = load_prompt(prompts_dir, summary_cfg["user_prompt_file"])
    max_input_tokens = summary_cfg.get("max_input_tokens", 4000)
    tokenizer_cfg = config["chunking"]["tokenizer"]

    summaries = {}
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        tokens = count_tokens(text, tokenizer_cfg)
        if tokens > max_input_tokens:
            ratio = max_input_tokens / tokens
            text = text[: int(len(text) * ratio)]

        user_prompt = render_prompt(
            user_template,
            source_file=md_file.name,
            document_text=text,
        )

        response = await llm_client.call(
            role="judge",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=summary_cfg.get("temperature", 0.3),
            max_tokens=summary_cfg.get("max_tokens", 300),
        )

        if response.success and response.text.strip():
            summaries[md_file.name] = response.text.strip()
            logger.info("Generated summary for %s", md_file.name)
        else:
            logger.warning(
                "Failed to generate summary for %s: %s",
                md_file.name,
                response.error,
            )
            summaries[md_file.name] = ""

    return summaries


# ---------------------------------------------------------------------------
# 1b. Sentence Boundary Detection
# ---------------------------------------------------------------------------

_ABBREVIATION_PATTERN: re.Pattern | None = None
_HORIZONTAL_RULE_RE = re.compile(r"^[-*_]{3,}\s*$")
_MARKDOWN_IMAGE_RE = re.compile(r"^!\[.*\]")
_OCR_ARTIFACT_PATTERNS: list[re.Pattern] | None = None


def _build_abbreviation_regex(abbreviations: list[str]) -> re.Pattern:
    escaped = [re.escape(a.rstrip(".")) for a in abbreviations]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\.", re.IGNORECASE)


def _compile_ocr_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


# ---------------------------------------------------------------------------
# 1b-pre. Document-level text pre-processing
# ---------------------------------------------------------------------------


def preprocess_text(text: str, config: dict) -> str:
    """Strip document-level noise (page headers, code blocks) before chunking."""
    preproc_cfg = (
        config["chunking"]["sentence_detection"].get("preprocessing", {})
    )

    lines = text.split("\n")

    if preproc_cfg.get("remove_code_blocks", False):
        filtered: list[str] = []
        in_code_block = False
        for raw_line in lines:
            if raw_line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if not in_code_block:
                filtered.append(raw_line)
        lines = filtered

    repeat_threshold = preproc_cfg.get("repeating_line_threshold", 0)
    if repeat_threshold > 0:
        line_counts: dict[str, int] = {}
        for raw_line in lines:
            key = raw_line.strip()
            if not key:
                continue
            # Protect HTML tags, headings, and table pipe rows from removal
            if key.startswith("<") or key.startswith("#") or key.startswith("|"):
                continue
            line_counts[key] = line_counts.get(key, 0) + 1

        repeating = {ln for ln, cnt in line_counts.items() if cnt >= repeat_threshold}
        if repeating:
            before = len(lines)
            lines = [l for l in lines if l.strip() not in repeating]
            logger.info(
                "Removed %d repeating line patterns (%d lines, threshold=%d)",
                len(repeating),
                before - len(lines),
                repeat_threshold,
            )

    boilerplate = set(preproc_cfg.get("boilerplate_lines", []))
    if boilerplate:
        lines = [l for l in lines if l.strip() not in boilerplate]

    return "\n".join(lines)


def detect_sentences(text: str, source_file: str, config: dict) -> list[Sentence]:
    global _ABBREVIATION_PATTERN, _OCR_ARTIFACT_PATTERNS
    sent_cfg = config["chunking"]["sentence_detection"]
    filter_cfg = sent_cfg.get("line_filtering", {})

    abbreviations = sent_cfg.get("abbreviations", [])
    if _ABBREVIATION_PATTERN is None and abbreviations:
        _ABBREVIATION_PATTERN = _build_abbreviation_regex(abbreviations)

    if _OCR_ARTIFACT_PATTERNS is None and filter_cfg.get("ocr_artifact_patterns"):
        _OCR_ARTIFACT_PATTERNS = _compile_ocr_patterns(
            filter_cfg["ocr_artifact_patterns"]
        )

    skip_hr = filter_cfg.get("skip_horizontal_rules", False)
    skip_images = filter_cfg.get("skip_markdown_images", False)
    skip_ocr = filter_cfg.get("skip_ocr_artifacts", False)

    lines = text.split("\n")
    sentences: list[Sentence] = []
    heading_stack: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if skip_hr and _HORIZONTAL_RULE_RE.match(stripped):
            continue

        if skip_images and _MARKDOWN_IMAGE_RE.match(stripped):
            continue

        # OCR artifact check runs BEFORE heading/list detection so it can
        # catch noise lines regardless of their syntactic shape.
        if skip_ocr and _OCR_ARTIFACT_PATTERNS:
            if any(p.search(stripped) for p in _OCR_ARTIFACT_PATTERNS):
                logger.debug("Skipped OCR artifact: %.60s", stripped)
                continue

        # Heading detection
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack = [
                h for h in heading_stack if not h.startswith(f"h{level}:")
            ]
            heading_stack = [
                h for h in heading_stack
                if int(h.split(":")[0][1:]) < level
            ]
            heading_stack.append(f"h{level}:{heading_text}")
            hierarchy = " > ".join(h.split(":", 1)[1] for h in heading_stack)

            sentences.append(
                Sentence(
                    text=stripped,
                    source_file=source_file,
                    is_heading=True,
                    heading_level=level,
                    heading_hierarchy=hierarchy,
                )
            )
            continue

        # List item detection
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            hierarchy = (
                " > ".join(h.split(":", 1)[1] for h in heading_stack)
                if heading_stack
                else ""
            )
            sentences.append(
                Sentence(
                    text=stripped,
                    source_file=source_file,
                    heading_hierarchy=hierarchy,
                )
            )
            continue

        # Sentence splitting within paragraphs
        paragraph = stripped

        # Mask abbreviations
        masked = paragraph
        if _ABBREVIATION_PATTERN:
            masked = _ABBREVIATION_PATTERN.sub(
                lambda m: m.group().replace(".", "\x00"), masked
            )

        # Mask decimal numbers
        if sent_cfg.get("ignore_decimal_numbers", True):
            masked = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x00" + m.group(2), masked)

        # Mask URL dots
        if sent_cfg.get("ignore_url_dots", True):
            masked = re.sub(
                r"(https?://\S+|www\.\S+)",
                lambda m: m.group().replace(".", "\x00"),
                masked,
            )

        # Split on sentence boundaries
        endings = r"[.!?]"
        if sent_cfg.get("arabic_sentence_endings", True):
            endings = r"[.!?؟۔]"

        parts = re.split(rf"({endings})(\s+|$)", masked)

        hierarchy = (
            " > ".join(h.split(":", 1)[1] for h in heading_stack)
            if heading_stack
            else ""
        )

        current = ""
        for part in parts:
            current += part
            unmasked = current.replace("\x00", ".")
            if re.search(rf"{endings}\s*$", current.strip()):
                if unmasked.strip():
                    sentences.append(
                        Sentence(
                            text=unmasked.strip(),
                            source_file=source_file,
                            heading_hierarchy=hierarchy,
                        )
                    )
                current = ""

        remainder = current.replace("\x00", ".").strip()
        if remainder:
            sentences.append(
                Sentence(
                    text=remainder,
                    source_file=source_file,
                    heading_hierarchy=hierarchy,
                )
            )

    return sentences


# ---------------------------------------------------------------------------
# 1c. Table Isolation
# ---------------------------------------------------------------------------


@dataclass
class TableBlock:
    text: str
    preceding_heading: str
    start_line: int
    end_line: int


def extract_tables(text: str, config: dict) -> tuple[list[TableBlock], str]:
    table_cfg = config["chunking"]["table_detection"]
    if not table_cfg.get("enabled", True):
        return [], text

    min_rows = table_cfg.get("min_rows", 3)
    lines = text.split("\n")
    tables: list[TableBlock] = []
    table_line_ranges: list[tuple[int, int]] = []
    last_heading = ""

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if re.match(r"^#{1,6}\s+", line):
            last_heading = line

        # --- Markdown pipe tables ---
        if line.startswith("|") and line.endswith("|"):
            table_start = i
            table_lines = [lines[i]]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith("|") and next_line.endswith("|"):
                    table_lines.append(lines[j])
                    j += 1
                else:
                    break
            table_end = j

            has_separator = any(
                re.match(r"^\|[\s\-:]+\|", tl.strip()) for tl in table_lines
            )

            if len(table_lines) >= min_rows and has_separator:
                table_text = "\n".join(table_lines)
                prefix = ""
                if table_cfg.get("include_preceding_heading", True) and last_heading:
                    prefix = last_heading + "\n\n"

                tables.append(
                    TableBlock(
                        text=prefix + table_text,
                        preceding_heading=last_heading,
                        start_line=table_start,
                        end_line=table_end,
                    )
                )
                table_line_ranges.append((table_start, table_end))
                i = table_end
                continue

        # --- HTML tables (<table>...</table>) ---
        if re.match(r"<table[\s>]", line, re.IGNORECASE):
            table_start = i
            html_lines = [lines[i]]
            j = i + 1
            while j < len(lines):
                html_lines.append(lines[j])
                if re.search(r"</table>", lines[j], re.IGNORECASE):
                    j += 1
                    break
                j += 1
            table_end = j

            row_count = sum(
                1 for tl in html_lines if re.search(r"<tr[\s>]", tl, re.IGNORECASE)
            )

            if row_count >= min_rows:
                table_text = "\n".join(html_lines)
                prefix = ""
                if table_cfg.get("include_preceding_heading", True) and last_heading:
                    prefix = last_heading + "\n\n"

                tables.append(
                    TableBlock(
                        text=prefix + table_text,
                        preceding_heading=last_heading,
                        start_line=table_start,
                        end_line=table_end,
                    )
                )
                table_line_ranges.append((table_start, table_end))
                i = table_end
                continue

        i += 1

    # Remove table lines from text
    if table_line_ranges:
        remaining_lines = []
        for idx, line in enumerate(lines):
            in_table = any(start <= idx < end for start, end in table_line_ranges)
            if not in_table:
                remaining_lines.append(line)
        remaining_text = "\n".join(remaining_lines)
    else:
        remaining_text = text

    return tables, remaining_text


# ---------------------------------------------------------------------------
# 1d. Adaptive Chunk Assembly
# ---------------------------------------------------------------------------


def assemble_chunks(
    sentences: list[Sentence],
    tables: list[TableBlock],
    source_file: str,
    config: dict,
) -> list[Chunk]:
    chunk_cfg = config["chunking"]["chunking"]
    tokenizer_cfg = config["chunking"]["tokenizer"]

    min_tokens = chunk_cfg.get("min_chunk_tokens", 200)
    max_tokens = chunk_cfg.get("max_chunk_tokens", 5000)
    overlap_n = chunk_cfg.get("overlap_sentences", 2)
    force_break = chunk_cfg.get("force_break_on_headings", True)
    break_levels = set(chunk_cfg.get("heading_levels_to_break_on", [1, 2, 3]))
    merge_short = chunk_cfg.get("merge_short_trailing_chunks", True)

    chunks: list[Chunk] = []
    chunk_index = 0

    # Add table chunks first
    for table in tables:
        tok_count = count_tokens(table.text, tokenizer_cfg)
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}__chunk_{chunk_index}",
                chunk_text=table.text,
                source_file=source_file,
                section_hierarchy=table.preceding_heading.lstrip("#").strip(),
                token_count=tok_count,
                contains_table=True,
                chunk_index=chunk_index,
                is_oversized=tok_count > max_tokens,
            )
        )
        chunk_index += 1

    # Assemble text chunks from sentences
    current_sentences: list[Sentence] = []
    current_tokens = 0
    current_hierarchy = ""

    def finalize_chunk(sents: list[Sentence]) -> Chunk | None:
        nonlocal chunk_index
        if not sents:
            return None
        text = "\n".join(s.text for s in sents)
        tok_count = count_tokens(text, tokenizer_cfg)
        hierarchy = sents[0].heading_hierarchy or current_hierarchy
        c = Chunk(
            chunk_id=f"{source_file}__chunk_{chunk_index}",
            chunk_text=text,
            source_file=source_file,
            section_hierarchy=hierarchy,
            token_count=tok_count,
            contains_table=False,
            chunk_index=chunk_index,
        )
        chunk_index += 1
        return c

    for sent in sentences:
        sent_tokens = count_tokens(sent.text, tokenizer_cfg)

        if sent.is_heading and force_break and sent.heading_level in break_levels:
            if current_sentences:
                tok = count_tokens(
                    "\n".join(s.text for s in current_sentences), tokenizer_cfg
                )
                if tok >= min_tokens:
                    chunk = finalize_chunk(current_sentences)
                    if chunk:
                        chunks.append(chunk)
                    current_sentences = []
                    current_tokens = 0

            current_hierarchy = sent.heading_hierarchy
            current_sentences.append(sent)
            current_tokens = sent_tokens
            continue

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunk = finalize_chunk(current_sentences)
            if chunk:
                chunks.append(chunk)
            overlap = current_sentences[-overlap_n:] if overlap_n > 0 else []
            current_sentences = list(overlap)
            current_tokens = sum(
                count_tokens(s.text, tokenizer_cfg) for s in current_sentences
            )

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Handle remaining sentences
    if current_sentences:
        tok = count_tokens(
            "\n".join(s.text for s in current_sentences), tokenizer_cfg
        )
        if tok < min_tokens and merge_short and chunks:
            last_text_chunks = [c for c in chunks if not c.contains_table]
            if last_text_chunks:
                target = last_text_chunks[-1]
                merged_text = (
                    target.chunk_text + "\n" + "\n".join(s.text for s in current_sentences)
                )
                target.chunk_text = merged_text
                target.token_count = count_tokens(merged_text, tokenizer_cfg)
            else:
                chunk = finalize_chunk(current_sentences)
                if chunk:
                    chunks.append(chunk)
        else:
            chunk = finalize_chunk(current_sentences)
            if chunk:
                chunks.append(chunk)

    # Re-sort by chunk_index
    chunks.sort(key=lambda c: c.chunk_index)
    return chunks


# ---------------------------------------------------------------------------
# Main Stage 1 Entry Point
# ---------------------------------------------------------------------------


async def run_stage1(
    config: dict,
    llm_client: UnifiedLLMClient,
) -> tuple[list[dict], dict[str, str]]:
    input_dir = Path(config["paths"]["input_docs_dir"])
    intermediate_dir = Path(config["paths"]["intermediate_dir"])
    chunking_cfg = config["chunking"]

    chunks_path = intermediate_dir / chunking_cfg["output"]["chunks_file"]
    summaries_path = intermediate_dir / chunking_cfg["output"]["summaries_file"]

    # Check for existing outputs
    if chunks_path.exists() and summaries_path.exists():
        logger.info("Stage 1 outputs already exist, loading from disk")
        from src.utils import read_jsonl, read_json

        return read_jsonl(chunks_path), read_json(summaries_path)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        logger.error("No .md files found in %s", input_dir)
        return [], {}

    logger.info("Stage 1: Processing %d markdown files", len(md_files))

    # 1a. Generate summaries
    summaries = await generate_summaries(md_files, llm_client, config)

    # Process each file
    all_chunks: list[dict] = []
    for md_file in md_files:
        logger.info("Chunking: %s", md_file.name)
        text = md_file.read_text(encoding="utf-8")

        # 1b-pre. Pre-process (remove page headers, code blocks, boilerplate)
        text = preprocess_text(text, config)

        # 1c. Extract tables
        tables, remaining_text = extract_tables(text, config)

        # 1b. Detect sentences
        sentences = detect_sentences(remaining_text, md_file.name, config)

        # 1d. Assemble chunks
        chunks = assemble_chunks(sentences, tables, md_file.name, config)

        logger.info(
            "  %s: %d sentences, %d tables, %d chunks",
            md_file.name,
            len(sentences),
            len(tables),
            len(chunks),
        )

        all_chunks.extend(c.to_dict() for c in chunks)

    # Write outputs
    write_jsonl(chunks_path, all_chunks)
    write_json(summaries_path, summaries)
    logger.info(
        "Stage 1 complete: %d total chunks, %d summaries",
        len(all_chunks),
        len(summaries),
    )

    return all_chunks, summaries
