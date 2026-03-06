import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.llm_client import TokenUsageTracker, UnifiedLLMClient
from src.utils import ensure_dirs, load_all_configs, setup_logging


async def main():
    config = load_all_configs("configs")
    logger = setup_logging(config)
    ensure_dirs(config)

    logger.info("=" * 60)
    logger.info("Saudi Q&A Dataset Pipeline — Starting")
    logger.info("=" * 60)

    usage_tracker = TokenUsageTracker()
    llm_client = UnifiedLLMClient(config, usage_tracker)
    stages = config.get("stages", {})
    pipeline_start = time.time()

    # ── Stage 1: Chunking ──
    chunks: list[dict] = []
    summaries: dict[str, str] = {}

    if stages.get("run_chunking", True):
        logger.info("─" * 40)
        logger.info("STAGE 1: Semantic Text Chunking")
        logger.info("─" * 40)
        t0 = time.time()

        from src.stage1_chunking import run_stage1

        chunks, summaries = await run_stage1(config, llm_client)
        logger.info("Stage 1 took %.1fs", time.time() - t0)
        logger.info("Gemini usage (price till now): %s", llm_client.usage_tracker.format_summary())
    else:
        logger.info("Stage 1 skipped (run_chunking=false)")
        from src.utils import read_json, read_jsonl

        intermediate = config["paths"]["intermediate_dir"]
        chunks_file = Path(intermediate) / config["chunking"]["output"]["chunks_file"]
        summaries_file = Path(intermediate) / config["chunking"]["output"]["summaries_file"]
        if chunks_file.exists():
            chunks = read_jsonl(chunks_file)
        if summaries_file.exists():
            summaries = read_json(summaries_file)

    if not chunks:
        logger.error("No chunks available. Cannot proceed.")
        sys.exit(1)

    # ── Stage 2: Q&A Generation ──
    raw_qa: list[dict] = []

    if stages.get("run_generation", True):
        logger.info("─" * 40)
        logger.info("STAGE 2: Q&A Generation (Saudi Dialect)")
        logger.info("─" * 40)
        t0 = time.time()

        from src.stage2_generation import run_stage2

        raw_qa = await run_stage2(config, llm_client, chunks, summaries)
        logger.info("Stage 2 took %.1fs", time.time() - t0)
        logger.info("Gemini usage (price till now): %s", llm_client.usage_tracker.format_summary())
    else:
        logger.info("Stage 2 skipped (run_generation=false)")
        from src.utils import read_jsonl

        intermediate = config["paths"]["intermediate_dir"]
        qa_file = Path(intermediate) / config["generation"]["output"]["raw_qa_file"]
        if qa_file.exists():
            raw_qa = read_jsonl(qa_file)

    if not raw_qa:
        logger.error("No QA pairs available. Cannot proceed.")
        sys.exit(1)

    # ── Stage 3: Cleaning + Coverage ──
    cleaned_qa: list[dict] = []
    embeddings = None

    if stages.get("run_cleaning", True):
        logger.info("─" * 40)
        logger.info("STAGE 3: Data Cleaning + Coverage Analysis")
        logger.info("─" * 40)
        t0 = time.time()

        from src.stage3_cleaning import run_stage3

        cleaned_qa, embeddings = await run_stage3(config, llm_client, raw_qa)
        logger.info("Stage 3 took %.1fs", time.time() - t0)
        logger.info("Gemini usage (price till now): %s", llm_client.usage_tracker.format_summary())
    else:
        logger.info("Stage 3 skipped (run_cleaning=false)")
        from src.embeddings import load_embeddings
        from src.utils import read_jsonl

        intermediate = config["paths"]["intermediate_dir"]
        cleaned_file = Path(intermediate) / config["cleaning"]["output"]["cleaned_qa_file"]
        emb_file = Path(intermediate) / config["cleaning"]["output"].get(
            "embeddings_file", "embeddings.npy"
        )
        if cleaned_file.exists():
            cleaned_qa = read_jsonl(cleaned_file)
        embeddings = load_embeddings(emb_file)

    if not cleaned_qa:
        logger.error("No cleaned QA pairs available. Cannot proceed.")
        sys.exit(1)

    if embeddings is None:
        logger.error("No embeddings available. Cannot proceed to Stage 4.")
        sys.exit(1)

    # ── Stage 4: Evaluation Dataset ──
    if stages.get("run_evaluation", True):
        logger.info("─" * 40)
        logger.info("STAGE 4: Evaluation Dataset Creation")
        logger.info("─" * 40)
        t0 = time.time()

        from src.stage4_evaluation import run_stage4

        await run_stage4(config, llm_client, cleaned_qa, embeddings)
        logger.info("Stage 4 took %.1fs", time.time() - t0)
        logger.info("Gemini usage (price till now): %s", llm_client.usage_tracker.format_summary())
    else:
        logger.info("Stage 4 skipped (run_evaluation=false)")

    total = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", total)
    logger.info("Gemini total usage: %s", llm_client.usage_tracker.format_summary())
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
