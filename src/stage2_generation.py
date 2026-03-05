import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz

from src.llm_client import UnifiedLLMClient
from src.utils import (
    append_jsonl,
    load_prompt,
    read_json,
    read_jsonl,
    render_prompt,
    repair_json,
    write_json,
    write_jsonl,
)

logger = logging.getLogger("pipeline.stage2")


def _format_existing_questions(questions: list[str], fmt: str) -> str:
    if fmt == "json_list":
        return json.dumps(questions, ensure_ascii=False, indent=2)
    if fmt == "numbered_list":
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    # bullet_list
    return "\n".join(f"- {q}" for q in questions)


def _is_near_duplicate(
    new_q: str, existing_questions: list[str], threshold: float
) -> bool:
    for eq in existing_questions:
        if fuzz.ratio(new_q, eq) > threshold * 100:
            return True
    return False


async def _generate_for_chunk(
    chunk: dict,
    summaries: dict[str, str],
    llm_client: UnifiedLLMClient,
    config: dict,
    system_prompt: str,
    user_template: str,
    followup_template: str,
) -> list[dict]:
    gen_cfg = config["generation"]
    iter_cfg = gen_cfg["iteration"]

    domain_context = config.get("domain_context", "")
    summary = summaries.get(chunk["source_file"], "")

    user_prompt = render_prompt(
        user_template,
        document_summary=summary or "No summary available.",
        domain_context=domain_context,
        section_hierarchy=chunk.get("section_hierarchy", ""),
        source_file=chunk.get("source_file", ""),
        chunk_text=chunk["chunk_text"],
    )

    # Iteration 1 — initial generation
    response = await llm_client.call(
        role="generation",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=gen_cfg.get("temperature", 0.8),
        max_tokens=gen_cfg.get("max_tokens", 4096),
        json_mode=gen_cfg.get("response_format") == "json",
    )

    accumulated: list[dict] = []
    existing_questions: list[str] = []

    if response.success:
        parsed = _parse_qa_response(response.text, iter_cfg)
        for qa in parsed:
            qa["chunk_id"] = chunk["chunk_id"]
            qa["source_file"] = chunk["source_file"]
            qa["section_hierarchy"] = chunk.get("section_hierarchy", "")
            qa["context"] = chunk["chunk_text"]
            qa["generation_model"] = response.model
            accumulated.append(qa)
            existing_questions.append(qa.get("question", ""))
    else:
        logger.warning("Initial generation failed for %s: %s", chunk["chunk_id"], response.error)

    # Iterative follow-up
    max_iter = iter_cfg.get("max_iterations", 10)
    stagnation_count = 0
    stagnation_limit = iter_cfg.get("stagnation_limit", 2)
    dedup_cfg = iter_cfg.get("within_loop_dedup", {})
    dedup_enabled = dedup_cfg.get("enabled", True)
    dedup_threshold = dedup_cfg.get("similarity_threshold", 0.80)

    for iteration in range(1, max_iter):
        formatted_qs = _format_existing_questions(
            existing_questions, iter_cfg.get("feedback_format", "numbered_list")
        )

        followup_prompt = render_prompt(
            followup_template,
            chunk_text=chunk["chunk_text"],
            existing_questions=formatted_qs,
        )

        response = await llm_client.call(
            role="generation",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": followup_prompt},
            ],
            temperature=gen_cfg.get("temperature", 0.8),
            max_tokens=gen_cfg.get("max_tokens", 4096),
            json_mode=gen_cfg.get("response_format") == "json",
        )

        if not response.success:
            strategy = iter_cfg.get("parse_failure_strategy", "retry_once")
            if strategy == "stop_loop":
                break
            continue

        parsed = _parse_qa_response(response.text, iter_cfg)

        if not parsed:
            break

        new_count = 0
        for qa in parsed:
            q = qa.get("question", "")
            if dedup_enabled and _is_near_duplicate(q, existing_questions, dedup_threshold):
                continue
            qa["chunk_id"] = chunk["chunk_id"]
            qa["source_file"] = chunk["source_file"]
            qa["section_hierarchy"] = chunk.get("section_hierarchy", "")
            qa["context"] = chunk["chunk_text"]
            qa["generation_model"] = response.model
            accumulated.append(qa)
            existing_questions.append(q)
            new_count += 1

        if new_count == 0:
            stagnation_count += 1
            if stagnation_count >= stagnation_limit:
                logger.debug(
                    "Stagnation limit reached for %s at iteration %d",
                    chunk["chunk_id"],
                    iteration + 1,
                )
                break
        else:
            stagnation_count = 0

    # Add generation metadata
    for qa in accumulated:
        qa["generation_metadata"] = {
            "total_iterations": iteration + 1 if 'iteration' in dir() else 1,
            "total_questions": len(accumulated),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return accumulated


def _parse_qa_response(raw_text: str, iter_cfg: dict) -> list[dict]:
    text = raw_text.strip()
    if not text or text == "[]":
        return []

    repaired = repair_json(text)
    if repaired is None:
        logger.warning("Could not parse QA response: %.200s", text)
        return []

    try:
        data = json.loads(repaired)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict) and "question" in item]


async def _process_chunk_batch(
    chunks: list[dict],
    summaries: dict[str, str],
    llm_client: UnifiedLLMClient,
    config: dict,
    system_prompt: str,
    user_template: str,
    followup_template: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    results = []

    async def worker(chunk: dict):
        async with semaphore:
            return await _generate_for_chunk(
                chunk, summaries, llm_client, config,
                system_prompt, user_template, followup_template,
            )

    tasks = [worker(c) for c in chunks]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(completed):
        if isinstance(result, Exception):
            logger.error("Chunk %s failed: %s", chunks[i]["chunk_id"], result)
            if config["generation"]["batching"].get("skip_failed_chunks", True):
                continue
            raise result
        results.extend(result)

    return results


async def run_stage2(
    config: dict,
    llm_client: UnifiedLLMClient,
    chunks: list[dict],
    summaries: dict[str, str],
) -> list[dict]:
    intermediate_dir = Path(config["paths"]["intermediate_dir"])
    gen_cfg = config["generation"]
    batch_cfg = gen_cfg["batching"]

    output_path = intermediate_dir / gen_cfg["output"]["raw_qa_file"]
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_file = checkpoint_dir / "stage2_checkpoint.json"

    # Load checkpoint
    completed_chunk_ids: set[str] = set()
    existing_qa: list[dict] = []
    if checkpoint_file.exists():
        cp = read_json(checkpoint_file)
        completed_chunk_ids = set(cp.get("completed_chunks", []))
        logger.info("Resuming Stage 2: %d chunks already done", len(completed_chunk_ids))
    if output_path.exists() and completed_chunk_ids:
        existing_qa = read_jsonl(output_path)

    remaining_chunks = [c for c in chunks if c["chunk_id"] not in completed_chunk_ids]

    if not remaining_chunks:
        logger.info("Stage 2: All chunks already processed")
        return existing_qa if existing_qa else read_jsonl(output_path)

    logger.info(
        "Stage 2: Generating Q&A for %d chunks (%d already done)",
        len(remaining_chunks),
        len(completed_chunk_ids),
    )

    # Build system prompt
    prompts_dir = config["paths"]["prompts_dir"]
    system_prompt = load_prompt(prompts_dir, gen_cfg["prompts"]["system_prompt_file"])
    if gen_cfg.get("dialect_spec", {}).get("inject_into_system_prompt", True):
        dialect_file = gen_cfg["dialect_spec"].get("dialect_instructions_file")
        if dialect_file:
            dialect_text = load_prompt(prompts_dir, dialect_file)
            system_prompt = system_prompt.replace("{dialect_spec}", dialect_text)

    user_template = load_prompt(prompts_dir, gen_cfg["prompts"]["user_prompt_file"])
    followup_template = load_prompt(prompts_dir, gen_cfg["prompts"]["followup_prompt_file"])

    concurrency = batch_cfg.get("concurrency", 3)
    checkpoint_every = batch_cfg.get("checkpoint_every_n_chunks", 25)
    semaphore = asyncio.Semaphore(concurrency)

    all_qa = list(existing_qa)

    # Process in checkpoint-sized batches
    for batch_start in range(0, len(remaining_chunks), checkpoint_every):
        batch = remaining_chunks[batch_start : batch_start + checkpoint_every]
        logger.info(
            "Processing chunk batch %d-%d / %d",
            batch_start + 1,
            min(batch_start + len(batch), len(remaining_chunks)),
            len(remaining_chunks),
        )

        batch_results = await _process_chunk_batch(
            batch, summaries, llm_client, config,
            system_prompt, user_template, followup_template, semaphore,
        )

        all_qa.extend(batch_results)
        completed_chunk_ids.update(c["chunk_id"] for c in batch)

        # Write checkpoint
        write_jsonl(output_path, all_qa)
        write_json(
            checkpoint_file,
            {"completed_chunks": list(completed_chunk_ids)},
        )
        logger.info(
            "Checkpoint saved: %d chunks done, %d QA pairs total",
            len(completed_chunk_ids),
            len(all_qa),
        )

    logger.info("Stage 2 complete: %d total QA pairs from %d chunks", len(all_qa), len(chunks))
    return all_qa
