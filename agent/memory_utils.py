import asyncio
from datetime import datetime
from loguru import logger
from langchain_core.messages import HumanMessage

from agent.nodes.action import memory_store
from agent.llm import get_llm

_DAY_SUMMARY_PROMPT = """\
You are writing a diary entry about this day from your own first-person memory.
Recall the flow of the day: what happened in the morning, then afternoon, then evening.
Capture how your feelings changed as events unfolded — what made you happy, 
what frustrated you, what surprised you, what lingered in your mind.

Rules:
- Write in first person, as someone remembering their own lived day
- Follow the chronological arc: morning → afternoon → evening
- Include specific details: what you saw, who you talked to, what was said
- Show emotional shifts: how one event changed how you felt about the next
- Do NOT list events — weave them into a flowing narrative
- Do NOT include titles, headers, or markdown formatting
- Start directly with the first sentence of the entry
- 5-8 sentences. Write in Japanese.

{observations}

Write just the diary entry."""


async def generate_day_summary(date: str) -> None:
    """Generate and save a day summary for the given date using LLM."""
    try:
        observations = await asyncio.to_thread(
            memory_store.get_observations_for_date, date, 50
        )
        if not observations:
            logger.info(f"No observations for {date}, skipping day summary")
            return

        lines = []
        for obs in observations:
            emotion = f" [{obs['emotion']}]" if obs["emotion"] != "neutral" else ""
            lines.append(
                f"  {obs['time']} ({obs['kind']}){emotion}: {obs['content'][:150]}"
            )
        transcript = "\n".join(lines)

        logger.info(
            f"Generating day summary for {date} ({len(observations)} observations)"
        )

        llm = get_llm()
        prompt = _DAY_SUMMARY_PROMPT.format(observations=transcript)
        messages = [HumanMessage(content=prompt)]

        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30.0)
        summary = response.content

        if summary:
            await memory_store.save_async(
                summary,
                direction="記憶",
                kind="day_summary",
                emotion="neutral",
                override_date=date,
            )
            logger.info(f"Day summary generated for {date}: {summary[:80]}")
        else:
            logger.warning(f"Day summary for {date}: LLM returned empty response")
    except asyncio.TimeoutError:
        logger.warning(f"Day summary for {date} timed out (30s)")
    except Exception as e:
        logger.warning(f"Failed to generate day summary for {date}: {e}")


async def backfill_day_summaries() -> None:
    """Generate day summaries for past dates that don't have one yet."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        all_dates = await asyncio.to_thread(memory_store.get_dates_with_observations, 7)
        existing = await asyncio.to_thread(memory_store.get_dates_with_summaries)

        missing = [d for d in all_dates if d != today and d not in existing][:5]
        if missing:
            logger.info(f"Backfill: generating day summaries for {missing}")
            for date in missing:
                await generate_day_summary(date)
        else:
            logger.info("Backfill: no missing day summaries")
    except Exception as e:
        logger.warning(f"Day summary backfill failed: {e}")
