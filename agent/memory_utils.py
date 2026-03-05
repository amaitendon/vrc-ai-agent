import asyncio
from datetime import datetime
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.action import memory_store
from agent.llm import get_llm
from prompts.prompts import BASE_SYSTEM_PROMPT

_DAY_SUMMARY_PROMPT = """
あなたはこの日の日記を、自分の記憶として一人称で書いています。
印象的だった出来事と感情の変化を散文で書いてください。

【条件】
- 200文字以内

ルール：
- 一人称で、自分が思い出すように書く
- 箇条書き・タイトル・見出しは使わない
- 最初の一文から直接本文を始める
- 感情の変化を含める
- 具体的なディテールを1つ以上入れる

{observations}

日記本文のみを出力してください。文字数カウントや説明は不要です。
"""


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
        messages = [
            SystemMessage(content=BASE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

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
