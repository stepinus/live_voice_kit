import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import fal, openai, silero
from local_tts import LocalTTS

logger = logging.getLogger("openai-agent")

load_dotenv()


class OpenAIAgent(Agent):
    def __init__(self) -> None:
        # Load system prompt from file
        system_prompt_path = os.path.join(os.path.dirname(__file__), "characters", "system_lana.txt")
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_instructions = f.read().strip()
        except FileNotFoundError:
            # Fallback if file not found
            system_instructions = ("You are Lana, a virtual assistant. You must always communicate "
                                 "in Russian with warmth and personality. Be helpful and charming.")
        
        super().__init__(
            instructions=system_instructions,
        )

    async def on_enter(self) -> None:
        # Generate initial greeting when agent starts
        self.session.generate_reply(
            instructions="Поприветствуй пользователя тепло и мило. Спроси, как дела и чем можешь помочь."
        )

def prewarm(proc: JobProcess) -> None:
    """Preload models to reduce latency"""
    proc.userdata["vad"] = silero.VAD.load()
    # Create LocalTTS instance (lazy initialization)
    tts_engine = LocalTTS(
        language="ru",
        model_path="nvidia/llama-3.3-nemotron-super-49b-v1",
        voice_reference_path="voice/reference_audio.wav",
        device="cpu",
    )
    proc.userdata["tts"] = tts_engine


async def entrypoint(ctx: JobContext) -> None:
    """Main entry point for the OpenAI agent"""

    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create agent session with OpenAI components
    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(
            model="qwen/qwen3-8b",
            api_key="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",
            base_url="https://openrouter.ai/api/v1",
            max_completion_tokens=1000,
        ),
        stt=fal.WizperSTT(
            api_key="efcf420b-f475-4ea1-b3d8-0e67e69551db:3cbe0ff985bd8aa96161d3b265efe88e",
            language="ru",
        ),
        tts=ctx.proc.userdata["tts"],
    )

    # Set up usage metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info(f"Session usage summary: {summary}")

    # Register shutdown callback for logging final usage
    ctx.add_shutdown_callback(log_usage)

    # Start the agent session
    await session.start(
        agent=OpenAIAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Connect to the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
