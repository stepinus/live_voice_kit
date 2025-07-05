from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from livekit.agents import tts, utils, APIConnectOptions
from livekit.agents.tts import TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)


# Set MPS fallback before imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

try:
    from RealtimeTTS import TextToAudioStream, CoquiEngine
except ImportError as e:
    logger.error("Failed to import RealtimeTTS dependencies: %s", e)
    raise


class LocalTTS(tts.TTS):
    """
    Local TTS implementation using RealtimeTTS with Coqui XTTS model.
    Provides streaming TTS capabilities with voice cloning.
    """

    def __init__(
        self,
        *,
        language: str = "ru",
        model_path: str = "models/Lasinya",
        voice_reference_path: str = "voice/reference_audio.wav",
        device: str = "cpu",
        sample_rate: int = 24000,
    ) -> None:
        """
        Initialize LocalTTS with RealtimeTTS engine.

        Args:
            language: Language for synthesis (default: "ru")
            model_path: Path to XTTS model directory
            voice_reference_path: Path to reference audio for voice cloning
            device: Device to use ("cpu" or "cuda")
            sample_rate: Audio sample rate in Hz
        """
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._language = language
        self._model_path = f"/Users/y.stepin/dev/agents/{model_path}"
        self._voice_reference_path = f"/Users/y.stepin/dev/agents/{voice_reference_path}"
        self._device = device
        self._engine: CoquiEngine | None = None
        self._stream: TextToAudioStream | None = None

        logger.info("Initializing LocalTTS with language=%s, device=%s", language, device)

    def _ensure_engine(self) -> None:
        """Lazy initialization of TTS engine."""
        if self._engine is not None:
            return

        logger.info("Loading XTTS engine with model: %s", self._model_path)
        
        # Check if voice reference file exists
        if not os.path.exists(self._voice_reference_path):
            raise FileNotFoundError(f"Voice reference file not found: {self._voice_reference_path}")

        try:
            # Initialize CoquiEngine with voice cloning
            self._engine = CoquiEngine(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",  # Standard XTTS v2 model
                specific_model="Lasinya",  # Our custom model name
                local_models_path="/Users/y.stepin/dev/agents/models",  # Path to models folder
                language=self._language,
                device=self._device,
                voice=self._voice_reference_path
            )
            
            logger.info("XTTS engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize XTTS engine: %s", e)
            raise

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        """Synthesize text using chunked approach."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> SynthesizeStream:
        """Create streaming synthesis interface.""" 
        return SynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        """Clean up resources."""
        self._engine = None


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming synthesis for complete text input."""

    def __init__(self, *, tts: LocalTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: LocalTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run synthesis using callback approach without threads."""
        self._tts._ensure_engine()
        
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        
        engine = self._tts._engine
        if engine is None:
            raise RuntimeError("TTS engine not initialized")
        
        # Create completion event
        synthesis_complete = asyncio.Event()
        
        def on_audio_chunk(chunk: bytes) -> None:
            """Stream audio chunks immediately."""
            if chunk:
                output_emitter.push(chunk)
        
        def on_audio_stream_stop() -> None:
            """Signal synthesis completion."""
            logger.info("Audio synthesis completed")
            synthesis_complete.set()
        
        def before_sentence_synthesized(sentence: str) -> None:
            """Log sentence before synthesis."""
            logger.info(f"Synthesizing sentence: {sentence}")
        
        try:
            # Create streaming interface with callbacks
            from RealtimeTTS import TextToAudioStream
            stream = TextToAudioStream(
                engine,
                muted=True,
                on_audio_stream_stop=on_audio_stream_stop,
                language=self._tts._language
            )
            
            # Feed text
            stream.feed(self._input_text)
            
            # Start async synthesis with callbacks
            stream.play_async(
                on_audio_chunk=on_audio_chunk,
                before_sentence_synthesized=before_sentence_synthesized,
                log_synthesized_text=True,
                language=self._tts._language,
                muted=True,
                fast_sentence_fragment=True
            )
            
            # Wait for completion
            try:
                await asyncio.wait_for(synthesis_complete.wait(), timeout=30.0)
                logger.info("Synthesis completed successfully")
            except asyncio.TimeoutError:
                logger.warning("Synthesis timed out")
            
            # Flush any remaining audio
            output_emitter.flush()
            
        except Exception as e:
            logger.error("Error during synthesis: %s", e)
            raise


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis interface."""

    def __init__(self, *, tts: LocalTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: LocalTTS = tts
        self.tts_stream: Any = None

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run streaming synthesis with callback approach without threads."""
        self._tts._ensure_engine()
        
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        # Start single segment for entire response  
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        engine = self._tts._engine
        if engine is None:
            raise RuntimeError("TTS engine not initialized")

        # Create completion event
        synthesis_complete = asyncio.Event()
        text_buffer = []
        
        try:
            # Define callbacks
            def on_audio_chunk(chunk: bytes) -> None:
                """Stream audio chunks immediately."""
                if chunk:
                    # logger.debug(f"Received audio chunk: {len(chunk)} bytes")
                    output_emitter.push(chunk)
            
            def on_audio_stream_stop() -> None:
                """Signal synthesis completion."""
                logger.info("Audio stream stopped")
                synthesis_complete.set()

            # Create streaming interface
            from RealtimeTTS import TextToAudioStream
            stream = TextToAudioStream(
                engine,
                muted=True,
                tokenizer="stanza",
                language=self._tts._language,
                on_audio_stream_stop=on_audio_stream_stop
            )

            # Collect all text chunks first
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                
                if data.strip():
                    logger.debug(f"Collecting text: {data[:50]}...")
                    text_buffer.append(data)
            
            # Combine all text
            full_text = "".join(text_buffer)
            if not full_text.strip():
                logger.warning("No text to synthesize")
                output_emitter.end_input()
                return
            
            logger.info(f"Starting synthesis of {len(full_text)} characters")
            
            # Feed text and start async synthesis
            stream.feed(full_text)
            stream.play_async(
                on_audio_chunk=on_audio_chunk,
                log_synthesized_text=True,
                tokenizer="stanza",
                fast_sentence_fragment=True,
                muted=True,
                language=self._tts._language
            )
            
            # Wait for completion
            try:
                await asyncio.wait_for(synthesis_complete.wait(), timeout=30.0)
                logger.info("TTS synthesis completed")
            except asyncio.TimeoutError:
                logger.warning("TTS synthesis timed out")
            
            # End the segment
            output_emitter.end_input()
                
        except Exception as e:
            logger.error("Error during streaming synthesis: %s", e)
            raise