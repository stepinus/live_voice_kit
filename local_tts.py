from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from livekit.agents import tts, utils, APIConnectOptions
from livekit.agents.tts import TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)


class SSEParser:
    """
    Парсер для Server-Sent Events (SSE) потоков от OpenAI/OpenRouter API.
    Извлекает чистый текст из JSON-ответов в формате streaming.
    """
    
    def __init__(self):
        self.buffer = ""
    
    def parse_chunk(self, chunk: str) -> list[str]:
        """
        Парсит чанк SSE данных и возвращает извлеченный текст.
        
        Args:
            chunk: Сырой чанк данных от SSE потока
            
        Returns:
            Список текстовых фрагментов, извлеченных из JSON
        """
        if not chunk:
            return []
            
        self.buffer += chunk
        extracted_texts = []
        
        while True:
            # Ищем следующую полную SSE строку
            line_end = self.buffer.find('\n')
            if line_end == -1:
                break
            
            line = self.buffer[:line_end].strip()
            self.buffer = self.buffer[line_end + 1:]
            
            if line.startswith('data: '):
                data = line[6:]  # Убираем префикс 'data: '
                
                if data == '[DONE]':
                    logger.info("SSE stream completed")
                    break
                
                try:
                    # Парсим JSON
                    data_obj = json.loads(data)
                    
                    # Извлекаем контент из delta
                    if "choices" in data_obj and len(data_obj["choices"]) > 0:
                        delta = data_obj["choices"][0].get("delta", {})
                        content = delta.get("content")
                        
                        if content:
                            extracted_texts.append(content)
                            logger.debug(f"Extracted text: {content[:50]}...")
                            
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Error processing SSE line: {e}")
                    continue
        
        return extracted_texts



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
        device: str = "cuda",
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
                fast_sentence_fragment_allsentences_multiple=True,
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
        sse_parser = SSEParser()
        
        try:
            # Define callbacks
            def on_audio_chunk(chunk: bytes) -> None:
                """Stream audio chunks immediately."""
                if chunk:
                    # logger.info(f"Received audio chunk: {len(chunk)} bytes")
                    output_emitter.push(chunk)
                else:
                    logger.debug("Received empty audio chunk")
            
            def on_audio_stream_stop() -> None:
                """Signal synthesis completion."""
                logger.info("Audio stream stopped")
                synthesis_complete.set()

            # Create streaming interface
            from RealtimeTTS import TextToAudioStream
            stream = TextToAudioStream(
                engine,
                muted=True,
                language=self._tts._language,
                on_audio_stream_stop=on_audio_stream_stop
            )

            # Создаем отдельные задачи для отправки текста и получения аудио
            text_sent = asyncio.Event()
            
            async def send_text_task():
                """Отправляем текст в TTS по мере поступления"""
                try:
                    logger.info("Starting text sending task...")
                    
                    async for data in self._input_ch:
                        if isinstance(data, self._FlushSentinel):
                            continue
                        
                        if data:
                            logger.info(f"Received raw chunk: {data[:100]}...")
                            
                            # Сначала пробуем парсить как SSE данные
                            extracted_texts = sse_parser.parse_chunk(data)
                            logger.info(f"SSE parser extracted {len(extracted_texts)} text fragments")
                            
                            # Если SSE парсер ничего не извлек, обрабатываем как обычный текст
                            if not extracted_texts:
                                logger.info("No SSE data found, treating as plain text")
                                extracted_texts = [data]
                            
                            # Обрабатываем каждый извлеченный текстовый фрагмент
                            for text_fragment in extracted_texts:
                                if text_fragment:
                                    logger.info(f"Processing text fragment: '{text_fragment}'")
                                    stream.feed(text_fragment)
                        
                        # Запускаем TTS если еще не запущен
                        if not text_sent.is_set():
                            logger.info("Final text sent, starting playback")
                            text_sent.set()
                    
                    logger.info("Text sending completed")
                    
                except Exception as e:
                    logger.error("Error in text sending: %s", e)
            
            async def play_audio_task():
                """Воспроизводим аудио асинхронно"""
                try:
                    # Ждем первый текст
                    await text_sent.wait()
                    logger.info("Starting TTS playback...")
                    
                    # Запускаем воспроизведение в цикле событий
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: stream.play(
                            on_audio_chunk=on_audio_chunk,
                            log_synthesized_text=True,
                            fast_sentence_fragment=True,
                            minimum_sentence_length=10,
                            muted=True,
                            language=self._tts._language
                        )
                    )
                    logger.info("TTS playback completed")
                    
                except Exception as e:
                    logger.error("Error in TTS playback: %s", e)
                finally:
                    synthesis_complete.set()
            
            # Запускаем обе задачи параллельно
            send_task = asyncio.create_task(send_text_task())
            play_task = asyncio.create_task(play_audio_task())
            
            # Ждем завершения отправки текста
            await send_task
            
            # Если текст не был отправлен, принудительно запускаем воспроизведение
            if not text_sent.is_set():
                logger.warning("No text was processed, forcing TTS start")
                text_sent.set()
            
            # Wait for completion
            try:
                await asyncio.wait_for(play_task, timeout=60.0)
                logger.info("TTS synthesis completed")
            except asyncio.TimeoutError:
                logger.warning("TTS synthesis timed out")
            
            # End the segment
            output_emitter.end_input()
                
        except Exception as e:
            logger.error("Error during streaming synthesis: %s", e)
            raise