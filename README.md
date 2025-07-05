# OpenAI Voice Agent

Простой голосовой агент, использующий OpenAI плагины для STT, LLM и TTS.

## Компоненты

- **STT**: FAL WizperSTT (fal-ai/wizper)
- **LLM**: OpenAI GPT-4.1-nano  
- **TTS**: OpenAI TTS с голосом "alloy"
- **VAD**: Silero Voice Activity Detection

## Функции агента

- Приветствие пользователя при запуске
- Получение текущего времени и даты
- Выполнение математических вычислений
- Ведение дружественного диалога

## Запуск

### Консольный режим (для тестирования)
```bash
python openai_agent.py console
```

### Режим разработки
```bash  
python openai_agent.py dev
```

### Продакшн режим
```bash
python openai_agent.py start
```

## Настройки

Все настройки находятся в файле `.env`:
- `OPENAI_API_KEY` - API ключ OpenAI
- `OPENAI_BASE_URL` - Base URL для OpenAI API

Для режимов dev/start также необходимы LiveKit настройки.