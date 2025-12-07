# Voice Notes Toolkit

Tkinter toolkit for working with the `voice_notes` SQLite database:
- **Voice Note Analyst Pro 2.0** (`Audio_transcribe.py`): transcribe audio files with Whisper, summarize/analyze via OpenAI or Ollama, store results, browse history, and query the DB via the Explore tab.

## Requirements
- Python 3.10+ recommended.
- Install deps: `pip install -r requirements.txt`  
  (includes `ttkbootstrap`, `openai`, `openai-whisper`, `torch`, `requests`).
- Whisper/torch typically need FFmpeg available on PATH for audio decoding.

## Configuration (`vna_config.json`)
This file is git-ignored. Example:
```json
{
  "provider": "OpenAI",
  "openai_key": "sk-...your-key...",
  "openai_model": "gpt-5",
  "ollama_url": "http://localhost:11434",
  "ollama_model": "gemma3:27b",
  "whisper_model": "large"
}
```
Values are editable from the UI in `Audio_transcribe.py` and persisted back to this file.

## App 1: Voice Note Analyst Pro 2.0 (`Audio_transcribe.py`)
What it does
- Batch transcribes selected audio files with Whisper (model selectable).
- Analyzes transcripts via OpenAI Chat or Ollama to extract summary, calls_to_action, tone, people_mentioned, tags, and subject (JSON response enforced).
- Persists everything into `voice_notes` table in `voice_notes_data.db`.
- Library tab to search, preview, export to Markdown, and play audio files.
- Explore tab to query the SQLite DB with plain English (same flow as the standalone SQL assistant below).

Run
```bash
python Audio_transcribe.py
```
Flow
1) Set provider (OpenAI/Ollama), keys/URLs, and Whisper model.  
2) Add audio files, then **START PROCESSING**; progress shown per file.  
3) Results saved to SQLite; use the Library tab to search, view details, export, or play audio.

Notes
- MPS fallback is enabled for macOS; CUDA is used when available, otherwise CPU.
- Uses table `voice_notes` (schema below) and creates it if missing.

## Database Schema
`voice_notes_data.db.sql` documents the `voice_notes` table:
- id (INTEGER PK AUTOINCREMENT)
- filename, file_path, date_recorded, date_processed
- transcription, summary, calls_to_action, tone, people_mentioned, tags, subject
- ai_provider, ai_model

## Repo Hygiene
- `.gitignore` keeps `vna_config.json`, `voice_notes_data.db`, and `audio/` out of git, along with common Python/IDE artifacts.
- Do not commit API keys or real audio/transcript data.  
