---
date: 2026-02-27
type: technology
topic: "Whisper & Azure Speech Services"
project: "Phase 5 - Multi-Modal Enterprise Assistant"
status: complete
---

# Technology Brief: Whisper & Azure Speech Services

## Overview

Audio processing in an enterprise multi-modal system typically means **speech-to-text (STT)** — converting spoken audio into text that an LLM can reason over. There are two main Azure services for this:

| Service | Type | Best For |
|---------|------|----------|
| **Azure OpenAI Whisper** | Batch transcription model | Pre-recorded files, simple transcription |
| **Azure Speech Service** | Full speech platform | Real-time STT, TTS, translation, diarization |

## Azure OpenAI Whisper

### What Is Whisper?

Whisper is an open-source speech recognition model from OpenAI, available as a managed API on Azure OpenAI. It provides high-quality transcription across 57 languages.

### Capabilities

| Feature | Support |
|---------|---------|
| **Languages** | 57 languages + auto-detection |
| **Audio formats** | mp3, mp4, mpeg, mpga, m4a, wav, webm |
| **Max file size** | 25 MB |
| **Word timestamps** | ✅ Yes (word-level) |
| **Translation** | ✅ Any language → English |
| **Speaker diarization** | ❌ Not built-in (use Azure Speech instead) |
| **Real-time streaming** | ❌ Batch only |

### Basic Usage

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<key>",
    api_version="2024-10-21"
)

# Basic transcription
with open("meeting_recording.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper",  # deployment name
        file=audio_file,
        language="en",     # optional: hint language
        response_format="verbose_json"  # text | json | verbose_json | srt | vtt
    )

print(transcript.text)  # full transcription text
```

### Response Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `text` | Plain text | Simple transcription |
| `json` | JSON with text | Programmatic access |
| `verbose_json` | JSON with word timestamps + segments | Meeting processing, alignment |
| `srt` | SubRip subtitle format | Subtitles |
| `vtt` | WebVTT subtitle format | Web subtitles |

### Verbose JSON Response Structure

```json
{
  "text": "Hello everyone, let's start the meeting...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello everyone, let's start the meeting.",
      "tokens": [50364, 2425, ...],
      "temperature": 0.0,
      "avg_logprob": -0.15,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ],
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "everyone", "start": 0.5, "end": 1.0},
    ...
  ]
}
```

### Handling the 25 MB Limit

For longer recordings, you need to split audio into chunks:

```python
from pydub import AudioSegment
import io

def transcribe_long_audio(file_path: str, chunk_minutes: int = 10) -> str:
    """Transcribe audio longer than 25MB by splitting into chunks."""
    audio = AudioSegment.from_file(file_path)
    chunk_ms = chunk_minutes * 60 * 1000
    
    full_transcript = []
    
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        
        # Export chunk to bytes
        buffer = io.BytesIO()
        chunk.export(buffer, format="mp3")
        buffer.seek(0)
        buffer.name = f"chunk_{i}.mp3"
        
        # Transcribe chunk
        result = client.audio.transcriptions.create(
            model="whisper",
            file=buffer,
            response_format="verbose_json"
        )
        
        full_transcript.append({
            "text": result.text,
            "segments": result.segments,
            "offset_ms": i
        })
    
    return full_transcript
```

### Translation (Any Language → English)

```python
# Translate foreign language audio to English text
with open("german_meeting.mp3", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="whisper",
        file=audio_file,
        response_format="verbose_json"
    )

print(translation.text)  # English translation of German audio
```

## Azure Speech Service

### Overview

Azure Speech Service is a broader platform offering real-time STT, TTS, translation, and advanced features like speaker diarization.

### Capabilities Comparison

| Feature | Whisper (Azure OpenAI) | Azure Speech Service |
|---------|----------------------|---------------------|
| **Real-time STT** | ❌ | ✅ |
| **Batch STT** | ✅ (per-file) | ✅ (bulk jobs) |
| **Fast transcription** | ❌ | ✅ (near real-time batch) |
| **Speaker diarization** | ❌ | ✅ |
| **Custom models** | ❌ | ✅ (Custom Speech) |
| **Pronunciation assessment** | ❌ | ✅ |
| **Text-to-speech** | ❌ | ✅ |
| **Translation** | ✅ (to English only) | ✅ (60+ languages) |
| **Keyword spotting** | ❌ | ✅ |
| **On-premises deployment** | ❌ | ✅ (containers) |
| **Cost** | Per minute of audio | Per second of audio |

### When to Use Which

| Scenario | Use |
|----------|-----|
| Simple transcription of pre-recorded files | **Whisper** — simpler API, good quality |
| Real-time transcription (live meeting) | **Azure Speech** — streaming support |
| Need speaker identification | **Azure Speech** — diarization |
| Need custom vocabulary (medical, legal) | **Azure Speech** — Custom Speech |
| Translate audio to English | **Whisper** — built-in translation |
| High-volume batch processing | **Azure Speech** — Batch transcription |
| On-premises requirement | **Azure Speech** — Docker containers |

### Azure Speech SDK Usage

```python
import azure.cognitiveservices.speech as speechsdk

# Configuration
speech_config = speechsdk.SpeechConfig(
    subscription="<key>",
    region="<region>"
)
speech_config.speech_recognition_language = "en-US"

# Real-time recognition from microphone
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
result = recognizer.recognize_once()
print(result.text)

# Recognition from file
audio_config = speechsdk.AudioConfig(filename="meeting.wav")
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config
)

# Continuous recognition (for longer files)
all_results = []

def handle_recognized(evt):
    all_results.append(evt.result.text)

recognizer.recognized.connect(handle_recognized)
recognizer.start_continuous_recognition()

# Wait for completion...
recognizer.stop_continuous_recognition()
full_text = " ".join(all_results)
```

### Speaker Diarization

Speaker diarization identifies **who spoke when** — critical for meeting transcription:

```python
speech_config = speechsdk.SpeechConfig(
    subscription="<key>",
    region="<region>"
)

audio_config = speechsdk.AudioConfig(filename="meeting.wav")

# Enable diarization
auto_detect_source_language_config = speechsdk.AutoDetectSourceLanguageConfig(
    languages=["en-US"]
)

conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
    speech_config=speech_config,
    audio_config=audio_config
)

transcript = []

def transcribed_cb(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript.append({
            "speaker": evt.result.speaker_id,
            "text": evt.result.text,
            "offset": evt.result.offset,
            "duration": evt.result.duration
        })

conversation_transcriber.transcribed.connect(transcribed_cb)
conversation_transcriber.start_transcribing_async().get()

# Output:
# Speaker 1: "Let's discuss the Q3 results."
# Speaker 2: "Revenue was up 15% quarter over quarter."
# Speaker 1: "What drove that growth?"
```

### Batch Transcription (High Volume)

For processing large volumes of audio files:

```python
import requests
import json

endpoint = "https://<region>.api.cognitive.microsoft.com"
subscription_key = "<key>"

# Create batch transcription job
transcription = {
    "contentUrls": [
        "https://storage.blob.core.windows.net/audio/meeting1.wav",
        "https://storage.blob.core.windows.net/audio/meeting2.wav",
    ],
    "properties": {
        "diarizationEnabled": True,
        "wordLevelTimestampsEnabled": True,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked"
    },
    "locale": "en-US",
    "displayName": "Daily meeting batch"
}

response = requests.post(
    f"{endpoint}/speechtotext/v3.2/transcriptions",
    headers={
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Content-Type": "application/json"
    },
    json=transcription
)

job_url = response.headers["Location"]
# Poll job_url for completion, then download results
```

### Fast Transcription (Preview)

A newer option for near-real-time batch transcription without the overhead of a batch job:

```
POST /speechtotext/transcriptions:transcribe?api-version=2024-11-15

Supports:
- Synchronous API (submit and get results immediately)
- Up to 2 hours of audio
- Diarization
- Word timestamps
- Much simpler than batch transcription for single files
```

## Meeting Processing Pipeline

For the Phase 5 project, a practical meeting transcription pipeline:

```
Audio File
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 1. Validate & Preprocess                          │
│    ├── Check format, duration, size               │
│    ├── Convert to supported format if needed       │
│    └── Split if > 25 MB (for Whisper)             │
├──────────────────────────────────────────────────┤
│ 2. Transcribe                                     │
│    ├── Short files (< 25 MB) → Whisper API        │
│    ├── Need diarization → Azure Speech Service    │
│    └── Long files (> 2 hrs) → Batch transcription │
├──────────────────────────────────────────────────┤
│ 3. Post-Process                                   │
│    ├── Merge chunk transcripts                    │
│    ├── Clean up punctuation and formatting        │
│    ├── Assign speaker labels (if diarization)     │
│    └── Generate timestamps index                  │
├──────────────────────────────────────────────────┤
│ 4. Enrich with LLM                               │
│    ├── Generate meeting summary                   │
│    ├── Extract action items                       │
│    ├── Identify key decisions                     │
│    └── Create searchable index                    │
└──────────────────────────────────────────────────┘
```

### Example: Meeting Summary with LLM

```python
def summarize_meeting(transcript: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper model for summarization
        messages=[
            {
                "role": "system",
                "content": """Analyze this meeting transcript and extract:
1. Summary (3-5 sentences)
2. Key decisions made
3. Action items with owners
4. Open questions
5. Topics discussed"""
            },
            {"role": "user", "content": transcript}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

## Pricing

### Whisper (Azure OpenAI)

| Tier | Price |
|------|-------|
| Standard | $0.006 / minute |
| Global Standard | $0.006 / minute |

A 60-minute meeting costs ~$0.36

### Azure Speech Service

| Feature | Price (Standard) |
|---------|-----------------|
| STT (real-time) | $1.00 / hour |
| STT (batch) | $0.80 / hour |
| STT (fast transcription) | $1.10 / hour |
| Custom Speech (real-time) | $1.40 / hour |
| Conversation transcription (diarization) | $2.10 / hour |

A 60-minute meeting with diarization costs ~$2.10

**Trade-off**: Whisper is ~6x cheaper, but Azure Speech gives you diarization. Choose based on whether you need speaker identification.

## Best Practices for My Project

- ✅ **Use Whisper** for simple transcription of pre-recorded files — simpler, cheaper
- ✅ **Use Azure Speech** when you need speaker diarization for meetings
- ✅ **Always use `verbose_json`** format to get timestamps — needed for alignment and navigation
- ✅ **Pre-validate audio** — check format, duration, and size before processing
- ✅ **Split long audio intelligently** — at silence boundaries, not arbitrary time cuts (use pydub's `silence.detect_silence`)
- ✅ **Cache transcriptions** — audio doesn't change; transcribe once, store forever
- ✅ **Use LLM post-processing** — transcription + GPT-4o-mini for summaries and action items
- ❌ **Don't send huge files to Whisper** — split at 25 MB boundary
- ❌ **Don't ignore audio quality** — garbage in = garbage out; check sample rate and noise levels
- ❌ **Don't rely on automatic language detection for mixed-language meetings** — specify primary language

## Resources

- [Azure OpenAI Whisper](https://learn.microsoft.com/en-us/azure/ai-services/openai/whisper-quickstart) — Quickstart
- [Azure Speech Service](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/) — Full documentation
- [Conversation Transcription (Diarization)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-use-conversation-transcription) — Speaker identification
- [Batch Transcription](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription) — High-volume processing
- [Fast Transcription](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create) — Synchronous batch API
