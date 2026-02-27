---
date: 2026-02-27
type: concept
topic: "Audio Processing Pipelines"
project: "Phase 5 - Multi-Modal Enterprise Assistant"
status: complete
---

# Learning: Audio Processing Pipelines

## In My Own Words

Audio processing in the context of enterprise AI means converting spoken language into structured text and extracting meaningful insights from it. The core pipeline is: **raw audio → preprocessing → transcription (Speech-to-Text) → post-processing → downstream analysis**. The transcription step is where models like Whisper shine — they convert audio waveforms into text with remarkable accuracy across languages and accents.

But transcription is just the beginning. Enterprise use cases like meeting summarization require additional steps: speaker identification (who said what), timestamp alignment, topic segmentation, action item extraction, and summarization. The full pipeline is a chain of specialized processing stages.

## Why This Matters

Audio is one of the most information-rich modalities in enterprise settings:

- **Meetings**: Hours of discussion compressed into actionable summaries
- **Customer calls**: Support interactions analyzed for quality and sentiment
- **Lectures/training**: Educational content made searchable and summarizable
- **Voice notes**: Quick dictations converted to structured records
- **Earnings calls**: Financial audio analyzed for sentiment and key metrics

Without audio processing, this wealth of information remains locked in an inaccessible format.

## Core Principles

### 1. The Speech-to-Text Pipeline

```
Raw Audio ──▶ [Preprocessing] ──▶ [Feature Extraction] ──▶ [ASR Model] ──▶ [Post-processing] ──▶ Text
```

Each stage:

1. **Preprocessing**: Noise reduction, volume normalization, format conversion, chunking
2. **Feature Extraction**: Convert audio waveform to spectrograms (mel-frequency cepstral coefficients or mel spectrograms)
3. **ASR Model (Automatic Speech Recognition)**: Neural network that maps audio features to text
4. **Post-processing**: Punctuation restoration, capitalization, formatting, timestamp alignment

### 2. How Whisper Works

Whisper (by OpenAI) is a transformer-based ASR model trained on 680,000 hours of multilingual audio.

**Architecture**:
```
Audio ──▶ [Log-Mel Spectrogram] ──▶ [Encoder (Transformer)] ──▶ Audio Features
                                                                      │
                                                                      ▼
                                          [Decoder (Transformer)] ◄── Previous Tokens
                                                  │
                                                  ▼
                                            Predicted Text Tokens
```

**Key design decisions**:
- **Encoder-decoder transformer**: Similar architecture to language models, but encoder processes audio features
- **Multi-task training**: Trained on transcription, translation, language identification, and voice activity detection simultaneously
- **Robust to noise**: Trained on diverse, real-world audio with background noise, accents, and varying quality
- **Multilingual**: Supports 99+ languages, can auto-detect language

**Whisper Model Sizes**:

| Model | Parameters | English-only? | Relative Speed | Quality |
|-------|-----------|---------------|----------------|---------|
| tiny | 39M | Available | ~32x | Basic |
| base | 74M | Available | ~16x | Good |
| small | 244M | Available | ~6x | Better |
| medium | 769M | Available | ~2x | Great |
| large-v3 | 1.55B | No | 1x | Best |

### 3. Audio Chunking for Large Files

Whisper has a **25 MB file size limit** per API call. For longer recordings:

**Strategy 1: Fixed-Length Chunking**
```
[==========|==========|==========|====]
  30 sec      30 sec     30 sec    15 sec
```
- Simple but may cut words mid-sentence
- Add 1-2 second overlap between chunks to avoid losing words at boundaries

**Strategy 2: Silence-Based Chunking (VAD — Voice Activity Detection)**
```
[===speech===|..silence..|===speech===|..silence..|===speech===]
     chunk 1                  chunk 2                  chunk 3
```
- Split at natural pauses in speech
- Better word boundary preservation
- Requires VAD preprocessing (e.g., `webrtcvad`, `silero-vad`, or `pydub` silence detection)

**Strategy 3: Sliding Window**
```
[=====window 1=====]
         [=====window 2=====]
                  [=====window 3=====]
```
- Overlapping windows with deduplication
- Best quality but most expensive
- Useful for real-time streaming scenarios

### 4. Speaker Diarization

Answering "who said what?" — critical for meeting transcription.

**How it works**:
```
Audio ──▶ [Voice Activity Detection] ──▶ [Speaker Embedding Extraction] ──▶ [Clustering] ──▶ Speaker Labels
```

1. **VAD**: Identify segments of speech vs silence
2. **Embedding extraction**: Convert each speech segment into a speaker embedding vector (neural network trained on speaker identity)
3. **Clustering**: Group segments by speaker similarity (usually spectral clustering or agglomerative clustering)
4. **Label assignment**: Map clusters to speaker labels (Speaker 1, Speaker 2, etc.)

**Tools for diarization**:
- **Azure Speech Service**: Built-in diarization (up to 36 speakers)
- **pyannote.audio**: Open-source Python library, state-of-the-art performance
- **AssemblyAI**: Third-party API with built-in diarization
- **AWS Transcribe**: Diarization in Amazon's service

### 5. The Meeting Summarization Pipeline

The full pipeline for enterprise meeting processing:

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Audio Preprocessing                                  │
│  Raw Audio ──▶ Normalize ──▶ Chunk ──▶ Audio Segments        │
├──────────────────────────────────────────────────────────────┤
│ Phase 2: Transcription                                        │
│  Audio Segments ──▶ Whisper/Azure STT ──▶ Raw Transcript     │
├──────────────────────────────────────────────────────────────┤
│ Phase 3: Enhancement                                          │
│  Raw Transcript ──▶ Speaker Diarization ──▶ Timestamped +    │
│                     Punctuation Restoration   Speaker-tagged  │
│                     Entity Recognition         Transcript     │
├──────────────────────────────────────────────────────────────┤
│ Phase 4: Analysis                                             │
│  Enhanced Transcript ──▶ LLM ──▶ Summary                     │
│                              ──▶ Action Items                 │
│                              ──▶ Key Decisions                │
│                              ──▶ Topic Segments               │
│                              ──▶ Sentiment Analysis           │
└──────────────────────────────────────────────────────────────┘
```

## Approaches & Trade-offs

### STT Service Comparison

| Feature | OpenAI Whisper (API) | Azure Speech Service | Azure OpenAI Whisper | Self-Hosted Whisper |
|---------|---------------------|---------------------|---------------------|-------------------|
| **Accuracy** | Excellent | Excellent | Excellent | Model-dependent |
| **Languages** | 99+ | 100+ | 99+ | 99+ |
| **File size limit** | 25 MB | 2 GB (batch) | 25 MB | No limit |
| **Real-time** | No | Yes | No | Possible |
| **Diarization** | No | Yes (built-in) | No | With pyannote |
| **Custom models** | No | Yes | No | Yes |
| **Timestamps** | Word-level | Word-level | Word-level | Word-level |
| **Cost** | $0.006/min | $1/hr (standard) | $0.006/min | Compute only |
| **Privacy** | Cloud | Cloud / On-prem (container) | Cloud | Full control |
| **Batch processing** | Limited | Yes (dedicated API) | Limited | Yes |

### Transcription Approaches

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Real-time STT** | Live captioning, voice assistants | Immediate results | Lower accuracy, no context |
| **Fast transcription** | Pre-recorded files, quick turnaround | Good quality, fast | No diarization in some services |
| **Batch transcription** | Large volumes, overnight processing | Best quality, cheapest | High latency |
| **Self-hosted** | Privacy requirements, custom domains | Full control | Infrastructure burden |

## Audio Formats and Quality

### Supported Formats (Whisper)

| Format | Extension | Notes |
|--------|-----------|-------|
| MP3 | .mp3 | Common, compressed, good for most uses |
| MP4 | .mp4 | Video container, audio extracted |
| MPEG | .mpeg | Legacy format |
| MPGA | .mpga | MPEG audio |
| M4A | .m4a | Apple audio, compressed |
| WAV | .wav | Uncompressed, best quality, large files |
| WebM | .webm | Web audio, common in browser recordings |

### Audio Quality Impact on Accuracy

| Factor | Impact on Accuracy | Mitigation |
|--------|-------------------|------------|
| **Background noise** | Moderate-High decrease | Noise reduction preprocessing |
| **Multiple speakers talking over each other** | High decrease | Use diarization, warn users |
| **Accented speech** | Low-Moderate decrease | Whisper handles well; Azure custom models help |
| **Low bitrate** | Moderate decrease | Recommend minimum 128kbps |
| **Phone-quality audio** | Moderate decrease | Whisper trained on phone audio |
| **Domain jargon** | Moderate decrease | Custom vocabulary (Azure), post-processing |

## Best Practices

- ✅ **Use silence-based chunking** for files exceeding 25 MB — preserves word boundaries
- ✅ **Add 1-2 second overlap** between chunks to avoid losing boundary words
- ✅ **Normalize audio** before processing (sample rate: 16kHz, mono channel works best for Whisper)
- ✅ **Use word-level timestamps** when available — essential for alignment with other modalities
- ✅ **Implement speaker diarization** for any multi-speaker scenario (meetings, interviews)
- ✅ **Post-process transcripts** with an LLM to fix formatting, add punctuation, and correct domain terms
- ✅ **Cache transcriptions** — audio processing is expensive; never re-transcribe the same audio
- ❌ **Don't assume perfect transcription** — always account for errors in downstream processing
- ❌ **Don't send huge files without chunking** — you'll hit API limits and timeouts
- ❌ **Don't skip audio preprocessing** — garbage in, garbage out applies strongly to audio

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| **Truncated transcription** | File exceeds size limit | Chunk before sending; check file size |
| **Missing words at chunk boundaries** | Chunks cut mid-word | Overlap chunks by 1-2 seconds |
| **Wrong language detected** | Short audio or mixed languages | Specify language explicitly when known |
| **Speaker confusion** | Diarization fails with similar voices | Provide expected speaker count; use higher quality audio |
| **Hallucinated text** | Whisper generates text for silence or music | Pre-filter with VAD; post-validate |
| **Timestamp drift** | Accumulated errors across long audio | Re-anchor timestamps at chunk boundaries |

## Application to My Project

### How I'll Use This

The Multi-Modal Enterprise Assistant needs audio processing for:

1. **Meeting summarization**: Full pipeline from audio → transcript → diarized summary
2. **Audio Q&A**: Upload audio → transcribe → answer questions about content
3. **Voice input**: Accept voice queries that get transcribed and processed
4. **Cross-modal integration**: Align audio transcripts with slide/document content

### Pipeline Design

```python
# Pseudocode for the audio processing pipeline
async def process_audio(audio_file: UploadFile) -> ProcessedAudio:
    # 1. Validate and preprocess
    audio = preprocess(audio_file)  # normalize, convert to supported format
    
    # 2. Chunk if needed
    chunks = chunk_audio(audio, max_size_mb=24, overlap_seconds=2)
    
    # 3. Transcribe each chunk (parallel)
    transcripts = await asyncio.gather(*[
        transcribe_chunk(chunk) for chunk in chunks
    ])
    
    # 4. Merge and deduplicate
    full_transcript = merge_transcripts(transcripts)
    
    # 5. Post-process with LLM
    enhanced = await enhance_transcript(full_transcript)
    
    return ProcessedAudio(
        transcript=enhanced.text,
        speakers=enhanced.speakers,
        timestamps=enhanced.timestamps,
        summary=enhanced.summary
    )
```

### Decisions to Make

- [ ] Whisper API vs Azure Speech Service vs self-hosted — depends on privacy requirements
- [ ] Real-time vs batch processing mode
- [ ] Speaker diarization: Azure built-in vs pyannote
- [ ] Chunking strategy: fixed-length with overlap vs silence-based
- [ ] How to handle multi-language meetings

## Resources for Deeper Learning

- [Whisper Paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356) — Original Whisper research paper
- [Azure Speech Service Overview](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/overview) — Full capabilities of Azure Speech
- [Azure OpenAI Whisper Quickstart](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/whisper-quickstart) — Getting started with Whisper on Azure
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Open-source speaker diarization
- [Azure Batch Transcription](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription) — Processing large volumes of audio

## Questions Remaining

- [ ] What's the optimal chunk size for Whisper accuracy?
- [ ] How to handle real-time audio + image streams (e.g., live meeting with screen share)?
- [ ] Best approach for domain-specific vocabulary correction?
- [ ] How to handle audio files with music or non-speech audio segments?
