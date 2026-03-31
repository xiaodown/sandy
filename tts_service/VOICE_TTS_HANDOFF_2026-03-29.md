# Voice/TTS Handoff - 2026-03-29

This is the end-of-day handoff for the voice spike.

## Where We Ended

We have a working isolated voice stack:

- Discord voice join/leave works
- voice receive works
- STT works
- outbound playback works
- overlapping send/receive works
- echo path works: speech -> STT -> TTS -> spoken playback

We also isolated Qwen TTS into its own service:

- bot/runtime env: `/home/xiaodown/code/sandy/.venv`
- TTS service env: `/home/xiaodown/code/sandy/tts_service/.venv`
- TTS service code: `/home/xiaodown/code/sandy/tts_service/tts_service/app.py`
- voice spike code: `/home/xiaodown/code/sandy/sandy/voicemvp/app.py`

## Important Current State

The TTS service is currently configured for **voice clone mode** by default, not prompt-only VoiceDesign.

Current reference clip:

- audio: `/home/xiaodown/code/sandy/sandy/voicemvp/captures/candidates/05-1042-goodnight-robst-sleep-well-dream-of-snacks-and-r.wav`
- transcript: `goodnight robst. sleep well, dream of snacks and revenge.`

Current mode details in `tts_service/app.py`:

- `mode = "voice_clone"`
- clone model = `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- voice design model still exists as fallback path
- `clone_x_vector_only_mode = true`
- `do_sample = false`
- `subtalker_dosample = false`
- `max_new_tokens = 512`
- `max_audio_seconds = 20`

## What Worked

### Big win

Voice clone mode materially improved **speaker identity consistency**.

Observed behavior:

- repeated `hello there` lines sounded like the same person
- slight text variations (`hello there?`, `hello there!`) still sounded like the same person
- user liked the selected reference voice

This is much better than prompt-only VoiceDesign, which kept drifting between "cousins."

### Candidate reference auditioning

Useful candidate directories:

- batch 1: `/home/xiaodown/code/sandy/sandy/voicemvp/captures/candidates`
- batch 2: `/home/xiaodown/code/sandy/sandy/voicemvp/captures/candidates_batch2`

User picked this as the best reference:

- `05-1042-goodnight-robst-sleep-well-dream-of-snacks-and-r.wav`

## What Failed

### Core remaining issue

Qwen still occasionally falls into a pathological over-generation path and tries to emit a giant clip until it hits the safety cap.

The failure shape is very consistent:

- generated audio exceeds `20s`
- computed length comes out to about `40.88s`
- service returns `500`
- bot logs `HTTPStatusError`

This happened in multiple modes:

- prompt-only VoiceDesign
- clone mode with transcript/reference continuation (`x_vector_only_mode = false`)
- clone mode with speaker-embedding-only (`x_vector_only_mode = true`)

So the remaining failure is **not**:

- Discord
- queueing
- apostrophe parsing
- clone transcript conditioning specifically

It appears to be a model/runtime generation degeneracy issue for some text shapes.

### Example phrases that failed

These are known-bad or at least known-risk:

- `that's weird`
- `thats weird`
- `there hello`
- `this is a test of the emergency tts system`
- some other otherwise normal lines during candidate rendering

### Important conclusion

We isolated the variables enough to say:

- identity drift was mostly solved by clone mode
- the remaining blocker is generation reliability, not voice identity

## Queue / Bot Hygiene Changes Already Made

In `sandy/voicemvp/app.py`:

- `!say` now queues through the TTS worker instead of trying to play immediately
- queued TTS items are flushed on `!leave`
- queued TTS items are flushed before reconnect/move
- overlong TTS failures now reply back in-channel instead of disappearing into logs

## Candidate Rendering Helper

Script:

- `/home/xiaodown/code/sandy/tts_service/scripts/render_candidate_refs.py`

Purpose:

- pull curated Sandy lines from Recall
- render candidate WAVs into `voicemvp/captures/...`
- write `manifest.txt`, `manifest.json`, `failures.json`

## Likely Next Steps

These are the next sensible experiments, in order.

### 1. Add a tiny text normalization / speech rendering pass before TTS

This is the most pragmatic next experiment.

Goal:

- turn unsafe/awkward text into safer spoken forms before sending to Qwen

Examples:

- normalize weird punctuation
- smooth some contractions if needed
- maybe avoid certain word orders that seem to trigger degeneracy
- optionally split longer lines into smaller speakable chunks

This should be treated as a narrow guardrail layer, not a giant new subsystem.

### 2. Keep the clone path, stop expecting prompt-only VoiceDesign to be the final solution

We learned enough to stop spending time pretending VoiceDesign alone is the final mechanism for stable Sandy identity.

### 3. Consider testing an alternative Qwen runtime, but do not assume it fixes correctness

Possible candidate:

- `faster-qwen3-tts`

Reason to try it:

- maybe better performance/runtime behavior

Reason not to over-trust it:

- it may improve speed more than it improves reliability
- we do not yet know whether the 40.88s degeneration is model-side or implementation-side

### 4. If normalization + runtime variant still fail, evaluate another backend

At that point, moving on would be reasonable.

## Blunt Summary

Today was not wasted.

We now know:

- the voice stack itself is real and works
- isolated TTS service architecture was the right move
- clone mode is the first thing that actually made Sandy sound like a consistent person
- Qwen is still brittle on some inputs

That is real progress, even though it ended in "well shit."

## 2026-03-30 Update

We tested `faster-qwen3-tts` in a disposable sandbox at:

- `/home/xiaodown/code/sandy/tts_service2`

This used the same reference clip:

- `/home/xiaodown/code/sandy/sandy/voicemvp/captures/candidates/05-1042-goodnight-robst-sleep-well-dream-of-snacks-and-r.wav`

and the same basic clone setup:

- model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- clone mode
- `do_sample = false`
- `xvec_only = true`

### Important result

The exact phrases that were blowing up under the stock `qwen-tts` path rendered successfully under `faster-qwen3-tts`:

- `hello there`
- `there hello`
- `that's weird`
- `this is a test of the emergency tts system`

The user listened to those outputs and said:

- they sound like the reference WAV
- they all sound like the same person
- they are not garbled
- they seem fully functional

### Bigger batch result

We also rendered a batch of 10 real Sandy Recall lines through `faster-qwen3-tts` clone mode.

Output directory:

- `/home/xiaodown/code/sandy/sandy/voicemvp/captures/faster-moretests`

Manifest:

- `/home/xiaodown/code/sandy/sandy/voicemvp/captures/faster-moretests/manifest.txt`

Result:

- `10/10` rendered
- `0` failures

This included lines that definitely failed under the old path, such as:

- `sure! let's see it. i'm curious how the power draw looks.`
- `yup! that's cool. i'm happy to be here. you test whatever you need to. i'm ready for it. what's up for today?`

### Practical conclusion

At this point, the stock `qwen-tts` service should be treated as the failed path and `faster-qwen3-tts` should be treated as the candidate replacement.

Remaining work after this note:

1. preserve old `tts_service/` briefly as `tts_service_old/`
2. promote `tts_service2/` into the real `tts_service/` slot
3. wrap `faster-qwen3-tts` behind the same tiny HTTP API (`/health`, `/warmup`, `/synthesize`)
4. point `voicemvp` at the replacement service
5. rerun `!say` / echo-path tests end-to-end

Streaming was explicitly deferred until after the backend replacement is proven in the existing non-streaming voice spike.

## 2026-03-30 Update: canonical service swap complete

The backend replacement is now actually done in the canonical service path:

- `/home/xiaodown/code/sandy/tts_service`

What is true now:

- `tts_service/pyproject.toml` depends on `faster-qwen3-tts`
- `tts_service/tts_service/app.py` is the FastAPI wrapper around `FasterQwen3TTS`
- the service still exposes:
  - `GET /health`
  - `POST /warmup`
  - `POST /synthesize`
- the normal port is still `8777`
- `voicemvp` does not need an API change to talk to it

Notes:

- `tts_service2/` still exists as the sandbox where the faster path was proven, but it is no longer the canonical implementation
- the stock `qwen-tts` service path should now be treated as dead history unless there is a specific reason to revisit it
- the TTS service boundary still matters because the faster stack pulls in its own heavy Python/torch/huggingface dependency pile
