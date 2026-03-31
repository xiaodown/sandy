# Voice Handover — 2026-03-31

## Current State

Voice is integrated into Sandy Prime well enough to hold a short live Discord
conversation. This is no longer just spike code in `voicemvp`.

Working end-to-end:

- admin `!join` / `!leave`
- Discord VC connect / disconnect
- receive sink + preroll capture
- faster-whisper STT
- same-speaker fragment stitching
- separate rolling voice-session history
- voice-specific brain prompt path
- external TTS service over HTTP
- Discord playback
- vector-memory writes for voice turns
- paused text replies while VC is active

Verified live in Discord on 2026-03-31:

- multiple voice turns were captured, transcribed, answered, and played back
- TTS synth calls succeeded repeatedly after prompt/reply-length tuning
- Sandy left VC cleanly and runtime status returned to idle cleanly

## Important Files

- `/home/xiaodown/code/sandy/sandy/voice/manager.py`
- `/home/xiaodown/code/sandy/sandy/voice/capture.py`
- `/home/xiaodown/code/sandy/sandy/voice/stt.py`
- `/home/xiaodown/code/sandy/sandy/voice/tts.py`
- `/home/xiaodown/code/sandy/sandy/voice/history.py`
- `/home/xiaodown/code/sandy/sandy/pipeline.py`
- `/home/xiaodown/code/sandy/sandy/bot.py`
- `/home/xiaodown/code/sandy/sandy/prompt.py`
- `/home/xiaodown/code/sandy/sandy/llm.py`
- `/home/xiaodown/code/sandy/sandy/registry.py`
- `/home/xiaodown/code/sandy/sandy/maintenance.py`

## What Was Fixed Tonight

1. Voice admin and control plumbing

- added `voice_admin` support in registry
- added maintenance commands:
  - `set-voice-admin`
  - `lookup-registry`

2. Integrated voice runtime

- added `sandy.voice` package and wired it into the main bot / pipeline
- added one-global-session behavior
- added participant tracking from `on_voice_state_update`
- text replies now pause while VC is active

3. Silent voice-response failure

- initial integrated runs captured STT correctly but never reached TTS
- root cause was bad critical-path ordering plus poor task visibility
- fixed by:
  - supervising voice response tasks
  - adding task/brain/playback logs
  - moving voice memory writes off the foreground response path

4. Turn assembly / release issues

- thread-boundary issues around speaking-start/stop were fixed with
  `call_soon_threadsafe`
- added a force-release safety valve so stale speaker state cannot wedge a turn
- kept preroll capture

5. Overlong reply / TTS 500 failure

- one live run produced a long monologue
- TTS generated audio longer than the service limit and returned HTTP 500
- fixed by:
  - stronger voice prompt constraints
  - deterministic reply sanitizing before TTS
  - retrying with a shorter fallback if synthesis still fails

6. Voice tuning pass

- loosened the prompt after the first overcorrection made replies too thin
- increased stitch/release timing slightly
- latest live run sounded materially better

## Current Behavior Notes

- voice short-term context is working; this is not using text `Last10`
- voice still gets vector/RAG context in the brain call
- voice does not write to Recall in v1
- there is still no voice bouncer and no voice tool-calling path
- TTS remains an external service on `127.0.0.1:8777`
- Sandy observability API remains on `8765`

## Known Issues / Remaining Work

1. Spoken-language prompt quality still needs work

- Sandy still writes too much like chat sometimes
- abbreviations like `idk` should become `i don't know`
- voice should prefer spoken forms over text shorthand

2. Voice memory quality needs a pass

- vector/RAG is present in the voice brain path
- memory-grounded voice answers still need evaluation and likely formatting work
- no Recall for voice in v1 remains the right call for now

3. Observability is still thin

- logs are decent now
- API status exposes active voice session state
- richer trace/dashboard support for voice turns is still unfinished

4. Runtime/startup cleanup remains unfinished

- faster-whisper CUDA preload hack still exists
- TTS still expects the separate service to be started manually

## Useful Commands

Run Sandy:

```bash
cd ~/code/sandy
source .venv/bin/activate
python -m sandy
```

Run tests:

```bash
cd ~/code/sandy
source .venv/bin/activate
pytest
```

Run health check:

```bash
cd ~/code/sandy
source .venv/bin/activate
python -m sandy.health --json
```

Run TTS service:

```bash
cd ~/code/sandy/tts_service
source .venv/bin/activate
python -m tts_service
```

Grant voice admin:

```bash
cd ~/code/sandy
source .venv/bin/activate
python -m sandy.maintenance set-voice-admin --user-id USER --server-id GUILD --enable
```

Look up ids fuzzily:

```bash
python -m sandy.maintenance lookup-registry --user xiao
python -m sandy.maintenance lookup-registry --server snack
```

## IDs Used Tonight

- Xiaodown user id: `218896334130905090`
- Snack Bandits Co.: `599445001763815455`
- Xiaodown Bot Testing: `1359032772332621875`

## Useful Log Locations

During live debugging, these were useful:

- `/tmp/sandy-live.log`
- `/tmp/tts-live.log`

The clean live run near the end showed:

- repeated `Voice brain completed: has_response=True`
- repeated `POST /synthesize HTTP/1.1 200 OK`
- clean playback finish events

## Recommended First Task Next Session

Do a dedicated spoken-language prompt pass:

- explicitly discourage text-chat abbreviations in voice mode
- tell the model to write what should be spoken aloud
- keep the current anti-monologue guardrails
- test with lines likely to trigger `idk`, `ngl`, numbers, acronyms, and other
  ugly spoken-text cases

That is the highest-value next step. The system works now; the main problem is
quality and polish, not missing architecture.
