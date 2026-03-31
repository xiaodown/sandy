# Voice Integration Plan

## Summary

Integrate voice into Sandy Prime as a first-class but still non-streaming
pipeline, while keeping Discord voice transport, STT, and TTS behind adapter
boundaries and keeping the faster TTS backend as a separate local service.

The first integrated version should:

- let authorized users/admins make Sandy join and leave a voice channel
- capture a speaker utterance, transcribe it, and preserve speaker identity
- assemble split STT fragments back into one speaker turn before the brain sees
  them
- maintain a separate rolling voice-session context for the active voice channel
- keep Sandy in at most one active voice chat globally
- store voice transcripts conservatively and with explicit modality metadata
- pause text replies while Sandy is in VC so voice latency wins cleanly
- keep observing text channels while in VC and continue low-priority background
  memory persistence for text turns
- run the existing brain path with voice-specific prompt instructions instead of
  a second speech-render model pass
- skip the text bouncer and tool calls entirely for voice v1
- coalesce newly completed human turns while Sandy is already generating or
  speaking, then answer the resulting conversation block once she is free
- speak the final reply back into VC

This is the right stopping point for the spike and the right starting point for
integration. Do not do streaming, barge-in, a second post-brain render pass, or
a fully separate voice-brain persona path in v1.

## Implementation Status

As of 2026-03-31, the first integrated voice slice is no longer theoretical.
The following pieces are implemented in Sandy Prime and have been live-tested in
Discord:

- `sandy.voice` package exists with capture, STT, TTS, session/history, and
  manager/orchestration layers
- admin-only `!join` / `!leave` works through the main bot
- registry has per-(user, server) `voice_admin` support plus maintenance CLI
  helpers to set and look up admins
- one global active VC session is enforced
- voice session participant tracking is wired through `on_voice_state_update`
- text replies are paused while VC is active
- voice uses separate rolling short-term history instead of text `Last10`
- voice still pulls vector/RAG context into the brain path
- voice transcripts and Sandy spoken replies are stored in vector memory, not
  Recall
- same-speaker STT fragments are stitched before reply with preroll capture kept
- coalescing exists so completed human turns append while Sandy is busy instead
  of spawning overlapping replies
- TTS stays external over localhost HTTP
- shutdown tears down voice state before the wider pipeline exits
- voice state is visible in the observability API status output

Live-test status:

- end-to-end VC conversation now works: capture -> STT -> brain -> TTS ->
  playback
- earlier silent-failure bugs in response-task scheduling and memory-write
  ordering were fixed
- overly long spoken replies that caused TTS `500` failures were mitigated with
  voice-specific prompt constraints plus a deterministic reply sanitizer /
  fallback shortening pass before TTS

Known remaining work before calling v1 "done":

- make a focused prompt pass for spoken-language quality
- reduce chat-style abbreviations / acronyms in voice output (`idk` ->
  `i don't know`, etc.)
- improve memory-grounded answers for voice without dragging Recall into v1
- add richer voice traces / dashboard surfacing beyond current status output
- keep tuning stitch/release timing from real call logs

## Key Changes

### 1. Add a real voice subsystem to Sandy Prime

Create a real `sandy.voice` package instead of trying to promote
`voicemvp/app.py` wholesale.

Shape:

- `transport` layer: Discord voice join/leave, receive sink, playback,
  SSRC/member mapping
- `stt` adapter: wraps the existing Faster-Whisper path
- `turn assembly` layer: merges split same-speaker STT fragments into one
  conversational turn
- `tts` adapter/client: wraps the existing HTTP TTS service contract
- `session` layer: owns active voice sessions, rolling voice history, queueing,
  and per-session state
- `orchestrator` layer: turns transcript -> Sandy brain (with voice-specific
  prompt framing) -> playback

This keeps the spike’s useful code but stops the MVP blob from becoming
production architecture.

### 2. Keep Discord text bot and voice bot in one process, but separate concerns

Do not run a second Discord client for voice in production. Extend the existing
Sandy Discord runtime so the main bot can also handle voice commands and own
voice session state.

Conceptually this stays one Sandy pipeline with diverging text and voice paths,
not two unrelated bots bolted into one repo. Shared LLM, memory, registry, and
trace infrastructure should stay shared. The divergence is in orchestration
steps, not in the entire product identity.

Implementation intent:

- `bot.py` remains Discord lifecycle/event glue
- message/text handling stays in the main pipeline shape
- voice event registration and session management plug in beside the text
  pipeline, not inside the text turn waterfall
- voice should be supervised like other background work, not managed as naked
  forever-tasks

The main process should own:

- command handling for join/leave
- the one active voice session allowed globally in v1
- voice STT/TTS workers
- voice turn assembly / stitch state
- shutdown of capture/playback/queues

The TTS backend stays external over localhost HTTP.

v1 concurrency policy:

- when Sandy is in VC, text replies are paused globally
- text messages may still be observed and queued for low-priority persistence
  work
- voice turns get the fast path simply by being the only foreground response
  mode while VC is active

### 3. Introduce a voice-session context model

Use a separate rolling voice history per active voice session. Do not reuse the
text-channel `Last10` directly.

Behavior:

- each active VC session gets its own rolling transcript window
- entries include speaker display name, user id, utterance text, timestamps, and
  whether the line was spoken by a human or by Sandy
- the brain should consume assembled speaker turns, not raw STT fragments
- this voice history is used as short-term context for the next voice reply
- long-term memory/RAG can still inject relevant Recall/Chroma context into the
  brain prompt
- voice history is not the same object as text `Last10`, even if both later end
  up drawing from the same long-term stores
- voice-aware prompt context should include the current VC participant list,
  because that is meaningful situational context humans also have

Default:

- one active voice session globally
- if Sandy is already in VC, a new `!join` should be rejected with a simple
  message rather than moving her automatically
- moving Sandy between VCs in v1 requires `!leave` followed by `!join`
- voice-session history cleared on disconnect unless session persistence is added
  later
- if Sandy is left alone in VC for a configurable idle window, she should leave
  automatically and end the session

### 4. Persist voice turns into long-term memory, but treat Recall cautiously

Store transcripts and Sandy spoken replies in long-term memory with explicit
voice metadata, but keep voice out of Recall in v1.

Why:

- voice transcripts and replies should still become part of Sandy’s long-term
  awareness
- this avoids building a separate dead-end voice-only memory system
- it allows normal retrieval to stay useful across modalities
- voice transcripts should definitely enter vector/RAG memory
- voice transcripts should not enter Recall yet because STT is not a literal
  source of truth and Recall is still treated as Sandy's more exact text-memory
  layer

Needed metadata:

- modality: `voice`
- source: `stt`
- guild id / voice channel id
- speaker user id and display name
- utterance timestamp
- boolean for `is_transcript` vs `is_spoken_reply`

Important implementation rule:

- do not store raw audio in the normal memory path
- only store transcript text and structured metadata
- if audio is ever kept, it should stay debug-only or explicitly opted in

### 5. Add voice-specific authorization and command routing

For v1, make join/leave admin-only.

Needed behavior:

- only admins or a configured allowlist can issue voice-control commands
- command surface can still be text commands at first, because that is already
  proven and avoids premature slash-command scope creep
- voice commands should log authorization failures clearly

Likely shape:

- `!join [voice channel name]`
- `!leave`

Implementation note:

- keep the policy pluggable so it can be relaxed later without rewriting
  transport code
- do not bury auth decisions inside the voice sink or session classes; enforce
  it at command-entry level

### 6. Reuse the main brain, but give voice its own prompt instructions

For v1, use the existing Sandy brain path, not a totally separate voice brain.
But do not feed raw transcript directly to the normal text prompt and call it
done. Voice needs its own modality-specific instructions.

Flow:

1. STT transcript arrives
2. assemble same-speaker STT fragments into one conversational turn when they
   are clearly part of the same thought
3. build a voice-aware user turn for the brain
4. if Sandy is idle, generate one short spoken reply against the current voice
   session context
5. if Sandy is already generating or speaking, keep appending completed human
   turns to the session history and coalesce them into one later conversation
   block instead of spawning overlapping replies
6. send the final reply text to TTS
7. append Sandy’s spoken text to the voice-session history and long-term memory

Voice-aware brain input should preserve:

- speaker identity
- recent voice-session context
- relevant Recall/RAG context
- current VC participant list

Voice prompt / reply constraints for v1:

- keep the main personality/person memory blocks shared with text mode
- add voice-specific instructions that push for short, speakable replies
- target mostly short spoken responses, roughly 8-30 words in the common case
- allow occasional longer replies when the conversation genuinely calls for it,
  but keep a hard leash on rambling
- prefer one or two compact sentences, not text-message paragraphs
- prefer spoken-out language over chat abbreviations and internet shorthand when
  the line will be said aloud
- keep a voice-specific `num_predict` cap available if needed
- do not add a second LLM pass just to rewrite the output shorter

Future path:

- later add a true post-brain prosody/style pass only if the integrated loop
  proves it is worth the latency

Turn assembly / stitcher v1:

- STT should run on captured audio segments as soon as possible; do not hold raw
  audio waiting for a perfect end-of-thought boundary
- after STT returns, fragments from the same speaker should be merged in Python
  when they are clearly continuation pieces rather than separate turns
- stitching should be deterministic and cheap: string concatenation plus light
  whitespace/punctuation cleanup, not an LLM
- use the spike's preroll capture behavior; it materially improves transcript
  quality at utterance start
- release to the brain should be driven primarily by session state, not a dumb
  fixed wait:
  - hold if the same speaker is actively speaking
  - hold if another same-speaker fragment is currently in STT
  - hold if another same-speaker fragment has already returned and is pending
    merge
- keep a short max hold timeout only as a safety valve against stuck state, not
  as the main turn-boundary mechanism
- goal: avoid responding to transport/VAD fragmentation like `I went to` and
  `the grocery store` as if they were two independent thoughts

### 7. Preserve the TTS service boundary exactly

Keep the current local HTTP contract:

- `GET /health`
- `POST /warmup`
- `POST /synthesize`

And keep using the faster backend in the canonical `tts_service/`.

Do not import the faster TTS Python API into Sandy Prime directly.

Reason:

- heavy Torch/HF dependency pile
- service boundary already paid for itself
- swapping TTS later stays cheap

For integration, the Sandy-side TTS client should move out of `voicemvp` into a
reusable adapter module under main Sandy.

### 8. Move only the proven spike pieces into Prime

Promote the good pieces, not the whole spike.

Keep/adapt:

- utterance capture and preroll logic
- SSRC/member mapping, while treating SSRC as session-local transport metadata
  rather than stable identity
- WAV/transcript job model
- TTS HTTP client + WAV->Discord PCM conversion
- queue-based TTS playback
- the current faster-whisper CUDA preload hack as an explicitly temporary
  runtime workaround until startup/env handling is cleaned up

Do not blindly carry over:

- HTTP file server for captures
- spike-specific `!say` / `!playtest` debugging behavior
- isolated second-client assumptions
- echo mode as production behavior

### 9. Add voice observability from day one

Voice will become impossible to reason about if it is not traced.

Minimum telemetry:

- voice session started/stopped
- auto-leave due to idle/no participants
- authorized/denied join/leave command
- utterance capture start/stop
- speaker id / SSRC resolution
- STT latency and transcript text length
- brain latency for voice turns
- TTS request latency
- playback start/finish
- queue depth / dropped items / synthesis failures
- local voice turn/session ids for correlation in traces and the dashboard

If possible, fold this into the existing trace/log vocabulary instead of
inventing a separate logging universe.

## Public Interfaces / Behavior Changes

### Discord behavior

Add admin-only text commands for voice control:

- `!join [voice channel name]`
- `!leave`

No requirement to add slash commands in v1.
If Sandy is already in VC, `!join` should fail with a simple “already in a
voice chat” message rather than moving her automatically.

### New internal interfaces

Add internal adapter boundaries roughly like:

- `VoiceTransport`: join, leave, attach sink, play audio, query active session
- `SpeechToText`: transcribe captured utterance -> transcript result
- `VoiceTurnAssembler`: merge same-speaker STT fragments into one completed
  voice turn before the brain sees them
- `TextToSpeech`: synthesize text -> WAV bytes / playable audio source
- `VoiceSessionStore`: manage active session state and rolling voice history
- `VoiceTurnPolicy`: minimal voice gating for blank/garbage/non-actionable
  transcripts without dragging the full text bouncer into the voice path
- `VoicePromptContext`: compose the shared Sandy personality blocks plus
  voice-specific instructions and current participant/session context

### Memory / Recall representation

Voice turns should definitely enter vector memory with voice metadata. Recall
stays text-centric in v1.

## Expected Difficulties / Places Where Input Matters

### Product/policy decisions

These are the big ones that can turn into user thrash if left vague:

- exactly which Discord permission/admin rule counts as “allowed to control
  voice”
- whether voice transcripts should be visible in the same observability UI as
  text turns
- how aggressively to persist voice transcripts if STT quality is imperfect
- whether admin control is purely registry-driven or also checks Discord-native
  permissions as a fallback

### Technical risks

- `pipeline.py` is already large; dumping voice orchestration straight into it
  would be a bad decision
- Discord voice transport remains the shakiest external dependency surface
- voice memory will pollute long-term retrieval if transcript quality is bad and
  there’s no metadata-aware filtering
- long spoken replies will still feel sluggish unless the voice prompt is kept
  on a very short leash
- VAD / Discord speaking events can split one human thought into multiple
  transport utterances, so turn assembly has to reconstruct conversational
  turns without introducing too much latency
- mixing voice and text memories is useful, but retrieval formatting will need
  to surface modality cleanly so Sandy does not confuse “someone said this in
  voice” with “someone typed this in chat”
- SSRC must not be treated as a forever-stable user identifier; it can change
  and should be resolved dynamically within the session
- coalescing multiple completed human turns while Sandy is busy must not turn
  into an infinite backlog or response storm

### Human footguns

The likely dumb-human failure modes:

- letting the voice prompt drift until the brain starts speaking in full text
  paragraphs again
- trying to preserve text replies while Sandy is in VC and then acting surprised
  when voice latency feels broken
- stuffing all voice logic directly into `pipeline.py`
- importing faster TTS directly into main Sandy because “it works on my box”
- treating voice history and text `Last10` as the same thing too early
- overbuilding prosody/emotion ontologies before the integrated loop is stable
- treating SSRC like an identity primary key instead of transport metadata

## Test Plan

### Functional

- admin can make Sandy join and leave a VC
- non-admin cannot control voice join/leave
- `!join` while Sandy is already in VC fails cleanly
- `!leave` then `!join` starts a fresh session cleanly
- Sandy can capture a human utterance, transcribe it, reply, and speak back
- split same-speaker utterances are stitched into one turn before reply
- separate voice-session history updates correctly across multiple turns
- while Sandy is generating or speaking, newly completed human turns are
  coalesced into the next response context instead of triggering overlapping
  replies
- Sandy spoken replies are persisted as voice-originated text memories
- transcripts are persisted with speaker identity metadata
- reconnect/disconnect clears or resets voice session state cleanly
- Sandy auto-leaves after being alone for the configured idle interval
- text replies are paused while VC is active, but text messages can still be
  queued for background memory persistence

### Failure-path

- TTS service unavailable -> Sandy logs clearly and does not wedge the bot
- STT failure on one utterance does not kill the session
- synthesis/playback failure does not poison the next turn
- bot shutdown while connected to VC exits cleanly
- duplicate or stale queued playback does not survive leave/rejoin
- SSRC changes within a session do not corrupt speaker attribution
- blank or garbage transcripts are ignored without producing a reply
- if Discord/VAD splits one thought into multiple fragments, Sandy does not
  reply to the incomplete fragment prematurely
- if TTS fails for one turn, Sandy logs it, skips that spoken reply, and keeps
  the session alive

### Behavioral

- spoken replies are shorter than raw text replies
- retrieved memory can still inform voice turns
- voice turns do not contaminate text-channel short-term context directly
- repeated voice turns from different speakers preserve attribution cleanly
- current VC participant context is available to the brain and can affect reply
  content naturally
- Sandy does not reply in text chat while voice mode is active

## Assumptions / Defaults

- v1 voice control is admin-only
- v1 voice memory goes into vector/RAG with voice metadata
- v1 Recall handling for voice stays off
- v1 uses a separate rolling voice-session context, not text `Last10`
- v1 uses the current main Sandy brain with voice-specific prompt instructions
- v1 has no voice bouncer and no voice tool calls
- v1 remains non-streaming end-to-end
- v1 keeps the faster TTS backend as a separate HTTP service
- v1 assembles split STT fragments into completed speaker turns using simple
  deterministic Python logic
- v1 allows only one active VC globally
- v1 requires explicit `!leave` before moving to a different VC
- v1 auto-leaves if left alone in VC for too long
- v1 pauses foreground text replies while VC is active
- v1 coalesces completed human turns while Sandy is busy instead of attempting
  overlapping replies

Future work, not v1:

- streaming TTS/playback
- barge-in/interruption
- fully separate voice response prompt/path beyond the v1 modality-specific
  instructions
- richer prosody/style state
- slash-command/control-panel voice UX

## Do Not Do These Dumb Things

- do not merge voice into raw `pipeline.py`
- do not inline TTS dependencies into Prime
- do not preserve normal text replies while Sandy is in VC
- do not collapse voice history into text `Last10` on day one
- do not assume SSRC is stable forever



# Notes from the Human:

I agree with most of what's above.  A lot of the things that I was worried about are addressed above.

Thoughts on Spike code: 
There is nothing in voicemvp that I want to transfer as-is, whole cloth.  My objective here is to take the bits that work and carefully put them into sandy prime, but be very aware that there is an awful lot of cruft, dead code, and failed attempts in the spike.  Which is completely fine for the spike, but I want to keep the code in sandy prime healthy and clean.  
So, when we're looking for "Ok, how do we implement X", we look at the voicemvp and find where it is, but don't assume that voicemvp should be copied over line-for-line, rather use it as a directional reference to implement it in sandy prime.

Thoughts on Pipelines:
I agree that this is the right time to split the pipeline.  Possibly even into its own module - I'm thinking sandy/pipeline/controller.py and then one file for each pipeline step.  I don't know if that's fully the right way, since some of the steps are highly integrated, but to be honest, that seems like the cleanest and most supportable way.

Thoughts on admin access:  
We already have the registry database.  It would be easy to add a column to the user table that's like "voice-admin" which would be a bool.  
I think there's a foreign key relationship between users and servers (guilds) somewhere, so we could in theory, without doing a bad thing, create a system where (user+server) combinations can be admins. 
I.e. we have serverA and serverB.  Bob is on both.  Somewhere there's an entry in the registrydb that says "Bob's nickname on ServerA is whatever", so somewhere there's a foreign key row that has bob and serverA.  We could flip the bool from False to True in the voice-admin column and then bob would be able to do !join and !leave on serverA to make Sandy join a voice chat, but he wouldn't have that ability on serverB unless we also flipped the admin line on that row.  
There should also be an out-of-band tool to manage this, probably.  I.e. I wouldn't want to implement some sort of super admin permissions level that is like "people who can add people as admins", it'd just be a command line python script or something that only people who have CLI access (i.e. me) could run.

Thoughts on memory:
Sandy should have access to memories of things said in text chat when she is in voice chat (VC).  Just like a human would.  When I'm talking to Sam in the Snack Bandits server, and then we get into a VC, he doesn't forget the things that we typed in chat.  That's my guiding principle here.
I'm also thinking that, for the purposes of doing turns while she's in VC, it's fine if she starts with a blank whatever-we're-calling-last10 when she joins a voice chat.  She doesn't need to carry a short term memory into voice chat from the text chat.
But once she's in a voice chat, I think that things that are said should be added to some sort of short term memory cache similar to the last10, though I'm fine if we don't reuse that exact system.  But for the purposes of generating text responses, she probably should have access to the flow of the conversation.
I also think that things that are said in VC should get added to the RAG and become part of her long term memory.  Just like if I'm in VC with sam, and he says something, I will remember it a day later when we're talking in text chat.
I am still a bit ... unsure about voice chat messages getting added to recall.  Recall is more of a "photographic memory" and a "perfect representation of what's been said in text chat".  Adding that to recall might be odd; i.e. it might introduce errors into an otherwise objective memory system (STT transcribes something wrong, etc).  So, I dunno.  If we do, we should definitely start tagging messages with their modality so that there's a way to differentiate them.  But ultimately, this is kinda academic, since the bouncer wants to use recall as a tool pretty infrequently.

Thoughts on server isolation:
We should still do it.  Voice chats that exist on a server should become part of her memory on that server but should not bleed over into other servers.  In theory this means that sandy could have slightly different personalities (and WILL have different memories) on each server.
In addition, I'm fine if we restrict sandy to being in one voice chat globally at a time.  I don't see a world where she's in multiple VCs holding multiple conversations at a time.  She's a "person", not an entire call center.

Thoughts on latency:
Short phrases are generated by faster-qwen3-tts very quickly.  Longer phrases take substantial time.  Possibly at some point we should investigate streaming but that's a whole can of worms on its own; however, we should spend some effort making her respond with short phrases.  I mean, like, 40 words max kind of thing (not a hard number, I just pulled that out of my ass, but I'm thinking about a neurosama style interaction).  And mostly responses consisting of like 5-15 words.
Also, I'm wondering if we might want to consider cutting down the number of steps in the pipeline.  For instance, the image processor will never be used in the voice context.  But more than that... what if we eliminated the bouncer?  If she's in VC, it would be reasonable to assume that she's the center of attention and perhaps she will be responding to everything.  I'm not sold on this, but it's just a thought that occured to me.
And should we decide to eliminate the bouncer, we will also probably want to commit to not storing messages in recall (?) to avoid tagging, which would mean we could unload llama3 entirely.  Actually, let me talk about vram.

Thoughts on VRAM:
The voice model starts at about 4GB of vram.  Towards the end of my testing, it had grown to about 6GB.  So its context grows quickly.  That will be something that we want to keep a strong leash on.
Not having to invoke that stupid image processor at all in the concept of a voice pipeline means that's 11+GB that we don't have to account for.  We'll still have the brain using 18-23 of our 48GB.  And the voice model.  And whisper I guess but it was using like, 500mb, and same with the RAG model which is like 200MB.  
If we do want to add a prosody pass after whatever the brain generates, we just need to keep our vram restrictions in mind.  We have - for a home user at least - quite a lot to play with but it's not infinite.

Thoughts on brain:
Sigh, I dunno.  The current brain prompt has a lot of stuff in it that we should keep for personality purposes, but it also spits out longer phrases, often paragraphs.  Maybe we just tell it in the USER section to speak short sentences?  Or we add an injection into the SYSTEM prompt about "Sandy is currently talking in a voice chat and should keep her responses short" or whatever?   I dunno.  I would love to hear expanded / (expounded?) thoughts on this.

Thoughts on post-brain pass:
I think it's a good idea, I just have no specific idea of what we expect it to do precisely.  I don't know how to be like "Rewrite this in a way that the prosody doesn't suck but instead matches the intent of the words.".  Completely new territory to me.

Thoughts on observability: 
yes, we should add it to the observability - I don't think voice utterances magically are given a unique id by discord, so we'll have to tag them with some form of a uuid or whatever for tracing purposes and turn tracking.  Yes, we should do it.  Yes we should add it to the dashboard.
No, I don't think it needs to be in the same modal as existing tracing.

More thoughts on pipeline:
It's worth considering - and i'm neither convinced nor insisting - but it's worth considering, especially if we're splitting up pipeline into one file per step, if maybe we want a completely separate voice pipeline.  I.e. we have a text_controller and a voice_controller.py and where it makes sense we reuse the same, like, brain.py or whatever, but the two pipelines have different steps.  I dunno, just a thought.

General ending thoughts:
For the most part, the plan above is good.  These are just things that have been rattling around in my head since we started working on the voice spike.



# Update: notes from the human, next afternoon

 - We should account for weirdness with the speaker's mic.  I.e. if they are in the middle of a phrase and they pause to think, the discord audio compression / detection thing might stop transmitting, which is fine and normal, but the Sandy's STT will see two different utterances.
Proposal: we keep track of them.  Even if an incoming utterance is split into multiple audio segments, we still send them to the STT as soon as possible (latency), but we keep track of who said it and when the start/stop of it was.  After they come back from STT, if it's a.) by the same author and b.) continuing from something said, say, <1s before (i.e. the gap between utterances as measured by discord/opus/SSRC start/stop events), then we just stitch the text together before dropping it into the next segment of the pipeline. 
I guess this is an additional pipeline step - something like "stitcher" or whatever - but I mean, if someone was like "I went to the grocery store" and pauses in the middle, Sandy would otherwise get "I went to".  And then as a separate message "The grocery store".  That's not two different things she should respond to independently, that's the same thought.



# A second agent's assement of the plan

Voice Integration Plan — Assessment
What the plan gets right
The structural choices are solid. Separate sandy.voice package, not copying voicemvp wholesale, single-process architecture extending the existing bot, separate voice history from Last10, conservative Recall approach, TTS service boundary preserved, turn assembly before brain, admin-only via registry, auto-leave on idle, single VC globally. The "do not do" list at the bottom is well-calibrated to the actual codebase hazards. The human's annotations are practical and grounded.

What's missing or underdeveloped
1. The ollama lock is the elephant in the room
This is the biggest gap. OllamaInterface serializes all model calls (brain, bouncer, tagger, summarizer, vision, embedder) through a single asyncio.Lock. The plan discusses VRAM but never directly addresses what this means for voice latency.

If Sandy is in VC and text chat is active, a voice turn will queue behind whatever the text pipeline is doing (bouncer → brain → tagger → summarizer → two embed calls = up to 6 sequential model invocations before the voice brain call even starts).

Concrete things the plan should address:

Should voice brain calls get priority on the lock? (probably yes, but needs a mechanism — a priority queue, separate lock, or simply "skip text pipeline steps while in VC")
If the bouncer is eliminated for voice (the human's instinct is right), that removes one lock acquisition per turn, but tagger/summarizer/embed still compete
Should memory processing (tagger + summarizer + embed) for voice turns be deferred more aggressively — e.g., batched after the session ends?
2. Concurrent text + voice interaction model is unspecified
The plan doesn't address what happens when Sandy is simultaneously in VC and receiving text messages in the same server (or a different one). Options:

Text pipeline continues normally, voice competes for the lock (worst latency)
Text pipeline is throttled or paused while in VC (weird for text users)
Voice gets a fast path that bypasses or preempts certain lock-holders
This is an architectural decision that materially affects user experience. It should be explicit in the plan even if the v1 answer is "voice and text compete fairly on the lock, and we accept the latency."

3. The speech-render pass needs a concrete design, even if rough
The plan and human both acknowledge this is vague. But since voice latency is the critical metric, whether this is an LLM call or a rule-based pass is an architectural question, not a detail:

If LLM: adds another lock acquisition + inference round. On a 24B model, that's 1–3 seconds minimum. Total voice turn round-trip becomes STT + brain + render + TTS = potentially 8–15 seconds.
If rule-based: truncation, sentence splitting, markdown stripping, number-to-word conversion. Fast, deterministic, no lock. Probably good enough for v1.
Recommended for v1: Don't have a separate pass at all. Inject voice-specific framing into the brain's system prompt that constrains output length and style directly. One brain call instead of brain + render = significantly better latency. BRAIN_NUM_PREDICT is already configurable — a voice-specific VOICE_NUM_PREDICT=80 or similar enforces brevity at the generation level without an extra round-trip.
4. Voice turn-taking policy needs a concrete v1 default
The plan lists "whether Sandy should respond to every utterance" as an open product question. For implementation, this needs a stated default. Based on the human's notes about eliminating the bouncer for voice, the recommended v1 default:

Sandy responds to every non-trivial transcript when she's in VC. No bouncer.
Cheap rule-based gating only: skip blank/garbage transcripts, ignore transcripts under N characters, skip if she's currently generating.
This is faster (no bouncer lock acquisition), simpler, and matches the "she's the center of attention" mental model.
5. STT/Whisper model lifecycle and VRAM budget
The plan says "we have 48GB" and lists VRAM usage, but doesn't specify when Whisper loads into VRAM. The spike loads it at startup. Options for production:

Load on !join, unload on !leave — saves ~500MB–1GB idle, adds cold-start latency
Keep loaded always — wastes VRAM when not in VC
The TTS model (4–6GB) is already external via HTTP, so that's fine. But Whisper lives in-process and the plan should have an explicit position.

6. Preroll buffer not mentioned
The spike's 250ms preroll buffer — capturing audio before Discord's speaking-start event fires — is a proven mechanism that materially improves transcript quality at the attack of an utterance. It should be explicitly listed in the "carry forward" pieces; it's easy to overlook and easy to accidentally drop when the capture sink is rewritten.

7. CUDA library preloading for faster-whisper
The spike has a fragile but necessary ctypes.CDLL(RTLD_GLOBAL) hack to preload NVIDIA CUDA libraries so faster-whisper can find them at init time. This is environment-specific and will cause confusing failures on any fresh setup. The plan doesn't mention it. It should either be documented explicitly or replaced with a more robust approach (e.g., proper LD_LIBRARY_PATH in the systemd service file or startup wrapper).

8. Shutdown ordering
Existing shutdown is: flush memory queue → await background tasks → exit. Voice adds: disconnect from VC → flush TTS queue → stop STT worker → stop capture sink. The plan's test section says "bot shutdown while connected to VC exits cleanly" but doesn't specify ordering relative to the existing shutdown sequence.

Recommended ordering: voice session teardown should happen before pipeline.shutdown(), because in-flight voice turns may enqueue memory work. Getting this wrong causes hanging tasks or silently dropped state.

9. Registry admin column — schema migration needed
The human's proposal for a voice_admin bool on user_nicknames fits cleanly. However, the registry currently has no migration system (unlike Recall, which has auto-migrations v1→v4). Adding a column requires a decision:

Add a migration system to registry (overhead)
Use ALTER TABLE ... ADD COLUMN IF NOT EXISTS on init (pragmatic, matches the registry's current simplicity)
The plan should note this explicitly so the implementation doesn't make an ad-hoc choice.

10. Voice channel participant tracking
The plan says "voice-aware prompt context should include the current VC participant list" but doesn't specify who maintains it. Discord fires on_voice_state_update events for joins, leaves, and moves. Something needs to subscribe to those events and maintain a live roster. This is the VoiceTransport layer's job, but it's worth calling out because it isn't just "query the channel members" — it requires active event handling integrated with bot lifecycle.

11. Text normalization before TTS
The TTS handoff doc explicitly lists text normalization as the next priority after the backend was fixed. The voice plan mentions a "speech-render pass" but doesn't specifically address normalization as a distinct concern: numbers → words, abbreviations expanded, markdown artifacts stripped, contractions handled. This is distinct from the speech-render concept and should be its own step or at minimum its own function within the TTS adapter.

12. discord.py[voice] and voice_recv dependency
The spike depends on voice_recv.VoiceRecvClient and voice_recv.AudioSink for all audio capture. These come from discord.py[voice] extras. The plan doesn't mention this dependency at all. It's worth noting because:

It's the shakiest external dependency in the stack
It determines the entire audio capture API surface
Version pinning matters; these libraries have historically had compatibility gaps
13. TTS failure behavior during a session
The plan's test section says "TTS service unavailable → logs clearly, doesn't wedge." But v1 behavior should be explicitly specified: does Sandy fall back to typing in a text channel? Stay silent and continue listening? The recommended default: log the failure, skip that turn's spoken reply, continue the session normally, optionally post a text reply to the associated text channel.

Minor items
Audio format contract: 48kHz / 16-bit / 2-channel PCM is hardcoded in the spike's conversion path. Should be documented as the explicit contract between the TTS service and the voice transport layer.
Prompt factory methods: prompt.py needs new methods — at minimum voice_brain_prompt() and whatever drives the minimal non-bouncer transcript gating. These should be explicit tasklist items.
discord_message_id in Recall: Voice turns have no Discord message ID. The Recall schema has this field as nullable, which is fine, but query paths that assume it exists (e.g., the maintenance CLI) should handle the null case.
TTS instruct parameter: Currently hardcoded in the spike's env config. The production home for this (voice identity instructions, voice style prompt) should be specified — probably .env alongside other model config, but it should be listed explicitly.
Summary
The plan is solid at the architectural level. The biggest gaps are operational:

The single ollama lock's impact on voice latency — must be addressed explicitly; it's the most likely cause of a voice experience that feels broken even when everything else is working.
The speech-render pass needs at least a rough concrete shape — the recommendation is prompt injection over a separate LLM call for v1, which eliminates an entire lock acquisition and inference round.
The concurrent text + voice interaction model needs a stated default — even if the answer is "they compete fairly on the lock."
Everything else ranges from "should be documented" to "nice to specify." Items 1–3 will bite hard during implementation if left unaddressed.


# Human answers to the other agent's points

1.) The ollama lock

Fair point.  Honestly, I think for a v1 at least, we can just say "When Sandy is in a VC, she doesn't talk in text mode".  That eliminates lock contention.
I think it would be fine to continue to monitor text channels that she is in and shunt the messages off to be added to the RAG and to recall, via the low priority queue.  At least for v1.

2.) Concurrent text + voice interaction

Sandy is not some enterprise system that needs to be able to respond to 40 different voice chats and text channels simultaneously; she's a hobby project that I'm using to learn about AI.
Again, as said above, I think it's fine for the text chat to be effectively paused while she's in a VC.

3.) The speech-render pass design

I actually am in support of this agents' suggestion of giving the brain voice-specific framing and skipping a post-brain "clean up" pass.  Yes, eventually I want to get into prosody and emotional lerp, but for v1, 
i think it's fine to just have a brain prompt for voice and a brain prompt for text.
I did this with a previous project - I had the prompt split into various text strings that I just concatenated together as needed to form the prompt.  We could probably do that - all the personality suff as 
a "get_personality()" or whatever, and then the instructions bit as a "get_instructions(modality: voice|text)" or whatever, add those together and that becomes the system prompt.  I mean, something
like that.
We can re-do that later for emotional intelligence with a lerp, and for prosody, and all that, but hell, that brain model is 18b - it should be smart enough to do some basic instructions.
Caveat: it's possible that changing the NUM_PREDICT for the brain model will cause the vram to thrash.  I don't /think/ so, but I know previously when we had the same model load with different CTX numbers configured, 
it would treat every combo of (model+num_ctx) as a separate invocation of the model.  I don't even know that's a deal breaker - if the big brain model is invoked with one NUM_PREDICT for text and
another NUM_PREDICT for voice, even if that causes the VRAM to thrash, we can just warm it when joining voice, and as decided in 1. and 2. above, she won't do voice and text at the same time, so even if it
thrashes, it's probably fine.  Thrashes will be on the border of switching between text and speech mode.

4.) Voice turn-taking policy 

I kinda like this agent's suggestion.  Honestly, if we reduce the scope such that Sandy won't do tool calls at least in voice v1, and we commit to responding to every message that's not obvious garbage, then
I think that it's possible to skip the bouncer in the pipeline.  I'm willing to be convinced of the need for it, and honestly, it'd be nice to have tool calls, but ... there is something to be said for the
simplicity of the idea.  

5.) STT/Whisper / VRAM budget

I'm just not concerned about this.  When we're doing voice mode, whisper uses very little vram (<1GB), and the TTS was using ~6GB.  The RAG uses like 500mb, and the big brain uses something like 16-20GB.  That's
call it 25GB min and 30GB for safety.  Even if we still load llama3 for tagging/summarizing of recall (for text chats while she's in VC), and potentially for bouncer, that's still like 38GB.  We should 
be well under budget.

6.) Preroll buffer

Yes, we should keep this.  It's good.  If it's not called out explicitly, it should be. 

7.) CUDA libraries and hacks

I agree it's ugly, I just don't want to fix it now.  Obviously the answer is to create some sort of startup script that loads LD_LIBRARY_PATH as an env var.  We'll get there.  I kinda doubt we're going to
ever turn this into a systemd service, but maybe.  For now, I'm fine with the hack.  I don't get what the other agent is talking about with it being environment specific - it's just spelunking in the
site-packages, isn't it?  That should be ... i mean, not deterministic, but similar no matter where you install the deps with pip.  Maybe i'm missing something.

8.) Shutdown ordering

Sure.  Whatever.  Implementation detail where I think the path forward is the obvious one.  When Sandy is told to leave voice, have her physically (you know what i mean) disconnect from the voice channel, 
wait for all in progress pipeline steps to finish, then convert back to text mode.  If there is a pending TTS to send to the VC, it can just get dropped silently or logged.  I find myself not caring 
overly much how this is set up.
Also if we catch a ^C while she's in a voice chat, disconnecting should be part of the cleanup - same as above.

9.) Registry db + admin + schema change

Yeah, we should just create the same migration system for recall.  Copy, paste, done.  NBD.

10.) Voice channel participant tracking

Yep.  On join, figure out who's in the channel.  Track it.  On update, update the tracked status.  Inject it into the prompt in the brain.  I just thought this was obvious, so I'm not sure why it's
being called out.  Is the other agent afraid of telling me that we'll have to track state?  We're gonna have to track state.  We have to track state all over the whole project.  It's fine.

11.) TTS smoothing

Didn't we address this above?  I think for v1 we'll just add instructions to the brain model.  It is an INSTRUCT model anyway.  If we need a post-brain pass, we'll get there.   This is the kinda
shit that makes me want to lose motivation for the project, it's an implementation detail that is just ... it's a future problem, can we just move forward, fucking please?

12.) discord-ext-voice-recv

Yes, it should be mentioned.  We spent a lot of time in the spike proving it out.  Especially since currently `main` in that project doesn't even work; we had to pin it in the pyproject to a specific
git hash because there's a pending pull request from a 3rd party that fixes this 3rd party library.  Yes, it's fragile.  It's still the best option we have.

13.) TTS failure

No, don't fall back to typing in text.  Or maybe put an error message in text chat.  But probably not - for now we should just log an error on the console.


Minor items:

 - Audio format: whatever, don't care.
 - prompt methods: already discussed - I think splitting the prompt into a get_instructions and get_personality is a decent idea, we can iterate on something simlar to that.
 - discord_message_id in recall - we're skipping recall for voice comms in v1.  We'll fuck this pig when we get to it.
 - TTS instruct parameter: are we even using this anymore, since we're doing a clone?  Whatever, i'm sure the implementation is obvious.
