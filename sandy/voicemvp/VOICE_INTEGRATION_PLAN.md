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
- run the existing brain pipeline with a voice-specific input wrapper
- run a small speech-render pass before TTS so spoken replies stay short and
  natural
- speak the final reply back into VC

This is the right stopping point for the spike and the right starting point for
integration. Do not do streaming, barge-in, or a fully separate voice-brain
persona path in v1.

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
- `orchestrator` layer: turns transcript -> Sandy brain -> speech-render ->
  playback

This keeps the spike’s useful code but stops the MVP blob from becoming
production architecture.

### 2. Keep Discord text bot and voice bot in one process, but separate concerns

Do not run a second Discord client for voice in production. Extend the existing
Sandy Discord runtime so the main bot can also handle voice commands and own
voice session state.

Implementation intent:

- `bot.py` remains Discord lifecycle/event glue
- message/text handling stays in `pipeline.py`
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
voice metadata, but do not assume they belong in Recall by default.

Why:

- voice transcripts and replies should still become part of Sandy’s long-term
  awareness
- this avoids building a separate dead-end voice-only memory system
- it allows normal retrieval to stay useful across modalities
- voice transcripts should definitely enter vector/RAG memory
- voice transcripts should enter Recall only if there is explicit modality/source
  handling and you are comfortable treating STT output as a non-literal record

Needed metadata:

- modality: `voice`
- source: `stt`
- guild id / voice channel id
- speaker user id and display name
- utterance timestamp
- maybe a boolean for `is_transcript` vs `is_spoken_reply`

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

### 6. Reuse the main brain, but add a voice input wrapper and a speech-render pass

For v1, use the existing Sandy brain path, not a totally separate voice brain.
But do not feed raw transcript directly to the normal text reply and call it
done.

Flow:

1. STT transcript arrives
2. assemble same-speaker STT fragments into one conversational turn when they
   are clearly part of the same thought
3. build a voice-aware user turn for the brain
4. run the normal Sandy brain generation
5. run a speech-render step that turns the brain output into a short speakable
   reply
6. send the speech-rendered text to TTS
7. append Sandy’s spoken text to the voice-session history and long-term memory

Voice-aware brain input should preserve:

- speaker identity
- recent voice-session context
- relevant Recall/RAG context
- current VC participant list

Speech-render v1:

- not a separate full model/provider abstraction yet
- just a small post-brain render pass to shorten, de-textify, and keep replies
  speakable
- this matches the chosen first-pass goal: short spoken reply now, fully
  separate voice prompt/path later

Future path:

- later replace this with a true voice-specific response prompt/path if needed

Turn assembly / stitcher v1:

- STT should run on captured audio segments as soon as possible; do not hold raw
  audio waiting for a perfect end-of-thought boundary
- after STT returns, fragments from the same speaker should be merged in Python
  when they are clearly continuation pieces rather than separate turns
- stitching should be deterministic and cheap: string concatenation plus light
  whitespace/punctuation cleanup, not an LLM
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
- speech-render latency
- TTS request latency
- playback start/finish
- queue depth / dropped items / synthesis failures

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
- `SpeechRenderer`: brain text -> short speakable text
- `VoiceTurnPolicy`: minimal voice gating for blank/garbage/non-actionable
  transcripts without dragging the full text bouncer into the voice path

### Memory / Recall representation

Voice turns should definitely enter vector memory with voice metadata. Recall
should be treated more cautiously in v1; it may remain text-centric at first, or
store voice only with explicit modality/source handling.

## Expected Difficulties / Places Where Input Matters

### Product/policy decisions

These are the big ones that can turn into user thrash if left vague:

- exactly which Discord permission/admin rule counts as “allowed to control
  voice”
- whether voice transcripts should be visible in the same observability UI as
  text turns
- how aggressively to persist voice transcripts if STT quality is imperfect
- whether Sandy should respond to every utterance, or only when directly
  addressed / after turn-detection rules
- whether voice transcripts should enter Recall in v1 or only vector memory

### Technical risks

- `pipeline.py` is already large; dumping voice orchestration straight into it
  would be a bad decision
- Discord voice transport remains the shakiest external dependency surface
- voice memory will pollute long-term retrieval if transcript quality is bad and
  there’s no metadata-aware filtering
- long spoken replies will still feel sluggish unless the speech-render pass is
  kept tight
- VAD / Discord speaking events can split one human thought into multiple
  transport utterances, so turn assembly has to reconstruct conversational
  turns without introducing too much latency
- mixing voice and text memories is useful, but retrieval formatting will need
  to surface modality cleanly so Sandy does not confuse “someone said this in
  voice” with “someone typed this in chat”
- SSRC must not be treated as a forever-stable user identifier; it can change
  and should be resolved dynamically within the session

### Human footguns

The likely dumb-human failure modes:

- trying to skip the speech-render pass and speaking raw brain paragraphs
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
- Sandy spoken replies are persisted as voice-originated text memories
- transcripts are persisted with speaker identity metadata
- reconnect/disconnect clears or resets voice session state cleanly
- Sandy auto-leaves after being alone for the configured idle interval

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

### Behavioral

- spoken replies are shorter than raw text replies
- retrieved memory can still inform voice turns
- voice turns do not contaminate text-channel short-term context directly
- repeated voice turns from different speakers preserve attribution cleanly
- current VC participant context is available to the brain and can affect reply
  content naturally

## Assumptions / Defaults

- v1 voice control is admin-only
- v1 voice memory goes into vector/RAG with voice metadata
- v1 Recall handling for voice remains conservative and may stay text-only at
  first
- v1 uses a separate rolling voice-session context, not text `Last10`
- v1 uses the current main Sandy brain plus a post-brain speech-render pass
- v1 remains non-streaming end-to-end
- v1 keeps the faster TTS backend as a separate HTTP service
- v1 assembles split STT fragments into completed speaker turns using simple
  deterministic Python logic
- v1 allows only one active VC globally
- v1 requires explicit `!leave` before moving to a different VC
- v1 auto-leaves if left alone in VC for too long

Future work, not v1:

- streaming TTS/playback
- barge-in/interruption
- fully separate voice response prompt/path
- richer prosody/style state
- slash-command/control-panel voice UX

## Do Not Do These Dumb Things

- do not merge voice into raw `pipeline.py`
- do not inline TTS dependencies into Prime
- do not skip speech rendering
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
