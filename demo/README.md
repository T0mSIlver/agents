# SelfHostedSTT demo (not part of the PR)

Builds a 30 s clip from vLLM's public test audio, plays it into an AgentSession
(Silero VAD + `mistralai.SelfHostedSTT`) at real-time pace, and prints interim and
final transcripts with the time from the VAD's end of speech to each final.

```bash
# 1. Voxtral Realtime on vLLM (localvoxtral checkout on the dev box)
~/work/localvoxtral/scripts/linux/voxtral-vllm.sh up

# 2. this branch
uv sync --all-extras --dev
git lfs pull --include "livekit-plugins/livekit-plugins-silero/**"   # silero_vad.onnx

# 3. the clip: mary_had_lamb (16 s) + 2 s silence + the first 12 s of winning_call
B=https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset
curl -sO $B/mary_had_lamb.ogg && curl -sO $B/winning_call.ogg
ffmpeg -v error -y -i mary_had_lamb.ogg -f lavfi -t 2 -i anullsrc=r=16000:cl=mono -i winning_call.ogg \
  -filter_complex "[0:a]aresample=16000,aformat=channel_layouts=mono[a];[1:a]aformat=channel_layouts=mono[b];[2:a]aresample=16000,aformat=channel_layouts=mono,atrim=0:12[c];[a][b][c]concat=n=3:v=0:a=1" \
  -ar 16000 -ac 1 demo.wav

# 4. run it (record with asciinema: asciinema rec demo.cast -c "...")
uv run python demo/self_hosted_voxtral_demo.py --audio demo.wav --base-url http://127.0.0.1:8000
```
