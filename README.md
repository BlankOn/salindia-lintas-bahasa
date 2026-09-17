# Salindia — live Indonesian ⇄ English subtitles

Speak Indonesian or English into your browser and read the translation as
subtitles at the bottom of the screen. Toggle the direction (ID → EN / EN → ID)
with the buttons or the `T` key.

Space is deliberately left free for your slides.


## Option 1: Local model

All open weights, all running through MLX on the Metal GPU:

| Stage | Model | License |
|---|---|---|
| Speech → text (both languages) | `whisper-large-v3-turbo` | MIT |
| Indonesian → English | `Qwen3-4B-Instruct-2507` 4-bit | Apache 2.0 |
| English → Indonesian | `Qwen3-8B` 4-bit | Apache 2.0 |

EN → ID uses the larger model because the 4B one converted currencies
($ → Rp) and mixed *saya*/*aku*. The models add up to about 9 GB of memory.
On a 16 GB Mac, set both `MT_MODEL_*` variables to the 4B model.


## Option 2: OpenAI

Each finished sentence is sent to OpenAI (`whisper-1` by
  default, or a `gpt-4o-*-transcribe` model). The key can come from
  `OPENAI_API_KEY` in `.env`, or be typed into the chooser, in which case it is
  kept in server memory only.

With the OpenAI engine there is also an **Auto** direction: each sentence's
language is detected, and English is translated to Indonesian and Indonesian to
English.

See [DESIGN.md](DESIGN.md) for the reasoning, measurements and open work.

## Running it

```bash
./run.sh
```

The first run creates a venv, installs dependencies, and downloads about 8 GB
of weights to `~/.cache/huggingface`.

Then open http://127.0.0.1:8000, pick a
direction, and press **Start listening**. Use **Show original text** and
**Show latency** to choose what each subtitle shows.

Copy `.env.example` to `.env` to change anything.


## Deploying it

`Dockerfile` and `docker-compose.yml` are for a server, which means the OpenAI
engine: MLX is Apple-Silicon only, so the local models cannot run in a
container. The image installs no MLX wheels (the requirements are marked for
macOS and the imports are lazy) and the compose file sets
`FORCE_APPROACH=openai`, so the browser never offers an engine that would fail.

```bash
cp .env.example .env            # set OPENAI_API_KEY and ACCESS_PASSPHRASE
docker compose up -d --build
```

The talk records are a SQLite file on the host, mounted at `/data` — a
directory rather than the file itself, because WAL mode writes `-wal` and
`-shm` siblings next to it. Rebuilding the image leaves them alone. The
directory is created and chowned on first start, so a `./data` that Docker made
as root still works; uvicorn itself runs as an unprivileged uid either way.

Nothing else needs a volume: with the OpenAI engine no weights are downloaded,
so a restart re-downloads nothing. The local MLX models are the only thing that
would need a cache, and they cannot run in a container at all.

Nothing is published on the host: the container joins the reverse proxy's
network and is reached by name (`http://salindia:8000`). TLS is not optional in
front of it — microphone capture needs a secure context, so browsers refuse it
over plain http on anything but localhost.


---

License: MIT

> **Note:** This project has been developed with assistance from LLMs. The maintainers remain responsible for reviewing, testing, and maintaining the code.

