# Getting Started: clone, set up and run

This guide takes you from nothing to a working voice assistant that answers questions about ZTBL's
Islamic Banking FAQs, in about 45 minutes. Most of that time is spent on one-off downloads and
document processing.

When you finish you will have:

| Address | What it is |
|---|---|
| http://localhost:8000/ | **User page**: tap the mic, ask a question, hear the answer |
| http://localhost:8000/voice/console.html | **Developer console**: timings, sources, retrieval traces |
| http://localhost:8000/docs | API reference (Swagger) |

---

## 1. Install the prerequisites

| Tool | Why | Notes |
|---|---|---|
| **Git** | Clone the code | https://git-scm.com |
| **Docker Desktop** | Runs the database, the vector store and the API | https://www.docker.com/products/docker-desktop. Give it **at least 8 GB of RAM** (Settings → Resources). |
| **Python 3.11** | Runs the helper scripts (loading documents, tests) | Exactly 3.11; 3.12+ is not supported. On Windows, `py -3.11 --version` should work. |
| **About 15 GB free disk** | Docker images, models, data | |

You also need two API keys:

| Key | Used for | Where to get it |
|---|---|---|
| **Groq** (`gsk_...`) | The LLM that writes answers, and speech-to-text (Whisper) | https://console.groq.com/keys |
| **Soniox** | Text-to-speech (the spoken answers) | https://console.soniox.com. Optional: without it, answers come back as text only. |

And **access to the GitHub repository** `Pivot-Point-AI/hybrid-rag`, which is private. Ask an
admin to add your GitHub account.

---

## 2. Clone the code

```powershell
git clone https://github.com/Pivot-Point-AI/hybrid-rag.git
cd hybrid-rag
git checkout ztbl-hybrid-rag
```

> **Which branch?** `ztbl-hybrid-rag` has everything: the ZTBL documents, voice, the user page,
> Groq support and the evaluation tools. `master` holds only the original base platform.

---

## 3. Create your configuration (`.env`)

```powershell
copy .env.example .env      # macOS/Linux: cp .env.example .env
```

Open `.env` in an editor and change these lines:

```ini
# 1) Choose your own admin password for creating tenants (any long random string)
SECURITY__ADMIN_API_KEY=pick-a-long-random-secret

# 2) Delete this line (it starts a local Ollama LLM, which we don't use):
# COMPOSE_PROFILES=local-llm

# 3) Use Groq for answers (replace the three LLM__ lines)
LLM__PROVIDER=groq
LLM__MODEL=openai/gpt-oss-120b
LLM__API_KEY=gsk_your_groq_key

# 4) Voice: speech-to-text reuses the Groq key automatically; add the Soniox key for speech
VOICE__TTS_API_KEY=your_soniox_key
VOICE__STT_PROMPT=ZTBL, Zarai Taraqiati Bank, Islamic banking, Riba, Riba An-Nasiyah, Riba Al-Fadl, Ijarah, Ijarah wa Iqtina, Murabaha, Musawamah, Musharakah, Mudarabah, Rab-ul-Maal, Mudarib, Salam, Istisna, Wakalah, Kafalah, Rahn, Shariah, Qarz-e-Hasna, Zarai Amadani Certificate, Bakht, Assan
```

`VOICE__STT_PROMPT` teaches speech recognition the Islamic-finance vocabulary. In testing, exact
transcriptions of jargon-heavy questions went from 1 in 8 to 7 in 8.

Leave everything else as it is. `.env` is git-ignored: never commit it, because it holds your keys.

---

## 4. Start the platform

Make sure **Docker Desktop is running**, then:

```powershell
docker compose up -d --build
```

What happens:
- `postgres` and `qdrant` start. Their data lives in Docker volumes and survives restarts.
- `migrate` creates the database tables, then exits.
- `api` starts and, **on the first run only**, downloads the embedding and reranker models (about
  2 GB). This can take **5–15 minutes**.

Follow progress, and press `Ctrl+C` to stop watching; the services keep running:

```powershell
docker compose logs -f api
```

The platform is up when this returns `"status": "ok"`:

```powershell
curl http://localhost:8000/health
```

---

## 5. Set up the Python helper environment

The helper scripts (loading documents, asking from the terminal, tests) run on your machine:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Run this once. Afterwards, just activate (`.venv\Scripts\activate`) in each new terminal.

---

## 6. Load the ZTBL documents

This creates a tenant (an isolated customer space) and a knowledge base, then uploads the
documents in `corpora/ztbl/`:

```powershell
python scripts/load_corpus.py --dir corpora/ztbl --tenant-slug ztbl --tenant-name "Zarai Taraqiati Bank Limited" --kb islamic-banking --api http://localhost:8000 --admin-key pick-a-long-random-secret --source ztbl-website
```

Use the same admin key you put in `.env`.

- The tenant's API key is saved to `.secrets/ztbl.json` (git-ignored). Keep it private.
- The script waits until processing finishes. On a laptop CPU this takes **about 20 minutes**,
  because every text chunk is turned into an embedding. It is a one-off cost.
- At the end it prints a line like **`knowledge_base_id=be0656e0-...`**. Copy that ID.

Re-running the script is safe: unchanged files are skipped, and changed files become new versions.

---

## 7. Turn on the simple user page

The user page needs no login. It answers from one knowledge base that you choose. Add these two
lines to `.env`, using the ID from step 6:

```ini
VOICE__PUBLIC_KNOWLEDGE_BASE_ID=be0656e0-...
VOICE__PUBLIC_TITLE=ZTBL Islamic Banking Assistant
```

Apply the change:

```powershell
docker compose up -d
```

> Anyone who can open this page can ask about that knowledge base without a key. That is fine for
> public FAQ content, but don't point it at confidential documents.

---

## 8. Try it

**User page**: open **http://localhost:8000/**, tap the blue mic and ask, for example:
- "What is the financing limit for a rice transplanter?"
- "How are losses shared in Musharakah?"
- "Who are the members of the Shariah Board?"

Allow microphone access when the browser asks. Tap the mic while it is speaking to interrupt it.
You can also type questions.

**Developer console**: open **http://localhost:8000/voice/console.html**, paste the API key from
`.secrets/ztbl.json`, pick `islamic-banking` and click **Connect & enable mic**. Each answer shows
its sources, a latency breakdown and an **Inspect trace** button that explains why each passage
was chosen.

**Terminal**:

```powershell
python scripts/ask.py --tenant-slug ztbl --kb islamic-banking --api http://localhost:8000 "What is Ijarah?"
```

Expect each answer to take **15–40 seconds** on a CPU-only machine; the reranking step dominates.
The developer console shows where the time goes.

---

## 9. Everyday commands

| Task | Command |
|---|---|
| Start everything | `docker compose up -d` |
| Stop everything (data is kept) | `docker compose down` |
| Watch the API logs | `docker compose logs -f api` |
| Rebuild after pulling new code | `git pull` then `docker compose up -d --build` |
| Run the tests | `docker compose up -d postgres qdrant`, then `pytest` |
| **Delete all data** (start over) | `docker compose down -v`. This erases the database, vectors and downloaded models. |

---

## 10. Troubleshooting

**"Port is already allocated" or "ports are not available."** Another program uses that port. Pick
other ports in `.env`, then run `docker compose up -d`:

```ini
API_PORT=8080         # then open http://localhost:8080/
POSTGRES_PORT=15432   # already avoids a locally installed PostgreSQL on 5432
QDRANT_PORT=16333
```

On Windows, some port ranges are reserved by the system. `netsh int ipv4 show excludedportrange
protocol=tcp` lists them; choose a port outside those ranges.

**The page loads but every question says "Sorry, something went wrong", and the API logs show
`CERTIFICATE_VERIFY_FAILED ... self-signed certificate in certificate chain`.** Security software
on your PC (for example Kaspersky, Zscaler or a corporate proxy) is inspecting HTTPS traffic.
The best fix is to ask IT to exclude `api.groq.com` and `tts-rt.soniox.com` from HTTPS scanning.
If you can't, make the container trust your machine's inspection certificate.

1. Export the inspection certificate. Example for Kaspersky in PowerShell; change the name for
   other products:

   ```powershell
   mkdir docker\certs -Force
   $c = Get-ChildItem Cert:\LocalMachine\Root, Cert:\CurrentUser\Root | Where-Object Subject -like "*Kaspersky*" | Select-Object -First 1
   "-----BEGIN CERTIFICATE-----`n" + [Convert]::ToBase64String($c.RawData,'InsertLineBreaks') + "`n-----END CERTIFICATE-----" | Set-Content -Encoding ascii docker\certs\proxy-root.pem
   ```

2. Build a bundle of the standard certificates plus yours:

   ```powershell
   python -c "import certifi,pathlib; pathlib.Path('docker/certs/ca-bundle.pem').write_text(pathlib.Path(certifi.where()).read_text()+'\n'+pathlib.Path('docker/certs/proxy-root.pem').read_text())"
   ```

3. Create `docker-compose.override.yml` in the project root. It is git-ignored and Docker loads it
   automatically:

   ```yaml
   services:
     api:
       volumes:
         - ./docker/certs:/certs:ro
       environment:
         SSL_CERT_FILE: /certs/ca-bundle.pem
   ```

4. Apply it with `docker compose up -d`. Only do this on your own development machine.

**`/health` shows an error, or the API keeps restarting.** Run `docker compose logs api`. Usually
Docker has too little memory (raise it to 8 GB or more) or the first model download was
interrupted (run `docker compose up -d` again).

**The page says voice is unavailable, or answers come back as text with no audio.** A voice key is
missing or wrong: `LLM__API_KEY` (Groq, also used for speech-to-text) or `VOICE__TTS_API_KEY`
(Soniox). Fix `.env`, then run `docker compose up -d`.

**The browser never asks for the microphone, or the mic does nothing.** Browsers allow the mic only
on `http://localhost` or HTTPS. Open the page on the same machine as `localhost`, not by IP
address, and check the site's microphone permission (the icon in the address bar).

**The user page says "The assistant is not available right now."** Step 7 is missing:
`VOICE__PUBLIC_KNOWLEDGE_BASE_ID` is not set, or the API was not restarted after setting it.

**`load_corpus.py` fails with 401.** The `--admin-key` does not match `SECURITY__ADMIN_API_KEY`
in `.env`. If you changed `.env`, run `docker compose up -d` first.

**`load_corpus.py` says the tenant exists but its key is not in `.secrets/ztbl.json`.** The
tenant was created on another machine or before a reset. Pass its key with `--api-key rag_...`, or
use a new `--tenant-slug`.

---

## Where to go next

- [README.md](../README.md): what the platform does, the API overview and the design.
- [docs/ARCHITECTURE.md](ARCHITECTURE.md): how it works inside (retrieval, ranking, voice pipeline).
- [docs/REUSE_GUIDE.md](REUSE_GUIDE.md): adding your own documents and products, switching
  models, deploying.
- [docs/ZTBL.md](ZTBL.md): the ZTBL setup and its measured quality.
