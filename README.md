# Local LLM stack: llama.cpp + llmlocal + opencode

End-to-end setup for serving a GGUF model from a local GPU over an OpenAI-compatible
API, and pointing opencode at it. Every command below was run and verified on
an RTX 5090.

Three pieces:

| Piece | What it does | Lands at |
|---|---|---|
| `llama` | llama.cpp's unified CLI — downloads GGUFs and serves them | `~/.local/bin/llama` |
| `llmlocal` | wrapper in this repo: model menu, hash-verified downloads, restarts the server | `~/.local/bin/llmlocal` |
| `opencode` | coding agent TUI/CLI that talks to the server | `~/.opencode/bin/opencode` |

Once set up, `llmlocal` boots a server on `0.0.0.0:8080` and opencode reaches it at
`http://127.0.0.1:8080/v1`.

---

## 1. Prerequisites

You need an NVIDIA GPU with a working driver, and enough disk for the models.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h ~/.cache
```

- **GPU memory.** Both models below are ~16.5 GB on disk and want ~28 GB of VRAM at
  the 128K context `llmlocal` uses. A 32 GB card fits one at a time with a few GB
  spare; a 24 GB card does not, and you will need to lower `-c` (see §7).
- **Disk.** ~34 GB in `~/.cache/huggingface` for both models.
- **CUDA toolkit: not required.** The `llama` binary ships its own statically linked
  cuBLAS and dynamically links only `libcuda.so.1` from the driver. There is no `nvcc`
  on this machine and none is needed.
- **WSL2 note.** The driver library comes from `/usr/lib/wsl/lib/libcuda.so.1`, installed
  by the *Windows* NVIDIA driver. Don't install a Linux driver inside WSL — it breaks this.

---

## 2. Install llama.cpp

```bash
curl -LsSf https://llama.app/install.sh | sh
```

This drops a single ~530 MB static CUDA binary at `~/.local/bin/llama`. No source tree,
no build, no `cmake`. Confirm:

```bash
llama --version
# version: 0.3.0-dev (build 10679, commit 50f068fff)
# built with GNU 12.3.0 for Linux x86_64
```

If `llama` is not found, `~/.local/bin` is not on your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

The subcommands that matter:

```
llama serve      HTTP API server
llama cli        interactive terminal chat
llama download   fetch a model into the cache
llama update     re-run the installer to upgrade in place
```

`llama update` pulls from the same `llama.app/install.sh` the install used, so upgrading
is one command. Pin a version by not running it — this stack is sensitive to
llama.cpp regressions, so upgrade deliberately, not on a whim.

---

## 3. Install llmlocal

The script lives in this repo. Install it and create its config directory:

```bash
install -m 755 llmlocal ~/.local/bin/llmlocal
mkdir -p ~/.config/llmlocal
llmlocal --help
```

`llmlocal --help` prints the script's own header comment, which is the authoritative
usage reference. Two files in `~/.config/llmlocal/` hold its state, both created
automatically:

- `last` — the model served most recently. This becomes the menu default.
- `models` — optional extra menu entries, one `repo[:quant] [extra llama serve flags]`
  per line. `#` comments and blank lines are ignored; flags split on whitespace with no
  quoting. Override the path with `$LLMLOCAL_MODELS`.

The two built-in entries are hardcoded in the `MODELS` array at the top of the script.
Everything else is additive via `models`.

---

## 4. Set up the two models

Both are Qwen3.8-27B at Q4_K_M, and both are already in the cache on this machine.
On a fresh box, fetch them explicitly:

```bash
llmlocal --download unsloth/Qwen3.8-27B-GGUF:Q4_K_M
llmlocal --download 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M
```

| # | Model | On disk | Notes |
|---|---|---|---|
| 1 | `unsloth/Qwen3.8-27B-GGUF:Q4_K_M` | 16.5 GB (`Qwen3.8-27B-UD-Q4_K_M.gguf`) | Text only. Unsloth's UD (dynamic) quant. The default. |
| 2 | `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M` | 16.5 GB + 0.6 GB `mmproj` | Abliterated — refusal behavior removed. The `mmproj` file makes it **vision-capable**. |

Prefer `llmlocal --download` over a bare `llama download`. It does the same fetch, then
**sha256-verifies every cached GGUF for that repo** against the hash the cache filename
encodes, deletes a mismatch, and retries once. This exists because a torn resume once
produced a 20 GB blob that loaded fine and answered every prompt with `////`. Because it
re-hashes *all* of the repo's blobs, `--download` doubles as a repair tool for a model
you already have — and it leaves a running server alone, so the current model keeps
answering while the new one downloads.

Verify both landed:

```bash
llmlocal --list
#   1) unsloth/Qwen3.8-27B-GGUF:Q4_K_M                                      [cached]  (default)
#   2) 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M     [cached]
```

`[cached]` reads `llama serve --cache-list`. An uncached entry shows `[download 16.5 GB]`,
the size coming from the Hugging Face manifest — so the menu tells you what a pick will
cost before you commit to it.

### Adding a third model

Any Hugging Face GGUF repo works without touching the script:

```bash
llmlocal unsloth/gpt-oss-20b-GGUF:Q8_0        # one-off, downloads if needed
echo 'unsloth/gpt-oss-20b-GGUF:Q8_0' >> ~/.config/llmlocal/models   # add to the menu
```

Per-model flags go on the same line, after the spec:

```
# ~/.config/llmlocal/models
unsloth/gpt-oss-20b-GGUF:Q8_0 -c 262144 --temp 0.7
```

---

## 5. Run the server

```bash
llmlocal          # menu; Enter takes the default (last used, else entry 1)
llmlocal 2        # entry 2, no prompt
llmlocal --dry-run 2   # print the commands instead of running them
```

**Gotcha:** with no TTY — a script, a pipe, a CI step, an agent shell — the menu read
hits EOF and *silently takes the default*, starting a server. Always pass an explicit
model number in non-interactive contexts, or use `--list` / `--dry-run` if you only meant
to look.

What a serve run actually does, in order:

1. Resolves your pick to a `repo:quant`.
2. If it isn't cached, downloads and hash-verifies it — **before** touching the running
   server, so the old model keeps serving through the download.
3. Writes the pick to `~/.config/llmlocal/last`.
4. Kills whatever holds the target port (found via `ss -tlnp`), waiting up to 10s.
5. `exec`s `llama serve`.

The resulting command line, with `--dry-run` to show it:

```bash
llama serve -hf <model> --alias <model> --offline \
  --jinja -c 131072 -fa on --cache-type-k q8_0 --cache-type-v q8_0
```

`--alias <model>` is the important part: it makes `GET /v1/models` report the
`repo:quant` string as the model id, which is what opencode and `.ortusrc` pin.

### Host and port

Defaults to `0.0.0.0:8080`, from `LLAMA_ARG_HOST` / `LLAMA_ARG_PORT` — the same variables
`llama serve` itself reads. Binding to all interfaces is deliberate: Claude Code sandboxes
relay through a proxy to this host, and their `NO_PROXY` makes clients skip the proxy for
`127.0.0.1`, so a loopback-only bind is unreachable from inside one.

```bash
LLAMA_ARG_PORT=9931 llmlocal 2      # via environment
llmlocal 2 --port 9931              # explicit flag; wins over the variable
```

Anything `llmlocal` doesn't recognize passes straight through to `llama serve` and
overrides the built-in flags. **Put the model before pass-through flags** — the first
bare argument is read as the model selection.

### Verify

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool | head

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
       "messages":[{"role":"user","content":"Reply with exactly: llmlocal works"}],
       "max_tokens":40,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
# llmlocal works
```

---

## 6. Install and wire up opencode

```bash
curl -fsSL https://opencode.ai/install | bash
opencode --version    # 1.18.31
```

Installs to `~/.opencode/bin/opencode`; add that to `PATH` if the installer didn't.

opencode has no built-in entry for a local llama server, so declare one as an
OpenAI-compatible provider. Put this in `~/.config/opencode/opencode.json` to have it
everywhere, or in a project's `opencode.json` to scope it to that repo — opencode merges
global and project config, project winning.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ortuslocal": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ortus local model",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "unsloth/Qwen3.8-27B-GGUF:Q4_K_M": {},
        "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M": {}
      }
    }
  }
}
```

Three things to get right:

- **The keys under `models` must match `GET /v1/models` exactly**, including the `:Q4_K_M`
  suffix. That string comes from `--alias`, which is why `llmlocal` sets it. A mismatch
  shows up as a 404 from the server, not a config error from opencode.
- **`npm` pulls `@ai-sdk/openai-compatible` on first use**, into
  `~/.config/opencode/node_modules`. First launch after adding the provider is slower.
- **No API key.** `llama serve` doesn't check one and opencode doesn't require one for a
  custom provider. Nothing to run `opencode auth` for.

Only list models you actually serve. The server holds **one** model at a time, so a
second entry is a label for "what `llmlocal` will load next", not a live alternative —
picking one the server isn't currently running 404s until you `llmlocal` over to it.

### Use it

```bash
opencode                                                      # TUI, then /models
opencode run -m "ortuslocal/unsloth/Qwen3.8-27B-GGUF:Q4_K_M" "explain this repo"
```

`-m` is `provider/model`, split on the **first** slash only — so the model id keeps its
own `unsloth/` prefix and the full argument has two slashes. Verified working against
the live server.

---

## 7. Context size and VRAM

`llmlocal` serves at **128K context with a q8_0 KV cache**, raised from 64K on 2026-09-02
so long agent runs stop overflowing. Qwen3.8-27B was trained at 262K, so 128K is well
within the model's range — the limit here is VRAM, not the model.

Measured on the heretic Q4_K_M at 128K: **28.0 GB of 32.6 GB used, 58 tok/s** generation,
with ~5.5 GB taken by other apps.

**Do not raise `-c` without checking headroom first.** Overflow does not error — CUDA
falls back to system memory and throughput collapses silently, measured at 56 tok/s → 0.24.
A 200x slowdown with no log line is the failure mode to watch for.

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

On a smaller card, lower the context at the call site rather than editing the script:

```bash
llmlocal 1 -c 32768        # pass-through overrides the built-in -c
```

---

## 8. Optional integrations

### Claude Code sandboxes

`llama-sandbox-allow` (also in `~/.local/bin`) adds the server's host:port to
`sandbox.network.allowedDomains` in `~/.claude/settings.json`, which Claude Code merges
into every project's sandbox allowlist. Takes effect on the next session.

```bash
llama-sandbox-allow            # allows $(hostname):8080
llama-sandbox-allow --list
```

Sandboxed clients then use `http://$(hostname):8080/v1`, not `127.0.0.1` — see the
`NO_PROXY` note in §5.

### ortus

`ortus grind` drives the local model through opencode. A project `.ortusrc` pins it in a
`[local]` table, which must stay last in the file since TOML puts every following key
inside it:

```toml
[local]
base_url = "http://127.0.0.1:8080/v1"
model = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M"
```

`model` must be an id `GET {base_url}/models` reports — the same string as the
opencode `models` key.

---

## 9. Troubleshooting

**Model answers with garbage (`////`, repeated punctuation).** A corrupt GGUF from a torn
download, not a bad quant. Re-verify and repair:

```bash
llmlocal --download <model>     # re-hashes every cached GGUF for the repo
```

**Port already in use.** `llmlocal` kills the holder itself. To do it by hand:

```bash
ss -tlnp 'sport = :8080'
kill-port 8080
```

**`[download …]` for a model you know you have.** `is_cached` matches `llama serve
--cache-list` exactly. A bare repo with no `:quant` only matches `:q4_k_m`, `:latest`, or
the bare name — so spell the quant out in `models` entries.

**Throughput fell off a cliff.** VRAM overflow into system memory. Check `nvidia-smi`,
lower `-c`. See §7.

**Orphaned partial downloads.** A failed fetch leaves `.downloadInProgress` blobs that
`llama serve --cache-list` won't show but that still consume disk. There are currently
**14 GB** of these from an abandoned `unsloth/GLM-5.3-Flash-GGUF` fetch on 2026-09-04.
Check before you delete — a genuinely in-flight download looks the same:

```bash
du -sh ~/.cache/huggingface/hub/models--*
find ~/.cache/huggingface/hub -name '*.downloadInProgress' -printf '%TY-%Tm-%Td  %10sB  %p\n'
# then, for a fetch you're sure is dead:
rm -rf ~/.cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-GGUF
```

**Non-interactive run started a server you didn't want.** The no-TTY default, §5. Kill it
and pass an explicit selection next time.

---

## 10. Quick reference

```bash
llmlocal                     # menu, Enter = last used
llmlocal 1                   # unsloth/Qwen3.8-27B-GGUF:Q4_K_M
llmlocal 2                   # heretic abliterated + vision
llmlocal --list              # menu with cache status, no server
llmlocal --download M        # fetch + hash-verify; running server untouched
llmlocal --dry-run 2         # show the commands
llmlocal 2 --port 9931       # model first, then pass-through flags
llmlocal --help              # the script's header comment

llama serve --cache-list     # what's cached
llama update                 # upgrade llama.cpp in place
llama cli -hf <model>        # terminal chat, no server

opencode run -m "ortuslocal/<model-id>" "prompt"
llama-sandbox-allow --list   # Claude Code sandbox allowlist
```

| Path | Holds |
|---|---|
| `~/.local/bin/{llama,llmlocal,llama-sandbox-allow}` | binaries and scripts |
| `~/.config/llmlocal/{last,models}` | default pick, extra menu entries |
| `~/.cache/huggingface/hub/` | GGUF blobs, named by sha256 |
| `~/.config/opencode/opencode.json` | global opencode provider config |
| `<project>/opencode.json`, `<project>/.ortusrc` | per-project model pins |
