import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name):
    return (PROMPTS_DIR / f"{name}.txt").read_text()


def is_openai(model):
    return model.startswith("gpt-") and ":" not in model


def post(url, payload, headers=None, attempts=5):
    """POST JSON and return the parsed reply, retrying rate limits and 5xx."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=600) as handle:
                return json.loads(handle.read())
        except urllib.error.HTTPError as error:
            print(f"API error {error.code}: {error.read().decode()}", flush=True)
            if error.code not in (429, 500, 502, 503, 529) or attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def call_openai(prompt, model, options):
    """One chat completion, plus the token counts the response reports."""
    body = post(
        OPENAI_URL,
        {"model": model, "messages": [{"role": "user", "content": prompt}], **options},
        {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    )
    usage = body.get("usage", {})
    return {
        "response": body["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def call_ollama(prompt, model, options):
    """One completion from a local Ollama model"""
    body = post(
        OLLAMA_URL,
        {"model": model, "prompt": prompt, "stream": False, "options": options},
    )
    used = body["prompt_eval_count"]
    assert used < options["num_ctx"], (
        f"prompt used {used} of num_ctx={options['num_ctx']}; Ollama truncates "
        f"silently, so this response saw only part of the prompt."
    )
    return {"response": body["response"], "prompt_tokens": used}


def generate(prompt, model, cache_dir, num_ctx=16384, temperature=0.0, reasoning_effort="none"):
    """One completion from `model`, cached"""
    openai = is_openai(model)
    options = (
        {"reasoning_effort": reasoning_effort}
        if openai
        else {"num_ctx": num_ctx, "temperature": temperature}
    )

    key = hashlib.sha256(
        json.dumps([model, prompt, options], sort_keys=True).encode()
    ).hexdigest()
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]

    record = (
        call_openai(prompt, model, options)
        if openai
        else call_ollama(prompt, model, options)
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {"model": model, "options": options, "prompt": prompt, **record}, indent=2
        )
    )
    print(record["response"], flush=True)
    return record["response"]
