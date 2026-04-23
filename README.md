# YandexToLiteLLM

OpenAI-compatible wrapper for [Yandex Embeddings API](https://yandex.cloud/en/docs/foundation-models/embeddings/api-ref/), packaged with a [LiteLLM](https://github.com/BerriAI/litellm) proxy so any OpenAI-compatible client can use Yandex embeddings.

## Architecture

```
Client  →  LiteLLM proxy (:4000)  →  Yandex-embedding-wrapper (:8000)  →  Yandex Cloud API
```

The wrapper exposes a minimal OpenAI `/v1/embeddings` endpoint.  
LiteLLM connects to it using the `openai/` provider prefix and forwards your Yandex API key as `Authorization: Bearer <key>`.

## Quick start

### 1. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your Yandex API key, folder ID, and a master key for LiteLLM
```

| Variable | Description |
|---|---|
| `YANDEX_API_KEY` | Yandex Cloud IAM API key (starts with `AQVN…`) |
| `YANDEX_FOLDER_ID` | Yandex Cloud folder ID |
| `LITELLM_MASTER_KEY` | Secret key used to authenticate calls to the LiteLLM proxy |

### 2. Start services

```bash
docker compose up --build -d
```

Services:
- **LiteLLM proxy** — `http://localhost:4000`
- **Yandex embedding wrapper** — `http://localhost:8001` (direct access, optional)

### 3. Call the embedding API

```bash
curl http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-yandex", "input": "Hello, world!"}'
```

Available model names (defined in `litellm_config.yaml`):

| LiteLLM model name | Yandex model | Use for |
|---|---|---|
| `text-embedding-yandex` | `text-search-doc/latest` | Embedding documents |
| `text-embedding-yandex-query` | `text-search-query/latest` | Embedding search queries |

### 4. Use with LiteLLM Python SDK

```python
import litellm

response = litellm.embedding(
    model="text-embedding-yandex",
    input=["Hello, world!"],
    api_base="http://localhost:4000",
    api_key="sk-my-secret-key",
)
print(response.data[0].embedding)
```

## Adding more Yandex models

Edit `litellm_config.yaml` and add a new entry under `model_list`:

```yaml
- model_name: my-custom-alias
  litellm_params:
    model: openai/text-search-doc/latest   # model name forwarded to wrapper
    api_base: http://yandex-embedding-wrapper:8000
    api_key: os.environ/YANDEX_API_KEY
```

## How it works

1. **LiteLLM** receives an embedding request for `text-embedding-yandex`.
2. It looks up the `litellm_config.yaml` and routes the request to the wrapper using the `openai/` provider (plain HTTP pass-through).
3. The **wrapper** (`app/main.py`) translates the OpenAI-style request into a Yandex API call:
   - Builds `modelUri = emb://<YANDEX_FOLDER_ID>/<model>`
   - Calls `https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding` for each input text (concurrently).
4. The wrapper returns an OpenAI-compatible JSON response, which LiteLLM forwards to the caller.
