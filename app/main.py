import asyncio
import os
from typing import List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Yandex Embedding OpenAI-compatible wrapper")

YANDEX_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
MAX_CONCURRENT_REQUESTS = 10


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


async def _embed_one(
    client: httpx.AsyncClient,
    text: str,
    model_uri: str,
    api_key: str,
    folder_id: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.post(
            YANDEX_API_URL,
            headers={
                "Authorization": f"Api-Key {api_key}",
                "x-folder-id": folder_id,
                "Content-Type": "application/json",
            },
            json={"modelUri": model_uri, "text": text},
            timeout=30.0,
        )
    if response.status_code != 200:
        try:
            detail = response.json().get("message", "Yandex API returned an error")
        except Exception:
            detail = "Yandex API returned an error"
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


@app.post("/v1/embeddings")
async def create_embeddings(body: EmbeddingRequest, request: Request):
    # Allow API key to be passed via Authorization header (LiteLLM sends it as Bearer)
    auth = request.headers.get("Authorization", "")
    api_key = auth.removeprefix("Bearer ").strip() or YANDEX_API_KEY
    if not api_key:
        raise HTTPException(status_code=401, detail="No Yandex API key provided")

    folder_id = YANDEX_FOLDER_ID
    if not folder_id:
        raise HTTPException(status_code=500, detail="YANDEX_FOLDER_ID is not configured")

    # body.model should be something like "text-search-doc/latest"
    model_uri = f"emb://{folder_id}/{body.model}"

    texts: List[str] = [body.input] if isinstance(body.input, str) else body.input

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient() as client:
        tasks = [
            _embed_one(client, text, model_uri, api_key, folder_id, semaphore)
            for text in texts
        ]
        results = await asyncio.gather(*tasks)

    data = []
    total_tokens = 0
    for i, result in enumerate(results):
        data.append(
            {
                "object": "embedding",
                "embedding": result["embedding"],
                "index": i,
            }
        )
        total_tokens += int(result.get("numTokens", 0))

    return {
        "object": "list",
        "data": data,
        "model": body.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
