"""Wrapper de chamadas ao Ollama via SDK openai, com retry, backoff e log JSONL."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ValidationError

from src.config import LLM, LLM_LOG_FILE, PATHS

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM.base_url, api_key=LLM.api_key, timeout=LLM.timeout_s)
    return _client


def _encode_pil(img: Image.Image) -> str:
    img = img.convert("RGB")
    if LLM.image_max_side and LLM.image_max_side > 0:
        longest = max(img.size)
        if longest > LLM.image_max_side:
            ratio = LLM.image_max_side / longest
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _encode_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    with Image.open(path) as img:
        return _encode_pil(img)


def _log_call(record: dict[str, Any]) -> None:
    PATHS.logs.mkdir(parents=True, exist_ok=True)
    with LLM_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_messages(
    prompt: str, image_path: str | None, image: Image.Image | None = None
) -> list[dict[str, Any]]:
    if image_path is None and image is None:
        return [{"role": "user", "content": prompt}]

    img_b64 = _encode_pil(image) if image is not None else _encode_image(image_path)  # type: ignore[arg-type]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ]


def _parse_structured(content: str, schema: type[BaseModel]) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Saída do VLM não é JSON válido: {e}") from e
    return schema.model_validate(data).model_dump()


def ask_vlm(
    image_path: str | None,
    prompt: str,
    schema: type[BaseModel] | None = None,
    image: Image.Image | None = None,
) -> dict[str, Any] | str:
    """Chama o VLM com retry+backoff. Aceita path OU PIL Image (image=). Retorna dict validado se schema, senão str."""
    call_id = uuid.uuid4().hex[:12]
    client = _get_client()
    messages = _build_messages(prompt, image_path, image=image)
    response_format = {"type": "json_object"} if schema is not None else None

    last_error: Exception | None = None
    for attempt in range(1, LLM.max_retries + 1):
        started = time.monotonic()
        try:
            kwargs: dict[str, Any] = {"model": LLM.model, "messages": messages}
            if response_format is not None:
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            elapsed = time.monotonic() - started

            result: dict[str, Any] | str
            if schema is not None:
                result = _parse_structured(content, schema)
            else:
                result = content

            _log_call(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "call_id": call_id,
                    "model": LLM.model,
                    "attempt": attempt,
                    "elapsed_s": round(elapsed, 3),
                    "image": image_path,
                    "prompt_chars": len(prompt),
                    "schema": schema.__name__ if schema else None,
                    "ok": True,
                    "raw_chars": len(content),
                }
            )
            return result

        except (ValueError, ValidationError) as e:
            last_error = e
            elapsed = time.monotonic() - started
            logger.warning(
                "VLM parse falhou (call={} tentativa={}/{}): {}",
                call_id,
                attempt,
                LLM.max_retries,
                e,
            )
            _log_call(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "call_id": call_id,
                    "model": LLM.model,
                    "attempt": attempt,
                    "elapsed_s": round(elapsed, 3),
                    "image": image_path,
                    "prompt_chars": len(prompt),
                    "schema": schema.__name__ if schema else None,
                    "ok": False,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
        except Exception as e:
            last_error = e
            elapsed = time.monotonic() - started
            logger.warning(
                "VLM erro de transporte (call={} tentativa={}/{}): {}",
                call_id,
                attempt,
                LLM.max_retries,
                e,
            )
            _log_call(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "call_id": call_id,
                    "model": LLM.model,
                    "attempt": attempt,
                    "elapsed_s": round(elapsed, 3),
                    "image": image_path,
                    "prompt_chars": len(prompt),
                    "schema": schema.__name__ if schema else None,
                    "ok": False,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )

        if attempt < LLM.max_retries:
            sleep_s = LLM.backoff_base_s ** attempt
            time.sleep(sleep_s)

    raise RuntimeError(
        f"VLM falhou após {LLM.max_retries} tentativas (call={call_id}): {last_error}"
    )


def _ping() -> int:
    logger.info("Pingando {} em {}", LLM.model, LLM.base_url)
    try:
        reply = ask_vlm(None, "Olá, está funcionando? Responda em uma frase curta.")
    except Exception as e:
        logger.error("Ping falhou: {}", e)
        return 1
    print(reply)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper LLM/VLM (Ollama).")
    parser.add_argument("--ping", action="store_true", help="Testa se o servidor responde.")
    args = parser.parse_args()

    if args.ping:
        return _ping()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
