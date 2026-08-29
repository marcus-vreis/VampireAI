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


class ModelUnavailableError(RuntimeError):
    """O servidor responde mas o modelo não roda. Repetir não adianta."""

# Guardar a resposta é o que permite auditar acurácia depois. Antes só gravávamos
# o tamanho, então "a leitura piorou" era impressão e não número.
_LOG_TEXT_CHARS = 2000

# Sonda de saúde depois de erro de transporte. Observado numa execução real: o
# runner do Ollama morreu e cada chamada seguinte queimava 3 tentativas de ~42s
# antes de desistir — mais de 2 minutos por passo, com o agente parecendo
# travado. A sonda é texto puro e curta: se ela responde, o problema era daquela
# chamada; se não responde, insistir só desperdiça tempo.
_HEALTH_TIMEOUT_S = 10.0
_RUNNER_DEAD_MARKERS = ("unexpectedly stopped", "GGML_ASSERT", "failed to load model")


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
    model: str | None = None,
) -> dict[str, Any] | str:
    """Chama o modelo com retry+backoff. Aceita path OU PIL Image (`image=`).

    O modelo é escolhido pela presença de imagem: com imagem vai pro `VLM_MODEL`,
    sem imagem vai pro `TEXT_MODEL`. `model=` força um específico — usado pelo
    bench, que compara candidatos. Retorna dict validado se houver schema, senão
    a string crua.
    """
    call_id = uuid.uuid4().hex[:12]
    client = _get_client()
    messages = _build_messages(prompt, image_path, image=image)
    response_format = {"type": "json_object"} if schema is not None else None
    # Roteamento por natureza da chamada: quem manda imagem precisa do VLM, quem
    # não manda está pedindo raciocínio e vai pro modelo de texto (ver LLMConfig).
    model = model or LLM.pick(image_path is not None or image is not None)

    last_error: Exception | None = None
    for attempt in range(1, LLM.max_retries + 1):
        started = time.monotonic()
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
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
                    "model": model,
                    "attempt": attempt,
                    "elapsed_s": round(elapsed, 3),
                    "image": image_path,
                    "prompt_chars": len(prompt),
                    "schema": schema.__name__ if schema else None,
                    "ok": True,
                    "raw_chars": len(content),
                    "response": result if schema is not None else content[:_LOG_TEXT_CHARS],
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
                    "model": model,
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
            if _looks_like_dead_runner(e) and not _runner_alive():
                raise ModelUnavailableError(
                    f"o modelo '{model}' não está rodando. O servidor Ollama "
                    "responde, mas o runner morreu — reinicie `ollama serve` e "
                    "confira `ollama ps`."
                ) from e
            _log_call(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "call_id": call_id,
                    "model": model,
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


def _looks_like_dead_runner(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _RUNNER_DEAD_MARKERS)


def _runner_alive() -> bool:
    """Uma chamada de texto curta e barata. False se o modelo não carrega."""
    try:
        probe = OpenAI(
            base_url=LLM.base_url, api_key=LLM.api_key, timeout=_HEALTH_TIMEOUT_S
        )
        probe.chat.completions.create(
            model=LLM.text_model, messages=[{"role": "user", "content": "ok"}]
        )
    except Exception:  # noqa: BLE001 - a sonda existe pra devolver um booleano
        return False
    return True


def _ping() -> int:
    """Testa os dois modelos. Se forem o mesmo, testa uma vez só."""
    models = {LLM.text_model: "texto", LLM.vision_model: "visão"}
    logger.info("Servidor: {}", LLM.base_url)
    failed = False
    for name, role in models.items():
        logger.info("Pingando {} ({})...", name, role)
        try:
            reply = _ask_model(name, "Responda em uma frase curta: você está funcionando?")
        except Exception as e:  # noqa: BLE001 - o ping existe pra reportar falha
            logger.error("  falhou: {}", e)
            failed = True
            continue
        print(f"[{name}] {reply}")
    if LLM.text_model == LLM.vision_model:
        logger.warning(
            "TEXT_MODEL não está configurado — decisões de combate e escolha vão "
            "pro VLM. São chamadas de texto puro; um modelo de raciocínio dedicado "
            "tende a jogar melhor. Ver .env.example."
        )
    return 1 if failed else 0


def _ask_model(model: str, prompt: str) -> str:
    resp = _get_client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content or ""


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
