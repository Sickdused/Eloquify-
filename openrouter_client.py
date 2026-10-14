import requests

FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning:free"


def post_with_fallback(url, *, fallback_model=FALLBACK_MODEL, **kwargs):
    """Try the configured model, then a fallback on temporary failures."""
    original_payload = kwargs.get("json")

    if not isinstance(original_payload, dict) or not original_payload.get("model"):
        raise ValueError("OpenRouter request must include a model in its JSON payload.")

    primary_model = original_payload["model"]
    models = list(dict.fromkeys([primary_model, fallback_model]))
    last_response = None
    last_exception = None

    for index, model in enumerate(models):
        payload = dict(original_payload)
        payload["model"] = model
        request_kwargs = dict(kwargs)
        request_kwargs["json"] = payload

        try:
            response = requests.post(url, **request_kwargs)
        except requests.RequestException as exc:
            last_exception = exc
            if index < len(models) - 1:
                continue
            raise

        if response.ok:
            if model != primary_model:
                print(f"OpenRouter fallback used: {model}")
            return response

        last_response = response

        # Retry with the fallback for rate limits, server errors,
        # or a model that is unavailable/not found.
        if response.status_code not in (404, 429, 500, 502, 503, 504):
            response.raise_for_status()

    if last_response is not None:
        last_response.raise_for_status()

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("All OpenRouter models failed.")
