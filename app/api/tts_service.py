import aiohttp
from app.core.config import settings

async def synthesize_supertone_url(
    *, text: str, voice_id: str, style: str,
    fmt: str = "mp3", speed: float = 1.0, pitch: int = 0,
) -> str:
    """
    파일 저장 없이 1회성: 수퍼톤 TTS 호출 후 audioUrl 문자열만 반환.
    케이스 분기 없음. (audioUrl 없으면 에러)
    """
    emotion_key = getattr(settings, "SUPERTONE_EMOTION_PARAM_NAME", "style")
    payload = {
        "voiceId": voice_id,
        emotion_key: style,
        "text": text,
        "format": fmt,
        "speed": speed,
        "pitch": pitch,
    }
    headers = {
        "Authorization": f"Bearer {settings.SUPERTONE_API_KEY}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.post(settings.SUPERTONE_TTS_ENDPOINT, json=payload, headers=headers) as r:
            if r.status != 200:
                raise RuntimeError(f"TTS {r.status}: {await r.text()}")
            data = await r.json()

    url = data.get("audioUrl")
    if not url:
        raise RuntimeError("TTS response missing audioUrl")
    return url