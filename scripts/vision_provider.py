"""
多供应商视觉识别 — 轻量包装
GLM-4.6V-Flash（免费/默认） + qwen-vl-max（付费兜底）+ openai/doubao
"""
import os, sys, base64, json
from pathlib import Path
from openai import OpenAI

PROVIDERS = {
    "glm": {
        "key_env": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6v-flash",
    },
    "qwen": {
        "key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "doubao": {
        "key_env": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-0-pro-260215",
    },
}

MIME_MAP = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif"}

def vision_analyze(image_path: str, prompt: str, provider: str = "glm") -> str:
    """分析单张图片，返回文字描述"""
    if provider not in PROVIDERS:
        raise ValueError(f"未知供应商: {provider}，可选: {list(PROVIDERS.keys())}")

    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"], "")
    model = os.environ.get(f"{provider.upper()}_MODEL", cfg["model"])

    if not api_key:
        raise ValueError(f"请设置环境变量 {cfg['key_env']}")

    # 编码图片
    ext = Path(image_path).suffix.lower()
    mime = MIME_MAP.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0,
        max_tokens=int(os.environ.get("VISION_MAX_TOKENS", "4096")),
    )
    return resp.choices[0].message.content or ""

def analyze_with_fallback(image_path: str, prompts: list,
                          primary: str = "glm", fallback: str = "qwen") -> dict:
    """
    GLM 优先，异常时回退 qwen。
    prompts: ["通用prompt", "详细prompt（用于回退）"]
    返回 {"text": "...", "provider": "glm|qwen", "retries": 0}
    """
    import time

    # 尝试主供应商
    for attempt in range(2):
        try:
            text = vision_analyze(image_path, prompts[0], primary)
            # 检查输出质量
            if len(text) >= 50:
                return {"text": text, "provider": primary, "retries": attempt}
            # 输出太短，换详细 prompt 重试
            prompts = [prompts[1] if len(prompts) > 1 else prompts[0]]
        except Exception as e:
            if '429' in str(e):
                time.sleep(5)
                continue
            break  # 非限流错误，直接回退

    # 回退供应商
    try:
        text = vision_analyze(image_path, prompts[-1], fallback)
        return {"text": text, "provider": fallback, "retries": 1}
    except Exception as e:
        return {"text": f"[识别失败: {e}]", "provider": "error", "retries": 2}
