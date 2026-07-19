"""
emotion_prompt.py — 将 V-A 坐标转为 MusicGen 文本描述
=====================================================
不修改模型内部，只通过文本 prompt 控制情感。

用法:
  from emotion_prompt import va_to_prompt
  prompt = va_to_prompt(0.8, 0.2)  # "calm relaxing piano, soft and peaceful"
"""

# V-A 空间 → 文本关键词映射
_V_KEYWORDS = {
    "very_positive": ["joyful", "uplifting", "bright", "happy", "cheerful"],
    "positive": ["pleasant", "warm", "gentle", "sweet", "hopeful"],
    "neutral": ["neutral", "ordinary", "plain", "simple"],
    "negative": ["sad", "melancholic", "somber", "gloomy", "dark"],
    "very_negative": ["depressing", "tragic", "mournful", "hopeless", "grim"],
}

_A_KEYWORDS = {
    "very_high": ["energetic", "intense", "powerful", "aggressive", "wild"],
    "high": ["lively", "fast", "dynamic", "vibrant", "dramatic"],
    "medium": ["moderate", "steady", "balanced", "flowing"],
    "low": ["calm", "gentle", "slow", "relaxed", "peaceful"],
    "very_low": ["still", "quiet", "tranquil", "meditative", "motionless"],
}

_INSTRUMENTS_BY_V = {
    "high": ["piano", "strings", "flute", "glockenspiel", "harp"],
    "medium": ["guitar", "synthesizer", "organ", "marimba"],
    "low": ["cello", "bassoon", "viola", "trombone", "double bass"],
}

_GENRES_BY_VA = {
    (True, True):   ("upbeat pop", "dance", "electronic"),
    (True, False):  ("ambient", "classical", "lo-fi", "new age"),
    (False, True):  ("rock", "metal", "industrial", "soundtrack"),
    (False, False): ("blues", "folk ballad", "dark ambient", "dirge"),
}


def _interpolate(val: float, bins: list[float], labels: list[str]) -> str:
    """在离散区间中插值选择关键词。"""
    import random
    for i, threshold in enumerate(bins):
        if val <= threshold:
            return random.choice(labels[i])
    return random.choice(labels[-1])


def _valence_level(v: float) -> str:
    if v >= 0.80: return "very_positive"
    if v >= 0.60: return "positive"
    if v >= 0.40: return "neutral"
    if v >= 0.20: return "negative"
    return "very_negative"


def _arousal_level(a: float) -> str:
    if a >= 0.80: return "very_high"
    if a >= 0.60: return "high"
    if a >= 0.40: return "medium"
    if a >= 0.20: return "low"
    return "very_low"


def va_to_prompt(valence: float, arousal: float,
                 base_prompt: str = "",
                 detail_level: str = "medium") -> str:
    """将 V-A 坐标转为 MusicGen 文本 prompt。

    Args:
        valence: 0~1, 积极度
        arousal: 0~1, 兴奋度
        base_prompt: 基础描述（如 "piano melody"）
        detail_level: "simple" / "medium" / "detailed"

    返回: 文本 prompt
    """
    import random
    random.seed(hash((valence, arousal, base_prompt)))

    v_level = _valence_level(valence)
    a_level = _arousal_level(arousal)

    v_word = random.choice(_V_KEYWORDS[v_level])
    a_word = random.choice(_A_KEYWORDS[a_level])

    # 情感词
    mood_words = [v_word, a_word]

    # 体裁
    va_key = (valence > 0.5, arousal > 0.5)
    genre = random.choice(_GENRES_BY_VA[va_key])
    mood_words.append(genre)

    if detail_level == "simple":
        # "happy energetic music"
        words = mood_words[:2]
        if base_prompt:
            words.append(base_prompt)
        return " ".join(words) + " music"

    # 乐器
    if valence > 0.6:
        inst = random.choice(_INSTRUMENTS_BY_V["high"])
    elif valence < 0.4:
        inst = random.choice(_INSTRUMENTS_BY_V["low"])
    else:
        inst = random.choice(_INSTRUMENTS_BY_V["medium"])

    if detail_level == "medium":
        # "joyful energetic electronic music with piano"
        parts = [f"{v_word} {a_word}", genre, "music"]
        if base_prompt:
            parts.append(f"with {base_prompt}")
        else:
            parts.append(f"with {inst}")
        return ", ".join(parts)

    # detailed
    # "A joyful and energetic electronic piece with bright piano and driving beat"
    template = random.choice([
        f"A {v_word} and {a_word} {genre} piece with {inst}",
        f"{v_word.capitalize()} {a_word} {genre} music featuring {inst}",
        f"A {a_word} {genre} melody with a {v_word} atmosphere and {inst}",
    ])
    if base_prompt:
        template += f", {base_prompt}"
    return template


if __name__ == "__main__":
    # 测试
    test_cases = [
        (0.85, 0.80, "happy excited"),
        (0.80, 0.15, "calm relaxing"),
        (0.20, 0.25, "sad melancholic"),
        (0.15, 0.85, "angry aggressive"),
        (0.50, 0.50, "neutral"),
        (0.70, 0.65, "uplifting"),
        (0.20, 0.45, "dark mysterious"),
        (0.80, 0.40, "romantic"),
    ]
    print("V-A → 文本 prompt 测试:\n")
    for v, a, desc in test_cases:
        p = va_to_prompt(v, a, detail_level="medium")
        print(f"  {desc:>15} (V={v:.2f},A={a:.2f}) → {p}")
