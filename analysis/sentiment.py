"""News sentiment via VADER."""
from __future__ import annotations
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()


def compute(news: list) -> dict:
    if not news:
        return {"score": 50.0, "signals": [], "headlines": []}
    scores = []
    headlines = []
    for item in news[:5]:
        title = item.get("title") or (item.get("content") or {}).get("title", "")
        if not title:
            continue
        s = _sia.polarity_scores(title)["compound"]
        scores.append(s)
        headlines.append({"title": title, "sentiment": s,
                          "link": item.get("link") or (item.get("content") or {}).get("canonicalUrl", {}).get("url", "")})

    if not scores:
        return {"score": 50.0, "signals": [], "headlines": []}

    avg = float(np.mean(scores))
    # map [-1, 1] -> [0, 100]
    score = float(np.clip(50 + avg * 50, 0, 100))
    signals = []
    if avg > 0.3:
        signals.append("Positive news flow")
    elif avg < -0.3:
        signals.append("Negative news flow")
    return {"score": score, "signals": signals, "avg_sentiment": avg, "headlines": headlines}
