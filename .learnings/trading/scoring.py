#!/usr/bin/env python3
"""Moovon Fund — 4-Layer Combined Scoring Engine.
News sentiment + regime scoring → combined with LLM → dynamic veto threshold.
Never blocks all trades: worst-case threshold is capped at 0.50."""
import os, re, json, time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv(dotenv_path=Path.home() / ".openclaw/workspace/moovon/.env")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
CACHE_DIR = Path.home() / ".openclaw/workspace/.learnings/trading"
NEWS_CACHE = CACHE_DIR / "news_cache.json"
REGIME_CACHE = CACHE_DIR / "regime_cache.json"

# ── News Sentiment (30% weight) ─────────────────────────────
def fetch_news_sentiment():
    """Scrape headlines → score 0-1 using LLM summary."""
    # Use cache for 30 minutes
    if NEWS_CACHE.exists():
        data = json.loads(NEWS_CACHE.read_text())
        age = time.time() - data.get("ts", 0)
        if age < 1800:
            return data
    
    headlines = []
    # Try multiple free sources
    try:
        with httpx.Client(timeout=8) as c:
            # CoinDesk RSS
            r = c.get("https://www.coindesk.com/arc/outboundfeeds/v2/headlines/?outputType=xml&limit=5",
                     headers={"User-Agent": "Mozilla/5.0"})
            for m in re.finditer(r"<title>(?:\s*<!\[CDATA\[)?([^<]+)", r.text):
                headlines.append(m.group(1).strip())
    except:
        pass
    
    try:
        with httpx.Client(timeout=8) as c:
            r = c.get("https://cryptopanic.com/api/v1/posts/?auth_token=&public=true&limit=5",
                     headers={"User-Agent": "Mozilla/5.0"})
            for item in r.json().get("results", [])[:5]:
                headlines.append(item.get("title", ""))
    except:
        pass
    
    # Default if no headlines fetched
    if not headlines:
        headlines = ["Crypto market data unavailable — using neutral sentiment"]
    
    # LLM summary
    score, summary = _llm_score_news(headlines)
    
    result = {"ts": time.time(), "score": score, "summary": summary, "headlines": headlines[:5]}
    NEWS_CACHE.write_text(json.dumps(result))
    return result

def _llm_score_news(headlines):
    """Ask DeepSeek to score news sentiment."""
    if not LLM_API_KEY:
        return 0.5, "No LLM key — neutral"
    
    prompt = f"""You are a crypto market analyst. Score the following headlines for overall market sentiment.
Return ONLY a JSON with:
- "score": float 0.0 (extreme fear/bearish) to 1.0 (extreme greed/bullish)
- "summary": 1 sentence in English summarizing the market mood

Headlines:
{chr(10).join(f"- {h}" for h in headlines[:5])}"""
    
    try:
        provider_urls = {
            "deepseek": "https://api.deepseek.com/v1/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions",
        }
        url = provider_urls.get(LLM_PROVIDER, provider_urls["deepseek"])
        
        with httpx.Client(timeout=12) as c:
            r = c.post(url, json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100
            }, headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            })
            text = r.json()["choices"][0]["message"]["content"].strip()
            # Extract JSON
            match = re.search(r'\{[^}]+\}', text)
            if match:
                data = json.loads(match.group())
                return float(data.get("score", 0.5)), data.get("summary", "Neutral")
    except:
        pass
    return 0.5, "Neutral (fallback)"

# ── Regime Detection (30% weight) ────────────────────────────
def detect_regime():
    """Detect market regime from BTC price action → score 0-1."""
    if REGIME_CACHE.exists():
        data = json.loads(REGIME_CACHE.read_text())
        age = time.time() - data.get("ts", 0)
        if age < 900:  # 15 min cache
            return data
    
    try:
        with httpx.Client(timeout=10) as c:
            # BTC daily klines for macro regime
            r = c.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "BTCUSDT", "interval": "1d", "limit": 30})
            klines = r.json()
            if len(klines) < 20:
                return _neutral_regime()
            
            closes = [float(k[4]) for k in klines]
            sma_20 = sum(closes[-20:]) / 20
            sma_7 = sum(closes[-7:]) / 7
            current = closes[-1]
            
            # Determine trend direction
            price_vs_sma20 = (current / sma_20 - 1) * 100  # % above/below 20-day MA
            short_vs_long = (sma_7 / sma_20 - 1) * 100  # MA crossover
            
            # Volatility (ATR-like)
            ranges = [abs(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
            avg_range = sum(ranges[-14:]) / 14 if len(ranges) >= 14 else sum(ranges) / len(ranges)
            
            # Regime classification
            if short_vs_long > 1.5 and price_vs_sma20 > 2:
                regime = "bullish"
                score = min(1.0, 0.5 + short_vs_long / 10)
                detail = f"BTC +{price_vs_sma20:.1f}% vs SMA20, SMA7 > SMA20 by {short_vs_long:.1f}%"
            elif short_vs_long < -1.5 and price_vs_sma20 < -2:
                regime = "bearish"
                score = max(0.1, 0.5 + short_vs_long / 10)
                detail = f"BTC {price_vs_sma20:.1f}% vs SMA20, SMA7 < SMA20 by {abs(short_vs_long):.1f}%"
            else:
                regime = "sideways"
                score = 0.5
                detail = f"BTC {price_vs_sma20:+.1f}% vs SMA20, vol={avg_range:.1f}%"
            
            result = {
                "ts": time.time(),
                "regime": regime,
                "score": round(score, 2),
                "price_vs_sma20": round(price_vs_sma20, 1),
                "detail": detail,
                "avg_volatility": round(avg_range, 1)
            }
            REGIME_CACHE.write_text(json.dumps(result))
            return result
    except:
        return _neutral_regime()

def _neutral_regime():
    return {"ts": time.time(), "regime": "sideways", "score": 0.5,
            "price_vs_sma20": 0, "detail": "Unable to determine regime", "avg_volatility": 2.0}

# ── Combined Scoring Engine ─────────────────────────────────
def combined_score(llm_sentiment, news_data=None, regime_data=None):
    """
    Calculate combined trade score with 3 weights + anti-starvation.
    
    Weights:
      - LLM sentiment (existing): 0.40
      - News sentiment: 0.30
      - Regime score: 0.30
    
    Dynamic threshold:
      - Bullish regime: 0.25 (aggressive)
      - Sideways: 0.35 (normal)
      - Bearish: 0.45 (defensive, NOT blocking all)
      - CAP at 0.50 — worst case still passes with strong LLM+News
    """
    if news_data is None:
        news_data = fetch_news_sentiment()
    if regime_data is None:
        regime_data = detect_regime()
    
    news_score = news_data.get("score", 0.5)
    regime_score = regime_data.get("score", 0.5)
    
    # Weighted combined
    combined = (llm_sentiment * 0.40) + (news_score * 0.30) + (regime_score * 0.30)
    combined = round(combined, 3)
    
    # Dynamic threshold — adapts to regime but CAPS at 0.50
    regime = regime_data.get("regime", "sideways")
    if regime == "bullish":
        threshold = 0.25
    elif regime == "bearish":
        threshold = 0.45
    else:
        threshold = 0.35
    
    # Anti-starvation: if LLM alone is very confident (>0.70), lower threshold
    if llm_sentiment > 0.70:
        threshold = min(threshold, 0.30)
    
    # CAP: threshold never exceeds 0.50
    threshold = min(threshold, 0.50)
    
    passed = combined >= threshold
    reason = (
        f"LLM={llm_sentiment:.2f}×0.40 + News={news_score:.2f}×0.30 + Regime={regime_score:.2f}×0.30 "
        f"= {combined:.3f} vs threshold={threshold:.2f} ({regime})"
    )
    
    return {
        "combined": combined,
        "threshold": threshold,
        "passed": passed,
        "regime": regime,
        "news_score": news_score,
        "news_summary": news_data.get("summary", ""),
        "regime_detail": regime_data.get("detail", ""),
        "reason": reason
    }

# ── For external use ──
def get_scoring_context():
    """Return pre-fetched news + regime for the trader to use."""
    return {
        "news": fetch_news_sentiment(),
        "regime": detect_regime()
    }

if __name__ == "__main__":
    # Test
    news = fetch_news_sentiment()
    regime = detect_regime()
    print(f"📰 News: {news['score']} — {news.get('summary','?')}")
    print(f"📊 Regime: {regime['regime']} ({regime['score']}) — {regime['detail']}")
    
    for test_llm in [0.00, 0.30, 0.55, 0.80]:
        result = combined_score(test_llm, news, regime)
        emoji = "✅" if result["passed"] else "🛑"
        print(f"{emoji} LLM={test_llm:.2f}: {result['reason']}")
