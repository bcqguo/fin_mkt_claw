import json
import os
import re
from datetime import datetime

import feedparser
import pandas as pd
import yfinance as yf

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def get_prices(tickers=None):
    if tickers is None:
        tickers = ["SPY", "QQQ", "DIA", "NVDA", "SNOW", "BABA"]

    data = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="2d")
        hist_dict = hist.to_dict()

        # Convert Timestamp keys to strings so json.dump works
        for col in hist_dict:
            hist_dict[col] = {str(k): v for k, v in hist_dict[col].items()}

        data[t] = hist_dict

    return data


def get_news(feeds=None, max_articles=20):
    if feeds is None:
        feeds = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]

    articles = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.title,
                "summary": entry.summary,
                "link": entry.link,
            })
    return articles


def compute_returns(prices):
    results = {}
    for t, hist in prices.items():
        df = pd.DataFrame(hist)
        if "Close" in df:
            returns = df["Close"].pct_change().iloc[-1]
            results[t] = float(returns)
    return results


def generate_signals(features, news):
    signals = []

    returns = features.get("returns", {})

    # --- Rule 1: Momentum ---
    for ticker, r in returns.items():
        if r > 0.02:
            signals.append({
                "type": "momentum_bullish",
                "ticker": ticker,
                "value": r,
                "strength": "high",
            })
        elif r < -0.02:
            signals.append({
                "type": "momentum_bearish",
                "ticker": ticker,
                "value": r,
                "strength": "high",
            })

    # --- Rule 2: Market Breadth ---
    pos = sum(1 for r in returns.values() if r > 0)
    neg = sum(1 for r in returns.values() if r < 0)

    if pos > neg * 2:
        signals.append({"type": "broad_bullish", "strength": "medium"})
    elif neg > pos * 2:
        signals.append({"type": "broad_bearish", "strength": "medium"})

    # --- Rule 3: News Sentiment (simple heuristic) ---
    negative_keywords = ["inflation", "war", "rate hike", "selloff", "recession"]
    positive_keywords = ["growth", "beat", "rally", "upgrade"]

    neg_score, pos_score = 0, 0
    for article in news:
        text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        neg_score += sum(k in text for k in negative_keywords)
        pos_score += sum(k in text for k in positive_keywords)

    if neg_score > pos_score * 1.5:
        signals.append({"type": "negative_news_pressure", "strength": "medium"})
    elif pos_score > neg_score * 1.5:
        signals.append({"type": "positive_news_momentum", "strength": "medium"})

    return signals


def generate_alerts(signals):
    alerts = []

    for s in signals:
        if s.get("strength") == "high":
            alerts.append(f"🚨 HIGH: {s}")
        elif s.get("type") in ["broad_bearish", "negative_news_pressure"]:
            alerts.append(f"⚠️ RISK: {s}")

    return alerts


def build_prompt(features, signals, alerts, news):
    return f"""
You are a senior macro hedge fund analyst.

Market returns:
{features.get('returns')}

Detected signals:
{signals}

Alerts:
{alerts}

Key news:
{news}

Output STRICTLY in the following format only. Do not include Detailed Analysis, Conclusion, Recommendations, or Disclaimer sections:

## 🚨 Alerts
- ...

## 📊 Trading Signals
- ...

## Executive Summary
- ...

## Key Drivers
- ...

## Risks
- ...

## Outlook
- ...
"""


def summarize(prompt, model_name=None, device=None):
    if model_name is None:
        #"mistralai/Mistral-7B-Instruct-v0.3"
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") 

    if device is None:
        # Default to CUDA when available, otherwise fall back to CPU.
        device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    hf_token = os.environ.get("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
        token=hf_token,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    headings = [
        "## 🚨 Alerts",
        "## 📊 Trading Signals",
        "## Executive Summary",
        "## Key Drivers",
        "## Risks",
        "## Outlook",
    ]

    def _decode_with_max(max_new_tokens: int) -> str:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()

    base_max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "800"))
    max_new_tokens = base_max_new_tokens

    decoded = _decode_with_max(max_new_tokens)

    # The model sometimes repeats the whole template multiple times.
    # Also, it sometimes emits headings on the same line; normalize formatting first.
    decoded = re.sub(r"[ \t]*## ", "\n## ", decoded).strip()

    # If the template is repeated, keep only from the first Alerts block.
    start = decoded.find(headings[0])
    if start != -1:
        decoded = decoded[start:]

    outlook_present = "## Outlook" in decoded
    allow_retry = os.environ.get("RETRY_ON_MISSING_OUTLOOK", "true").lower() in ["1", "true", "yes"]
    if (not outlook_present) and allow_retry:
        # If max_new_tokens was too small and the model never reached Outlook, try once more.
        decoded = _decode_with_max(min(base_max_new_tokens * 2, 1600))
        decoded = re.sub(r"[ \t]*## ", "\n## ", decoded).strip()
        start = decoded.find(headings[0])
        if start != -1:
            decoded = decoded[start:]

    # If Outlook is present and the model starts a second template after it, cut at the next Alerts.
    if "## Outlook" in decoded:
        outlook_start = decoded.find("## Outlook")
        next_alerts = decoded.find(headings[0], outlook_start + len("## Outlook"))
        if next_alerts != -1:
            decoded = decoded[:next_alerts].rstrip()

    # Final formatting cleanup:
    # - Ensure all headings start on their own line
    # - Ensure exactly one blank line before every "## ..." heading
    # collapse 2+ blank lines -> 1 blank line between sections
    decoded = re.sub(r"\n{3,}## ", "\n\n## ", decoded)
    # add a missing blank line when there's only a single newline before a heading
    decoded = re.sub(r"(?<!\n)\n## ", "\n\n## ", decoded)
    decoded = decoded.strip()

    return decoded.strip()


def send_report(
    report_text,
    from_email=None,
    to_emails=None,
    api_key=None,
    subject="Market Summary",
):
    if from_email is None:
        from_email = os.environ.get("FROM_EMAIL")
    if to_emails is None:
        to_emails = os.environ.get("TO_EMAIL")
    if api_key is None:
        api_key = os.environ.get("SENDGRID_API")

    if not api_key:
        raise RuntimeError("SENDGRID_API not set")
    if not from_email or not to_emails:
        raise RuntimeError("FROM_EMAIL and TO_EMAIL must be set")

    if isinstance(to_emails, str):
        to_emails = [e.strip() for e in to_emails.split(",") if e.strip()]

    message = Mail(
        from_email=from_email,
        to_emails=to_emails,
        subject=subject,
        html_content=f"<pre>{report_text}</pre>",
    )

    sg = SendGridAPIClient(api_key=api_key)
    return sg.send(message)


def dump_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_all(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    prices = get_prices()
    news = get_news()

    dump_json(f"{output_dir}/raw.json", {"timestamp": str(datetime.now()), "prices": prices, "news": news})

    features = {"returns": compute_returns(prices)}
    dump_json(f"{output_dir}/features.json", features)

    signals = generate_signals(features, news)
    dump_json(f"{output_dir}/signals.json", signals)

    alerts = generate_alerts(signals)
    dump_json(f"{output_dir}/alerts.json", alerts)

    prompt = build_prompt(features, signals, alerts, news)
    summary = summarize(prompt)
    open(f"{output_dir}/report.txt", "w").write(summary)

    return summary
