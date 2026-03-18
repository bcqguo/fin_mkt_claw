import os

from utils import load_json, run_all, send_report


def main():
    summary = run_all(output_dir="data")

    alerts = load_json("data/alerts.json")
    subject = "🚨 MARKET ALERT - ACTION REQUIRED" if len(alerts) > 0 else "📊 Daily Market Summary"

    send_report(summary, subject=subject)


if __name__ == "__main__":
    main()
