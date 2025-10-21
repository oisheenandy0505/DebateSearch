import os
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Config ---
SAVE_DIR = "/Users/oishee/Documents/Fall 2025/INFSCI 2140/Final Project/DebateSearch/data/raw/kaggle"
os.makedirs(SAVE_DIR, exist_ok=True)
OUT_PATH = os.path.join(SAVE_DIR, "reddit_subset.jsonl")

# --- Parameters ---
TARGET_SUBS = {"cryptocurrency", "finance", "economics", "technology", "worldnews", "politics"}
SAMPLE_SIZE = 50000  # adjust up/down as needed

print("📡 Loading dataset from Hugging Face (pushshift Reddit)...")
# You can change to another variant if this one fails, e.g. "reddit-dump/reddit-comments"
ds = load_dataset("fddemarco/pushshift-reddit-comments", split="train", streaming=True)

print("🔍 Filtering and saving subset...")
count_out = 0

with open(OUT_PATH, "w") as f_out:
    for item in tqdm(ds, total=SAMPLE_SIZE * 10):  # just a safety upper bound
        try:
            body = item.get("body", "")
            subreddit = item.get("subreddit", "").lower()
            if subreddit in TARGET_SUBS and len(body) > 40:
                record = {
                    "id": item.get("id"),
                    "subreddit": subreddit,
                    "body": body,
                    "created_utc": item.get("created_utc"),
                    "score": item.get("score", 0)
                }
                f_out.write(json.dumps(record) + "\n")
                count_out += 1
            if count_out >= SAMPLE_SIZE:
                break
        except Exception:
            continue

print(f"✅ Saved {count_out:,} comments to {OUT_PATH}")