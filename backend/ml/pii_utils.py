#!/usr/bin/env python3
import re

# --- PII regexes (conservative) ---
EMAIL_RE   = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE   = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
IPV4_RE    = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSN_RE     = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
HANDLE_RE  = re.compile(r"(^|(?<=\s))@[A-Za-z0-9_./-]+")

# Redaction token
REDACT = "[REDACTED]"

def redact_text(s: str):
    """Return (masked_text:str, masked:bool)."""
    if not s:
        return s, False
    orig = s
    s = EMAIL_RE.sub(REDACT, s)
    s = PHONE_RE.sub(REDACT, s)
    s = IPV4_RE.sub(REDACT, s)
    s = SSN_RE.sub(REDACT, s)
    s = HANDLE_RE.sub(" ", s)
    masked = (s != orig)
    return s, masked

# --- NSFW / toxicity heuristics (lightweight, toggle-able) ---
# Keep list short & general; you can extend for your demo topics.
NSFW_BADWORDS = {
    "fuck","shit","asshole","bitch","bastard","slur","porn","xxx","nsfw",
    "rape","kill yourself","go die","nigger","faggot"  # include a few strong slurs for filtering
}
def is_nsfw(s: str):
    if not s: 
        return False
    low = s.lower()
    return any(w in low for w in NSFW_BADWORDS)
