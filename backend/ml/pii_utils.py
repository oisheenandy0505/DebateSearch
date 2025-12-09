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
