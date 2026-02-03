from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_EMAIL_RE = re.compile(r"\b[\w.+'-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s]+\b|\bwww\.[^\s]+\b")
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,30}\b")
_WS_RE = re.compile(r"\s+")


@dataclass
class PrivacyConfig:
    mask_emails: bool = True
    mask_phones: bool = True
    mask_urls: bool = True
    mask_handles: bool = True
    normalize_whitespace: bool = True
    max_chars: int = 2048


def privacy_preprocess(text: str, cfg: PrivacyConfig) -> str:
    if not isinstance(text, str):
        text = str(text)

    s = text.strip()
    if cfg.mask_emails:
        s = _EMAIL_RE.sub("[EMAIL]", s)
    if cfg.mask_urls:
        s = _URL_RE.sub("[URL]", s)
    if cfg.mask_handles:
        s = _HANDLE_RE.sub("[HANDLE]", s)
    if cfg.mask_phones:
        s = _PHONE_RE.sub("[PHONE]", s)
    if cfg.normalize_whitespace:
        s = _WS_RE.sub(" ", s).strip()

    if cfg.max_chars and len(s) > int(cfg.max_chars):
        s = s[: int(cfg.max_chars)]
    return s


def privacy_cfg_from_dict(d: dict[str, Any]) -> PrivacyConfig:
    return PrivacyConfig(
        mask_emails=bool(d.get("mask_emails", True)),
        mask_phones=bool(d.get("mask_phones", True)),
        mask_urls=bool(d.get("mask_urls", True)),
        mask_handles=bool(d.get("mask_handles", True)),
        normalize_whitespace=bool(d.get("normalize_whitespace", True)),
        max_chars=int(d.get("max_chars", 2048)),
    )
