# Safety & Ethics Guardrails

- Outputs are **risk signals**, not diagnoses.
- Do not display medical claims, probability of suicide, or treatment recommendations.
- For high-risk signals, only provide **general, non-actionable** supportive guidance and encourage reaching out to trusted people and local professional resources.
- Never provide self-harm instructions.
- All text is privacy-preprocessed (PII masking) before storage/indexing.
- Logs must not store raw user identifiers or raw unmasked content.
- Any “support module” message must remain neutral, supportive, and non-coercive.

Default runtime behavior:
- API returns class labels + calibrated scores.
- API can return a short “reason code” list (feature flags), not sensitive spans.
- For high-risk, API includes a “seek help” generic banner.
