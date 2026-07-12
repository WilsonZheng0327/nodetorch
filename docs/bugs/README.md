# Bug log

Write-ups of notable bugs — the ones where the *root cause* was more interesting
than the fix, and worth remembering so we don't reintroduce the assumption that
caused them. Think short postmortem, not an issue tracker: symptom, root cause,
why it hid, the fix, and the lesson.

Write one here when a bug (a) was silent or misleading, (b) exposed a wrong
assumption baked into the design, or (c) is likely to recur in a different guise.
Skip the routine stuff — a plain typo fix doesn't need a page.

## Entries

- [`seqpool-last-reads-padding.md`](seqpool-last-reads-padding.md) — `sequence_pool`
  `"last"` pooled over padding on short text, silently pinning a binary classifier
  at 50%; fixed with an optional padding mask on the pool node.
