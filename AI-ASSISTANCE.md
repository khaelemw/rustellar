# AI assistance log

A record of where generative AI was used in this repository, kept for the JOSS
AI usage disclosure. One row per session, written at the time.

All design and architecture decisions are the author's. AI-written changes are
small, made under the author's direction, and reviewed by the author before
they are committed. From 11 Nov 2026 onward the only tool used is Claude Opus
5.5 through Claude Code.

| Date | Tool and model | Where | What the AI did | What the author did |
|---|---|---|---|---|
| 2026-11-10 | Claude Sonnet 5.5 (`claude-sonnet-5-5`) via Claude Code 2.1.321 | planning notes (not in this repository); initial empty workspace | Drafted a planning guide and the empty two-crate workspace layout | Chose the project scope and structure; reduced the skeleton to a bare first commit |
| 2026-11-10 to 2026-11-11 | Claude Opus 5.5 (`claude-opus-5-5`) via Claude Code 2.1.321 | `rustellar-core`, CI | Code review and explanations; small requested edits (some tests, fold sorting, time-order check, error `Display`, doc drafts, CI workflow) | Wrote the core implementation and most tests; made all design decisions; reviewed all AI changes |
| 2026-11-11 | Claude Opus 5.5 (`claude-opus-5-5`) via Claude Code 2.1.321 | `rustellar-core` time types | Review; docs; some tests; switched `LightCurve` to the time types; added `from_btjd`; renamed `Duration` | Designed and wrote the time types, operators, and most tests |
| 2026-11-11 | Claude Opus 5.5 (`claude-opus-5-5`) via Claude Code 2.1.321 | `rustellar-model` parameters | Review; docs; tests; `LimbDarkening`; small fixes | Designed the parameters; wrote the error types, newtypes, `TransitParams`, and inclination conversion |
| 2026-11-12 | Claude Opus 5.5 (`claude-opus-5-5`) via Claude Code 2.1.321 | `rustellar-model` geometry | Review; docs; tests; `separation`; durations; contacts; `transit_times`; split `params` | Planned the geometry; wrote `phase_angle`, `in_front`, and `TimeDelta * f64` |
