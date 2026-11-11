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
| 2026-11-10 to 2026-11-11 | Claude Opus 5.5 (`claude-opus-5-5`) via Claude Code 2.1.321 | `rustellar-core`: `src/lightcurve.rs`, `src/error.rs`, `tests/lightcurve.rs`, `tests/error.rs` | Reviewed the plan and corrected formulas in the private notes; explained Rust concepts and reviewed the author's code; at the author's request made small edits: some test cases, the phase-sorting step in `fold`, the strictly-increasing time check, the `Display`/`Error` implementations, and first drafts of doc comments | Wrote `LightCurve::new`, `fold`, the accessors, and most tests; made every design decision (validation rules, rejecting repeated times, sorting folded output by phase); edited the doc comments; reviewed every AI change and ran the tests before committing |
