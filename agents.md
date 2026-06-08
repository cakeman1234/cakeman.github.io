Local file reference rules (strict):
- Do not use local-file Markdown/URL links (e.g., `[x](...)`, `file:///...`).
- Only use pure references inside single backticks: `@path[:line[:column]]`.
- Use `/` in paths (including Windows), e.g., `@D:/repo/src/main.cpp:128`; relative paths use the same format, e.g., `@src/main.cpp:128`.
- Write explanations in normal prose; keep file refs as standalone `@...`; if multiple files are mentioned, output multiple separate `@...` refs.
