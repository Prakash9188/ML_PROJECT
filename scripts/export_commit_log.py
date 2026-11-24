"""Write docs/COMMIT_LOG.md: every commit's short hash, full hash, ISO timestamp and subject.
Needed for the assignment report ("commits, hashes, timestamps"). Run any time:
    python scripts/export_commit_log.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    fmt = "%h|%H|%aI|%an|%s"
    out = subprocess.run(["git", "log", "--reverse", f"--pretty=format:{fmt}"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    lines = ["# Commit log", "", "| # | short hash | timestamp (author date) | message | full hash |", "|---|---|---|---|---|"]
    for i, ln in enumerate(out, 1):
        sh, fh, ts, _, msg = ln.split("|", 4)
        lines.append(f"| {i} | `{sh}` | {ts} | {msg} | `{fh}` |")
    (ROOT / "docs" / "COMMIT_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
