#!/usr/bin/env python3
"""Print key-up -> last-paste timelines from process.log "release:" lines.

    scripts/release-timeline.py [process.log] [--last N]

Offsets are ms after the hotkey key-up event (capture clock).
"""

import re
import sys
from pathlib import Path

LOG = Path.home() / "Library/Application Support/voice-control/logs/process.log"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=(\S+)")


def parse(path):
    runs = []
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if "release: " in line:
            message = line.split("release: ", 1)[1]
            name = FIELD.sub("", message).strip()
        elif "clipboard" in line and "clipboard_capture_ms" in line:
            name = "clipboard snapshot"
        elif "live dictation ended" in line:
            name = "live dictation ended"
        else:
            continue
        fields = {k: v for k, v in FIELD.findall(line.split(": ", 1)[-1])}
        if name == "key up":
            runs.append({"base": int(fields["key_up_ms"]), "events": []})
        if runs:
            runs[-1]["events"].append((name, fields))
    return runs


def show(run):
    base = run["base"]
    last_done = last_post = None
    pending_at_key_up = 0
    print(f"--- key up @ {base} ms")
    for name, f in run["events"]:
        at = f.get("at_ms")
        offset = f"{int(at) - base:>6} ms" if at else " " * 9
        extra = ""
        if name == "live paste done":
            queued, dequeued = int(f["queued_ms"]) - base, int(f["dequeued_ms"]) - base
            extra = (f"queued {queued:+} dequeued {dequeued:+} "
                     f"paste {int(at) - int(f['dequeued_ms'])} ms chars={f['chars']} last={f['last']}")
            pending_at_key_up += queued < 0
            last_done = int(at) - base
        elif name == "cmd-v posted":
            last_post = int(at) - base
        elif name == "capture drained":
            extra = f"drain {f['drain_ms']} ms"
        elif name == "clipboard snapshot":
            extra = f"{f['clipboard_capture_ms']} ms"
        elif name == "live dictation ended":
            extra = f"audio {f.get('audio_ms')} ms, ended->finalized {f.get('finalize_ms')} ms"
        elif name == "key up":
            extra = f"handled {int(at) - base} ms after the event"
        print(f"  {offset}  {name:<24} {extra}")
    print(f"  key-up -> last cmd-v: {last_post} ms (text on screen); -> last paste settled: {last_done} ms; "
          f"pastes queued before key-up: {pending_at_key_up}")


def main():
    args = sys.argv[1:]
    last = None
    if "--last" in args:
        i = args.index("--last")
        last = int(args[i + 1])
        del args[i : i + 2]
    runs = parse(Path(args[0]) if args else LOG)
    for run in runs[-last:] if last else runs:
        show(run)


if __name__ == "__main__":
    main()
