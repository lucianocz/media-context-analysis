"""Related-event groups from objective visual links already found by detection.

A group never removes an event, never moves a timestamp, and never opens a
translation segment. The only link used here is the brief-insert return: the
detector already found a later shot that matches the look from before the
departure.
"""

from __future__ import annotations

from media_context_analysis.timeutil import as_float


def group_timeline(events: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for event in events:
        if event.get("category") != "brief_visual_insert":
            continue
        frames = event.get("frames")
        if not isinstance(frames, dict) or "return_s" not in frames:
            continue
        return_s = as_float(frames["return_s"])
        start = as_float(event["time_s"])
        members = [str(event["id"])]
        evidence = [f"{event['id']} is a brief visual insert that returns at {return_s:.3f}s"]
        for other in events:
            if other["id"] == event["id"]:
                continue
            other_time = as_float(other["time_s"])
            if abs(other_time - return_s) <= 0.001:
                members.append(str(other["id"]))
                evidence.append(
                    f"{other['id']} is at {other_time:.3f}s, the return shot of {event['id']}"
                )
            elif start + 0.001 < other_time < return_s - 0.001:
                members.append(str(other["id"]))
                evidence.append(
                    f"{other['id']} at {other_time:.3f}s falls inside brief insert {event['id']} "
                    f"({start:.3f}s to {return_s:.3f}s)"
                )
        if len(members) < 2:
            continue
        evidence.append("This group does not open a translation segment.")
        groups.append(
            {
                "id": f"g{len(groups) + 1:04d}",
                "kind": "brief_insert_return",
                "event_ids": members,
                "evidence": evidence,
                "creates_translation_segment": False,
            }
        )
    return groups
