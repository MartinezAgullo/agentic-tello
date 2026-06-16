"""Generate two Excalidraw diagrams for the agentic-tello system.

Run:  uv run python docs/gen_excalidraw.py
Open the resulting .excalidraw files at https://excalidraw.com (File ▸ Open).

This is a plain generator — Excalidraw files are just JSON, so we build the
element list programmatically (boxes with centred labels + bound arrows).
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent


def _seed():
    return random.randint(1, 2**31)


def rect(eid, x, y, w, h, *, bg="#ffffff", stroke="#1e1e1e", dash=False, group=None):
    return {
        "id": eid, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "dashed" if dash else "solid", "roughness": 1,
        "opacity": 100, "groupIds": [group] if group else [], "frameId": None,
        "roundness": {"type": 3}, "seed": _seed(), "versionNonce": _seed(),
        "isDeleted": False, "boundElements": [], "updated": 1, "link": None, "locked": False,
    }


def text(eid, x, y, w, h, label, *, size=20, container=None, color="#1e1e1e", align="center"):
    lines = label.count("\n") + 1
    return {
        "id": eid, "type": "text", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
        "seed": _seed(), "versionNonce": _seed(), "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False, "fontSize": size, "fontFamily": 1,
        "text": label, "textAlign": align, "verticalAlign": "middle",
        "containerId": container, "originalText": label, "autoResize": True,
        "lineHeight": 1.25, "baseline": int(size * lines),
    }


def boxed(elements, eid, x, y, w, h, label, *, bg="#ffffff", stroke="#1e1e1e",
          size=20, dash=False, group=None, tcolor="#1e1e1e"):
    """A rectangle with a centred text label bound to it."""
    r = rect(eid, x, y, w, h, bg=bg, stroke=stroke, dash=dash, group=group)
    tid = eid + "_t"
    t = text(tid, x, y, w, h, label, size=size, container=eid, color=tcolor)
    r["boundElements"] = [{"type": "text", "id": tid}]
    elements += [r, t]
    return r


def arrow(eid, x1, y1, x2, y2, *, start=None, end=None, color="#1e1e1e",
          dash=False, label=None, head="arrow"):
    el = {
        "id": eid, "type": "arrow", "x": x1, "y": y1,
        "width": abs(x2 - x1), "height": abs(y2 - y1), "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "dashed" if dash else "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": {"type": 2},
        "seed": _seed(), "versionNonce": _seed(), "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
        "points": [[0, 0], [x2 - x1, y2 - y1]], "lastCommittedPoint": None,
        "startBinding": {"elementId": start, "focus": 0, "gap": 4} if start else None,
        "endBinding": {"elementId": end, "focus": 0, "gap": 4} if end else None,
        "startArrowhead": None, "endArrowhead": head,
    }
    return el


def scene(elements):
    return {
        "type": "excalidraw", "version": 2, "source": "agentic-tello/docs/gen_excalidraw.py",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }


# palette
BLUE = "#a5d8ff"; GREEN = "#b2f2bb"; YELLOW = "#ffec99"; ORANGE = "#ffd8a8"
RED = "#ffc9c9"; GREY = "#e9ecef"; PURPLE = "#d0bfff"


# ───────────────────────── diagram 1: architecture (chokepoint) ──────────────
def architecture():
    e = []
    cx, w = 360, 320
    # title
    e.append(text("t_title", 300, 20, 420, 30, "Arquitectura — un único chokepoint de actuación", size=24))

    # central actuation spine
    spine = [
        ("a0", "tools / agent loop / web UI", GREY, 56),
        ("a1", "ControlArbiter\nAUTO vs MANUAL · operator preempts", BLUE, 70),
        ("a2", "SafeTello\nclamp steps · geofence · height/battery · watchdog", GREEN, 70),
        ("a3", "TelloController\ndjitellopy + vídeo low-latency", YELLOW, 70),
        ("a4", "Tello (drone)", ORANGE, 56),
    ]
    y = 80
    ys = []
    for eid, lbl, bg, h in spine:
        boxed(e, eid, cx, y, w, h, lbl, bg=bg, size=16)
        ys.append((eid, y, h))
        y += h + 46
    # spine arrows
    for i in range(len(spine) - 1):
        eid_a, ya, ha = ys[i]; eid_b, yb, hb = ys[i + 1]
        e.append(arrow(f"sp{i}", cx + w / 2, ya + ha, cx + w / 2, yb, start=eid_a, end=eid_b))

    # E-STOP bypass (right side, arbiter -> drone, dashed red)
    boxed(e, "estop", cx + w + 90, ys[1][1], 150, 56, "E-STOP\narbiter.emergency()", bg=RED, size=14)
    e.append(arrow("estop_in", cx + w, ys[1][1] + 28, cx + w + 90, ys[1][1] + 28,
                   start="a1", end="estop", color="#e03131"))
    e.append(arrow("estop_out", cx + w + 165, ys[1][1] + 56, cx + w + 165, ys[4][1] + 28,
                   color="#e03131", dash=True, end="a4"))
    e.append(text("estop_lbl", cx + w + 175, (ys[1][1] + ys[4][1]) / 2, 150, 40,
                  "bypassea\ntodos los guards", size=12, color="#e03131", align="left"))

    # left modules feeding the top (the source-of-truth library + packages)
    mods = [
        ("m_perc", "perception/\nopen-vocab detector", BLUE),
        ("m_brain", "brain/\nOllama VLM + prompts", PURPLE),
        ("m_agent", "agent/\nsense-plan-act loop", GREEN),
        ("m_mcp", "tello_mcp/\nMCP sobre tools.py", GREY),
        ("m_web", "web/\nFastAPI dashboard", YELLOW),
    ]
    mx, mw, mh = 20, 230, 50
    my = 90
    for eid, lbl, bg in mods:
        boxed(e, eid, mx, my, mw, mh, lbl, bg=bg, size=13)
        e.append(arrow(eid + "_a", mx + mw, my + mh / 2, cx, ys[0][1] + 28, start=eid, end="a0", dash=True))
        my += mh + 16
    e.append(text("m_note", mx, my + 4, mw, 40,
                  "tools.py = registro único de acciones\nconfig.py = todos los caps", size=12, align="left"))

    return scene(e)


# ───────────────────────── diagram 2: functioning (dual loop) ────────────────
def functioning():
    e = []
    e.append(text("t_title", 280, 20, 560, 30,
                  "Funcionamiento — dual-loop sense-plan-act (latencia por cadencia)", size=22))

    # goal
    boxed(e, "goal", 380, 80, 320, 56, 'GOAL en NL\n"find the backpack and take a picture"', bg=ORANGE, size=14)

    # ── SLOW LOOP container ──
    e.append(rect("slow_box", 60, 180, 460, 230, bg="#f8f0fc", stroke="#9c36b5", dash=True))
    e.append(text("slow_lbl", 80, 190, 400, 24, "SLOW LOOP  ·  VLM cada VLM_INTERVAL_S≈3s (en hover)", size=15, color="#9c36b5", align="left"))
    boxed(e, "vlm", 100, 230, 380, 70,
          "VLM (Ollama, non-thinking, keep_alive=-1)\ntraduce goal→targets · juzga done · NO actúa", bg=PURPLE, size=13)
    boxed(e, "writes", 130, 320, 320, 70,
          "escribe en blackboard:\ntarget_queries · search_hint · scene · done", bg="#eebefa", size=13)
    e.append(arrow("vlm_w", 290, 300, 290, 320, start="vlm", end="writes"))

    # ── BLACKBOARD ──
    boxed(e, "bb", 600, 250, 220, 120,
          "MissionState\n(blackboard)\n\nphase · target_queries\nsearch_hint · scene", bg=GREY, size=14)

    # ── FAST LOOP container ──
    e.append(rect("fast_box", 60, 470, 760, 260, bg="#ebfbee", stroke="#2f9e44", dash=True))
    e.append(text("fast_lbl", 80, 480, 600, 24, "FAST LOOP  ·  cada frame nuevo (~fps del detector)", size=15, color="#2f9e44", align="left"))
    boxed(e, "det", 100, 520, 200, 70, "Detector open-vocab\n(YOLO-World) localiza\nel target", bg=BLUE, size=13)
    boxed(e, "phase", 330, 520, 220, 90,
          "Phase machine\nSEARCH → APPROACH\n→ CAPTURE → DONE", bg=GREEN, size=13)
    boxed(e, "servo", 580, 520, 200, 70, "Servoing determinista\n→ ('rc' lr,fb,ud,yaw)\n→ move/rotate/snapshot", bg=YELLOW, size=12)
    e.append(arrow("d_p", 300, 555, 330, 555, start="det", end="phase"))
    e.append(arrow("p_s", 550, 560, 580, 560, start="phase", end="servo"))

    # flows
    e.append(arrow("g_vlm", 480, 136, 320, 230, start="goal", end="vlm"))
    e.append(arrow("w_bb", 450, 355, 600, 320, start="writes", end="bb", color="#9c36b5"))
    e.append(text("w_bb_l", 470, 330, 130, 20, "QUÉ perseguir", size=12, color="#9c36b5"))
    e.append(arrow("bb_fast", 700, 370, 430, 520, start="bb", end="phase", color="#2f9e44"))
    e.append(text("bb_fast_l", 560, 430, 160, 20, "lee QUÉ → decide CÓMO", size=12, color="#2f9e44"))

    # actuation out + loop back (servo -> drone -> new frame -> detector)
    boxed(e, "drone", 580, 650, 200, 50, "→ ControlArbiter→SafeTello→Tello", bg=ORANGE, size=11)
    e.append(arrow("s_d", 680, 590, 680, 650, start="servo", end="drone"))
    e.append(arrow("loop", 580, 675, 200, 675, color="#2f9e44", dash=True))
    e.append(arrow("loop2", 200, 675, 200, 555, end="det", color="#2f9e44", dash=True))
    e.append(text("loop_l", 300, 685, 260, 20, "nuevo frame → re-percibe", size=12, color="#2f9e44"))

    # resilience note
    e.append(text("note", 60, 745, 760, 40,
                  "Si Ollama cae: el fast loop SIGUE volando sobre el último target. El VLM nunca toca los motores.",
                  size=13, color="#e8590c", align="left"))
    return scene(e)


def main():
    (OUT / "architecture.excalidraw").write_text(json.dumps(architecture(), indent=2))
    (OUT / "functioning.excalidraw").write_text(json.dumps(functioning(), indent=2))
    print("wrote docs/architecture.excalidraw and docs/functioning.excalidraw")


if __name__ == "__main__":
    main()
