"""
app.py — the web application (Phase 4): FastAPI server + one web page
====================================================================

Runs in yolo-env after:  pip install fastapi uvicorn python-multipart reportlab pydicom onnxruntime

    python app.py --models "$HOME/AIML Project/Test 2/RSNA/outputs/webapp_models"
    then open  http://127.0.0.1:8000  in the browser (Windows browsers reach WSL's localhost)

How it fits together
  Browser (static/index.html + app.js + style.css)
     POST /api/analyze   X-ray file + age / sex / view  -> findings, report, image ids
     GET  /api/image/<id>/xray.png and heatmap.png     -> the pictures the page draws on
     POST /api/chat      {id, question}                -> answer about THIS X-ray
     GET  /api/report/<id>.pdf                         -> downloadable report
     GET  /api/status                                  -> what is loaded
  Server (this file) -> pipeline.py -> frozen models from step 18

Privacy: uploaded X-rays are kept only in memory, for the last 30 analyses, and are never
written to disk. The server listens on 127.0.0.1 (this computer only) unless --host is changed.
One analysis runs at a time (a lock), because the GPU models are shared.
"""

import argparse
import io
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MAX_KEEP = 30
MAX_BYTES = 40 * 1024 * 1024
AMBER = (45, 155, 227)          # BGR of #E39B2D

app = FastAPI(title="Pneumonia X-ray Assistant")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
STATE = {"pipe": None, "store": OrderedDict(), "lock": threading.Lock(), "started": time.time()}


# ---------------------------------------------------------------------------- helpers
def png(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def heat_png(img, cam):
    """X-ray with the classifier heat-map blended in (only where the map is warm)."""
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    col = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = (np.clip(cam, 0, 1) ** 1.5 * 0.6)[..., None]
    return png((base * (1 - a) + col * a).astype(np.uint8))


def annotated(img, rec, cam=None):
    """X-ray with numbered boxes, for the PDF (the web page draws its own)."""
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, f in enumerate(rec["findings"], 1):
        x1, y1, x2, y2 = (np.array(f["box"]) * 512).astype(int)
        if f["box_confidence"] == "clearly marked":
            cv2.rectangle(base, (x1, y1), (x2, y2), AMBER, 2)
        else:                                                       # dashed = approximate
            for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
                n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) // 10 + 1
                for k in range(0, n, 2):
                    p = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
                    q = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
                    cv2.line(base, p, q, AMBER, 2)
        cv2.rectangle(base, (x1, y1), (x1 + 22, y1 + 22), AMBER, -1)
        cv2.putText(base, str(i), (x1 + 6, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return base


def get(rid):
    item = STATE["store"].get(rid)
    if item is None:
        raise HTTPException(404, "This analysis is no longer in memory. Upload the X-ray again.")
    return item


# ---------------------------------------------------------------------------- endpoints
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    p = STATE["pipe"]
    return {"models_loaded": p is not None, "language_model_loaded": bool(p and p._llm),
            "language_model_enabled": bool(p and p.use_llm), "analyses_in_memory": len(STATE["store"])}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), age: str = Form(""), sex: str = Form(""), view: str = Form("")):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 40 MB. Upload a single chest X-ray.")
    try:
        age_v = int(age) if age.strip() else None
    except ValueError:
        raise HTTPException(422, "Age must be a whole number of years.")
    if age_v is not None and not 0 <= age_v <= 120:
        raise HTTPException(422, "Age must be between 0 and 120.")
    with STATE["lock"]:
        try:
            res = STATE["pipe"].analyze(data, file.filename, age=age_v, sex=sex or None, view=view or None)
        except ValueError as e:
            raise HTTPException(422, str(e))
    rid = uuid.uuid4().hex[:12]
    STATE["store"][rid] = {"record": res["record"], "report": res["report"], "source": res["report_source"],
                           "xray": png(res["image"]), "heat": heat_png(res["image"], res["cam"]),
                           "pdf_img": png(annotated(res["image"], res["record"])), "name": file.filename,
                           "time": datetime.now().strftime("%d %b %Y, %H:%M"), "chat": []}
    while len(STATE["store"]) > MAX_KEEP:
        STATE["store"].popitem(last=False)
    return JSONResponse({"id": rid, "record": res["record"], "report": res["report"],
                         "report_source": res["report_source"], "header": res["header"], "seconds": res["seconds"]})


@app.get("/api/image/{rid}/{kind}.png")
def image(rid: str, kind: str):
    item = get(rid)
    if kind not in ("xray", "heat"):
        raise HTTPException(404, "Unknown image.")
    return Response(item[kind], media_type="image/png", headers={"Cache-Control": "no-store"})


class Question(BaseModel):
    id: str
    question: str


@app.post("/api/chat")
def chat(q: Question):
    item = get(q.id)
    if not q.question.strip():
        raise HTTPException(422, "Type a question about this X-ray.")
    with STATE["lock"]:
        ans = STATE["pipe"].chat(item["record"], q.question)
    item["chat"].append((q.question.strip(), ans["answer"]))
    return ans


@app.get("/api/report/{rid}.pdf")
def report_pdf(rid: str):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    item = get(rid)
    rec = item["record"]; c = rec["classifier"]; p = rec["patient"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Chest X-ray report", author="Pneumonia X-ray Assistant")
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Heading1"], fontSize=16, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=11, leading=15)
    small = ParagraphStyle("s", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#5B6B7A"))
    who = f"{p['age'] if p['age'] is not None else 'age unknown'}{' years' if p['age'] is not None else ''}, " \
          f"{ {'F': 'female', 'M': 'male'}.get(p['sex'], 'sex unknown') }, {p['view']} view"
    story = [Paragraph("Chest X-ray report", h),
             Paragraph(f"{who} &nbsp;|&nbsp; file: {item['name']} &nbsp;|&nbsp; analysed {item['time']}", small),
             Spacer(1, 6 * mm)]
    img = Image(io.BytesIO(item["pdf_img"]), width=78 * mm, height=78 * mm)
    rows = [["Result", c["result"]], ["Probability", f"{c['probability']:.2f} ({c['band']})"],
            ["Location", {"boxes": "marked", "possible_location": "approximate only", "none": "not localised"}[rec["localisation"]]]]
    for i, f in enumerate(rec["findings"], 1):
        rows.append([f"Finding {i}", f"{f['side']} lung, {' and '.join(f['zones'])}; {f['extent']}; {f['box_confidence']}"])
    t = Table(rows, colWidths=[24 * mm, 62 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5B6B7A")),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#C9D1D9"))]))
    story += [Table([[img, t]], colWidths=[84 * mm, 90 * mm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
              Spacer(1, 6 * mm), Paragraph("Report", ss["Heading3"]), Paragraph(item["report"], body)]
    if item["chat"]:
        story += [Spacer(1, 4 * mm), Paragraph("Questions asked", ss["Heading3"])]
        for qq, aa in item["chat"]:
            story += [Paragraph(f"<b>{qq}</b>", body), Paragraph(aa, body), Spacer(1, 2 * mm)]
    src = ("Written by the fine-tuned language model and checked against the findings."
           if item["source"] == "llm" else "Template report: the language model's text did not pass the check.")
    story += [Spacer(1, 6 * mm), Paragraph(src + " Boxes: solid = clearly marked, dashed = approximate.", small),
              Paragraph("Research prototype (RSNA Pneumonia Detection data). Not a medical device and not a "
                        "diagnosis. A qualified clinician must review the X-ray and the result.", small)]
    doc.build(story)
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="xray_report_{rid}.pdf"'})


# ---------------------------------------------------------------------------- start
def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True, help="webapp_models folder from step 18")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-llm", action="store_true", help="template reports only (no GPU memory for GPT-Neo)")
    ap.add_argument("--preload-llm", action="store_true", help="load GPT-Neo at start instead of at the first X-ray")
    a = ap.parse_args()
    print("Loading the classifier and detector ...")
    STATE["pipe"] = Pipeline(a.models, use_llm=not a.no_llm)
    if a.preload_llm and not a.no_llm:
        print("Loading GPT-Neo + LoRA ...")
        STATE["pipe"].llm()
    print(f"Ready: open http://{a.host}:{a.port} in your browser")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
