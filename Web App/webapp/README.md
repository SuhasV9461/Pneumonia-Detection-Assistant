# Pneumonia X-ray Assistant: the web application

A FastAPI server (Python) and one web page (HTML, CSS and JavaScript). Upload a chest X-ray to get:
- the calibrated pneumonia result;
- the marked regions and the classifier heat-map;
- a written report from the fine-tuned GPT-Neo;
- a chat about that X-ray;
- a PDF report.

## Files

| File | What it does |
|---|---|
| `app.py` | Web server: endpoints `/api/analyze`, `/api/chat`, `/api/report/<id>.pdf`, `/api/image/...`, `/api/status` |
| `pipeline.py` | Everything done to one X-ray, copied exactly from steps 1, 4b, 5, 10, 13, 14 and 16. Includes a self-test |
| `static/index.html`, `style.css`, `app.js` | The web page |
| `../18_package_models.py` | Puts all frozen models in `outputs/webapp_models/` (run once, in ai-env) |

This `webapp` folder must sit **inside the code folder**, because `pipeline.py` imports `calibration_utils`, `findings_utils` and `llm_utils` from there.

## Setup, once

```bash
# 1. ai-env: export the classifier to ONNX and package everything
conda activate ai-env            # or however you activate it
pip install tf2onnx onnx onnxruntime
O="$HOME/AIML Project/Test 2/RSNA/outputs"
python 18_package_models.py --outputs "$O" \
  --detector-run "$O/detector_runs/yolo11s_512_20261028-0921" \
  --llm-run "$O/llm_runs/gpt-neo-1.3B_20261101-0754"
#    -> must print "max logit difference ... 0.0xxx" for both members, then the file list

# 2. yolo-env: web packages, then the self-test (must say PASS)
source ~/yolo-env/bin/activate
pip install onnxruntime fastapi uvicorn python-multipart reportlab pydicom
cd webapp
python pipeline.py --outputs "$O" --selftest 50
```

## Start the website

```bash
source ~/yolo-env/bin/activate
cd "$HOME/AIML Project/Test 2/webapp"
python app.py --models "$HOME/AIML Project/Test 2/RSNA/outputs/webapp_models" --preload-llm
```

Then open **http://127.0.0.1:8000** in your Windows browser (it reaches WSL's localhost).

- `--preload-llm` loads GPT-Neo at start, about 30 s. Without it, the first X-ray waits for the load instead.
- `--no-llm` gives template reports and answers only (useful if the GPU is busy).
- Stop the server with Ctrl+C.

## Safety design (worth explaining in the review)

- **The classifier decides, the detector locates, the LLM only writes.** GPT-Neo never sees the image, only the findings record.
- **Every LLM report is checked against the findings** (`llm_utils.check`). If it states anything not in the findings, or leaves out the result, an affected lung or the disclaimer, the checked template report is shown instead, and the page says so.
- **Chat answers are checked the same way.** The question is matched to one of the 9 trained question kinds, and the answer must contain the fact asked for. Unreadable or unsupported answers are replaced by the checked standard answer, again with a note on the page.
- **Privacy:** X-rays are kept only in memory (the last 30 analyses) and never written to disk. The server listens on this computer only (127.0.0.1).

## Sharing a link during a review (optional)

```bash
# in a second WSL terminal, with the site running:
curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared && ./cloudflared tunnel --url http://127.0.0.1:8000
```

This prints a temporary `https://….trycloudflare.com` link that works until you press Ctrl+C. Anyone with the link can use the site while it runs, so share it only for the demo, and don't upload real patient X-rays through it.
