from __future__ import annotations

import csv
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "annotations" / "condition_labels_template.csv"
TAGS = ("soiled", "creased", "torn", "folded")
VALUES = {"yes", "no", "uncertain"}
HOST = "127.0.0.1"
PORT = 8765

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UGX Banknote Condition Annotation</title>
<style>
:root { font-family: Segoe UI, Arial, sans-serif; color: #18202b; background: #f3f5f8; }
* { box-sizing: border-box; }
body { margin: 0; }
header { background: #172b4d; color: white; padding: 14px 22px; display:flex; justify-content:space-between; gap:12px; align-items:center; }
main { max-width: 1180px; margin: 18px auto; padding: 0 16px; }
#status { font-size: 14px; opacity: .9; }
.card { background:white; border:1px solid #d8dee8; border-radius:10px; padding:16px; margin-bottom:14px; }
.meta { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; font-size:15px; }
.image-wrap { height:min(58vh, 560px); display:flex; align-items:center; justify-content:center; background:#e9edf3; border-radius:8px; overflow:hidden; }
#note { max-width:100%; max-height:100%; object-fit:contain; }
.tags { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.tag { border:1px solid #d8dee8; border-radius:8px; padding:12px; }
.tag h2 { font-size:16px; margin:0 0 10px; text-transform:capitalize; }
.choices { display:flex; gap:6px; }
button, select, input, textarea { font:inherit; }
button { cursor:pointer; border:1px solid #bbc5d3; background:#fff; border-radius:7px; padding:8px 12px; }
button:hover { background:#edf3ff; }
.choice.selected { background:#2157a6; border-color:#2157a6; color:white; }
.controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.details { display:grid; grid-template-columns:1fr 180px 2fr; gap:12px; margin-top:14px; }
label { display:block; font-size:13px; color:#475467; margin-bottom:4px; }
input, select, textarea { width:100%; border:1px solid #bbc5d3; border-radius:6px; padding:8px; }
textarea { min-height:40px; resize:vertical; }
.spacer { flex:1; }
#saveState { color:#276749; min-width:100px; }
@media(max-width:760px) { .tags { grid-template-columns:repeat(2,1fr); } .details { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header><strong>UGX Banknote Condition Annotator</strong><span id="status">Loading images…</span></header>
<main>
  <section class="card meta">
    <strong id="imageInfo"></strong><span id="authInfo"></span>
  </section>
  <section class="card image-wrap"><img id="note" alt="Banknote being reviewed"></section>
  <section class="card">
    <div class="tags" id="tags"></div>
    <div class="details">
      <div><label for="annotator">Annotator</label><input id="annotator"></div>
      <div><label for="confidence">Confidence</label><select id="confidence"><option value="">Choose…</option><option>high</option><option>medium</option><option>low</option></select></div>
      <div><label for="notes">Notes (optional)</label><input id="notes"></div>
    </div>
  </section>
  <section class="card controls">
    <button id="previous">← Previous</button><button id="save">Save</button><button id="next">Next →</button>
    <span class="spacer"></span><span id="saveState"></span>
  </section>
  <p>Mark each tag yes, no, or uncertain. Tags are independent and may overlap. Each change is saved locally in the project CSV.</p>
</main>
<script>
const tagNames = ['soiled','creased','torn','folded'];
const choices = ['yes','no','uncertain'];
let rows = [];
let index = 0;
let current = null;
let states = {};
const byId = id => document.getElementById(id);
const tagsRoot = byId('tags');
for (const tag of tagNames) {
  const box = document.createElement('div');
  box.className = 'tag';
  const title = document.createElement('h2');
  title.textContent = tag;
  const buttons = document.createElement('div');
  buttons.className = 'choices';
  for (const value of choices) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'choice';
    button.textContent = value;
    button.dataset.tag = tag;
    button.dataset.value = value;
    button.addEventListener('click', () => setChoice(tag, value));
    buttons.appendChild(button);
  }
  box.append(title, buttons);
  tagsRoot.appendChild(box);
}
function setChoice(tag, value) {
  states[tag] = value;
  document.querySelectorAll('[data-tag="' + tag + '"]').forEach(button => {
    button.classList.toggle('selected', button.dataset.value === value);
  });
}
function showRow() {
  current = rows[index];
  states = {};
  byId('imageInfo').textContent = current.image_id + '  •  ' + (index + 1) + ' of ' + rows.length + '  •  ' + current.split;
  byId('authInfo').textContent = 'Existing label: ' + current.authenticity_label;
  byId('note').src = '/image/' + index;
  byId('annotator').value = current.annotator || '';
  byId('confidence').value = current.confidence || '';
  byId('notes').value = current.notes || '';
  for (const tag of tagNames) setChoice(tag, current[tag] || '');
  byId('saveState').textContent = current.annotation_status === 'complete' ? 'Already reviewed' : '';
  const complete = rows.filter(row => row.annotation_status === 'complete').length;
  byId('status').textContent = complete + ' of ' + rows.length + ' reviewed';
}
async function saveRow() {
  if (!current) return false;
  const payload = {
    index,
    tags: states,
    annotator: byId('annotator').value.trim(),
    confidence: byId('confidence').value,
    notes: byId('notes').value.trim()
  };
  byId('saveState').textContent = 'Saving…';
  const response = await fetch('/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Save failed');
  rows[index] = result.row;
  current = rows[index];
  const complete = rows.filter(row => row.annotation_status === 'complete').length;
  byId('status').textContent = complete + ' of ' + rows.length + ' reviewed';
  byId('saveState').textContent = result.row.annotation_status === 'complete' ? 'Saved • complete' : 'Saved';
  return true;
}
byId('save').addEventListener('click', () => saveRow().catch(error => alert(error.message)));
byId('previous').addEventListener('click', async () => {
  try { if (await saveRow() && index > 0) { index--; showRow(); } } catch (error) { alert(error.message); }
});
byId('next').addEventListener('click', async () => {
  try { if (await saveRow() && index < rows.length - 1) { index++; showRow(); } } catch (error) { alert(error.message); }
});
document.addEventListener('keydown', event => {
  if (event.target.matches('input, textarea, select')) return;
  if (event.key === 'ArrowRight') byId('next').click();
  if (event.key === 'ArrowLeft') byId('previous').click();
});
fetch('/rows').then(response => response.json()).then(data => {
  rows = data.rows;
  index = rows.findIndex(row => row.annotation_status !== 'complete');
  if (index < 0) index = 0;
  showRow();
}).catch(error => { byId('status').textContent = error.message; });
</script>
</body></html>
"""


def read_rows() -> tuple[list[dict[str, str]], list[str]]:
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary = CSV_PATH.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(CSV_PATH)


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, status: int, data: dict) -> None:
        self.respond(
            status,
            json.dumps(data).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.respond(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/rows":
            rows, _ = read_rows()
            self.json_response(200, {"rows": rows})
            return
        if path.startswith("/image/"):
            try:
                index = int(path.rsplit("/", 1)[1])
                rows, _ = read_rows()
                image_path = (ROOT / rows[index]["image_path"]).resolve()
                image_path.relative_to(ROOT.resolve())
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self.respond(200, image_path.read_bytes(), content_type)
            except (ValueError, IndexError, KeyError, OSError) as error:
                self.json_response(404, {"error": str(error)})
            return
        self.json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/save":
            self.json_response(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            index = int(payload["index"])
            rows, fieldnames = read_rows()
            row = rows[index]
            tags = payload.get("tags", {})
            if not all(tags.get(tag) in VALUES for tag in TAGS):
                raise ValueError("Choose yes, no, or uncertain for all four tags.")
            for tag in TAGS:
                row[tag] = tags[tag]
            row["annotator"] = str(payload.get("annotator", "")).strip()
            row["confidence"] = str(payload.get("confidence", "")).strip()
            row["notes"] = str(payload.get("notes", "")).strip()
            row["annotation_status"] = "complete"
            write_rows(rows, fieldnames)
            self.json_response(200, {"row": row})
        except (ValueError, KeyError, IndexError, json.JSONDecodeError, OSError) as error:
            self.json_response(400, {"error": str(error)})

    def log_message(self, format: str, *args) -> None:
        pass


if __name__ == "__main__":
    if not CSV_PATH.is_file():
        raise SystemExit(f"Annotation CSV not found: {CSV_PATH}")
    rows, _ = read_rows()
    if not rows:
        raise SystemExit("The annotation CSV is empty.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT} in your browser to review {len(rows)} images.")
    print("Press Ctrl+C here to stop the local annotation server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotation server stopped.")
    finally:
        server.server_close()
