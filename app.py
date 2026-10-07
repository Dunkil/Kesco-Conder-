"""KESCO Auto-Coder - local web app.  Run:  python app.py   then open http://127.0.0.1:5000"""
import os, re, io, uuid, difflib, json, webbrowser, threading
from pathlib import Path
import numpy as np, openpyxl, pandas as pd
from flask import Flask, request, jsonify, send_file, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
REF_SAVE = BASE / "kesco_reference.xlsx"

def _fail(msg): raise ValueError(msg)

# ---------------- matcher (identical logic to kesco_matcher.ipynb) ----------------
CODE_PATTERN = re.compile(r"^\d{3,4}-\d{1,3}$")

# strips gender markers (m/f/d), hours/schedule/company parentheticals, and
# "at <Company> GmbH/Ltd..." tails before matching -- doesn't touch your data
_GENDER_MARK = re.compile(r"\((?:[mwfdx]\s*/\s*){1,3}[mwfdx]\)", re.IGNORECASE)
_NOISE_PAREN = re.compile(
    r"\((?:[^()]*?(?:\d+\s*[-/]?\s*\d*\s*h(?:ours?|rs?)?\b|h/?week|std\.?/?wo\.?|"
    r"full[-\s]?time|part[-\s]?time|vollzeit|teilzeit|all genders|diverse|"
    r"gmbh|ltd\.?|llc|kg\b|\bag\b|inc\.?|co\.?\b)[^()]*?)\)",
    re.IGNORECASE,
)
_COMPANY_TAIL = re.compile(
    r"\b(?:at|@)\s+[A-Z][\w&.,'-]*(?:\s+[A-Z][\w&.,'-]*){0,4}\s*"
    r"(?:GmbH|Ltd\.?|LLC|Inc\.?|KG|AG|Co\.?|Group)\b.*$"
)


def load_kesco_reference(xlsx_path: str) -> pd.DataFrame:
    """Flatten the hierarchical KESCO workbook into {code, title, group ancestry} rows."""
    ws = openpyxl.load_workbook(xlsx_path, data_only=True).active
    norm = lambda g: re.sub(r"\s+", " ", str(g)).strip().lower() if g else None
    major = submajor = minor = unit = None
    records = []

    for g_raw, code, desc in (row[:3] for row in ws.iter_rows(min_row=2, values_only=True)):
        g, desc = norm(g_raw), (str(desc).strip() if desc else None)
        code_str = str(code).strip() if code is not None else None

        if g and code_str and not CODE_PATTERN.match(code_str):
            if g.startswith("major") and "sub" not in g: major = desc
            elif g.startswith("sub"): submajor = desc
            elif g.startswith("minor"): minor = desc
            elif g.startswith("unit"): unit = desc
            continue

        if code_str and CODE_PATTERN.match(code_str) and desc:
            records.append(dict(code=code_str, title=desc, major_group=major,
                                 sub_major_group=submajor, minor_group=minor, unit_group=unit))

    df = pd.DataFrame(records).drop_duplicates(subset=["code"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No occupation codes found -- expected columns [GROUP, CODE, DESCRIPTION].")
    return df


def _clean(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", str(text))
    text = re.split(r"\s*\|\s*", text)[0]
    text = re.split(r"\bLocation\s*:", text, flags=re.IGNORECASE)[0]
    text = re.split(r"\bwanted\.", text, flags=re.IGNORECASE)[0]
    text = _COMPANY_TAIL.sub("", text)
    text = _NOISE_PAREN.sub(" ", text)
    text = _GENDER_MARK.sub(" ", text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class KESCOMatcher:
    def __init__(self, reference, confidence_threshold: float = 0.45):
        """`reference`: path to KESCO .xlsx/.json/.csv, or an already-parsed DataFrame."""
        self.threshold = confidence_threshold
        if isinstance(reference, pd.DataFrame):
            self.ref = reference
        else:
            suf = Path(reference).suffix.lower()
            self.ref = (load_kesco_reference(reference) if suf in (".xlsx", ".xls") else
                        pd.read_json(reference) if suf == ".json" else
                        pd.read_csv(reference, dtype=str) if suf == ".csv" else
                        _fail("reference must be .xlsx, .json, .csv, or a DataFrame"))

        self._clean_titles = self.ref["title"].map(_clean).tolist()
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2)).fit(self._clean_titles)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(self._clean_titles)
        self.word_matrix = self.word_vec.transform(self._clean_titles)
        self.char_matrix = self.char_vec.transform(self._clean_titles)

    def match(self, title: str, top_n: int = 3):
        """Top-N candidate KESCO matches for one free-text title."""
        return self._match_many([title], top_n)[0]

    def _match_many(self, titles, top_n=3, prefilter_k=25, chunk_size=500):
        """Vectorised matching (fast on large datasets): TF-IDF narrows candidates,
        difflib only scores each row's top `prefilter_k` candidates."""
        clean_list = [_clean(t) for t in titles]
        results = [None] * len(clean_list)

        for start in range(0, len(clean_list), chunk_size):
            chunk = clean_list[start:start + chunk_size]
            word_sim = (self.word_vec.transform(chunk) @ self.word_matrix.T).toarray()
            char_sim = (self.char_vec.transform(chunk) @ self.char_matrix.T).toarray()
            pre_blend = 0.45 * word_sim + 0.35 * char_sim

            for i, clean in enumerate(chunk):
                if not clean:
                    results[start + i] = []
                    continue
                k = min(prefilter_k, pre_blend.shape[1])
                cand = np.argpartition(pre_blend[i], -k)[-k:]
                scored = sorted(
                    ((idx, 0.45 * word_sim[i, idx] + 0.35 * char_sim[i, idx]
                      + 0.20 * difflib.SequenceMatcher(None, clean, self._clean_titles[idx]).ratio())
                     for idx in cand),
                    key=lambda x: x[1], reverse=True,
                )[:top_n]
                results[start + i] = [self._row_dict(idx, s) for idx, s in scored]
        return results

    def _row_dict(self, idx, score):
        r = self.ref.iloc[idx]
        return {"code": r.code, "title": r.title, "confidence": round(float(score), 4),
                "major_group": r.major_group, "sub_major_group": r.sub_major_group,
                "minor_group": r.minor_group, "unit_group": r.unit_group}

    def match_batch(self, titles, top_n=3) -> pd.DataFrame:
        """Match a plain list of titles -> standalone lookup DataFrame."""
        rows = []
        for t, matches in zip(titles, self._match_many(list(titles), top_n)):
            if not matches:
                rows.append({"input_title": t, "status": "NEEDS_REVIEW"}); continue
            best = matches[0]
            row = {"input_title": t, "matched_code": best["code"], "matched_title": best["title"],
                   "confidence": best["confidence"], **{k: best[k] for k in
                   ("major_group", "sub_major_group", "minor_group", "unit_group")},
                   "status": "MATCHED" if best["confidence"] >= self.threshold else "NEEDS_REVIEW"}
            for j, alt in enumerate(matches[1:top_n], 1):
                row[f"alt{j}_code"], row[f"alt{j}_title"], row[f"alt{j}_confidence"] = \
                    alt["code"], alt["title"], alt["confidence"]
            rows.append(row)
        return pd.DataFrame(rows)

    def map_dataset(self, df: pd.DataFrame, occupation_column: str, code_column="kesco_code") -> pd.DataFrame:
        """Return a COPY of df with kesco_code* columns appended (original columns untouched).
        Matches unique titles once, then maps results back onto every row."""
        if occupation_column not in df.columns:
            raise KeyError(f"'{occupation_column}' not in columns: {list(df.columns)}")
        out = df.copy()
        titles = out[occupation_column].astype(str)
        uniq = titles.unique().tolist()
        by_title = dict(zip(uniq, self._match_many(uniq, top_n=1)))

        fields = ["code", "title", "confidence", "major_group", "sub_major_group", "minor_group", "unit_group"]
        cols = {f: [] for f in fields}
        statuses = []
        for t in titles:
            m = by_title.get(t) or [{}]
            best = m[0]
            for f in fields:
                cols[f].append(best.get(f))
            statuses.append("MATCHED" if (best.get("confidence") or 0) >= self.threshold else "NEEDS_REVIEW")

        out[code_column] = cols["code"]
        out[f"{code_column}_title"] = cols["title"]
        out[f"{code_column}_confidence"] = cols["confidence"]
        out[f"{code_column}_status"] = statuses
        out[f"{code_column}_major_group"] = cols["major_group"]
        out[f"{code_column}_sub_major_group"] = cols["sub_major_group"]
        out[f"{code_column}_minor_group"] = cols["minor_group"]
        out[f"{code_column}_unit_group"] = cols["unit_group"]
        return out

    def map_file(self, input_path, occupation_column, output_path=None, sheet_name=0, code_column="kesco_code"):
        """Read a dataset file, append KESCO columns, write it back out."""
        p = Path(input_path)
        df = (pd.read_excel(input_path, sheet_name=sheet_name) if p.suffix.lower() in (".xlsx", ".xls")
              else pd.read_csv(input_path) if p.suffix.lower() == ".csv"
              else _fail(f"Unsupported file type: {p.suffix}"))
        mapped = self.map_dataset(df, occupation_column, code_column=code_column)

        out_p = Path(output_path or p.with_name(p.stem + "_KESCO_mapped" + p.suffix))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        (mapped.to_excel if out_p.suffix.lower() in (".xlsx", ".xls") else mapped.to_csv)(out_p, index=False)
        return mapped

    def map_multiple_files(self, file_specs, output_dir=None, code_column="kesco_code"):
        """Batch-process several country/county datasets, each with its own column name.
        file_specs: [{"path": ..., "occupation_column": ..., "sheet_name": 0 (optional)}, ...]"""
        results = {}
        for spec in file_specs:
            p = Path(spec["path"])
            out_path = str(Path(output_dir) / (p.stem + "_KESCO_mapped" + p.suffix)) if output_dir else spec.get("output_path")
            mapped = self.map_file(spec["path"], spec["occupation_column"], out_path,
                                    spec.get("sheet_name", 0), code_column)
            results[spec["path"]] = mapped
            n_review = (mapped[f"{code_column}_status"] == "NEEDS_REVIEW").sum()
            print(f"{spec['path']}: {len(mapped)} rows, {n_review} flagged NEEDS_REVIEW")
        return results

# ---------------- web app ----------------
app = Flask(__name__)
PUBLIC = os.environ.get("PUBLIC") == "1"   # set on the web host: locks the reference file
app.config["MAX_CONTENT_LENGTH"] = (25 if PUBLIC else 100) * 1024 * 1024
STATE = {"matcher": None, "ref_name": None}
JOBS = {}  # id -> {"name", "raw", "mapped", "column"}
GUESS = re.compile(r"job.?title|occupation|position|role|designation|title", re.I)

def find_reference():
    cands = [REF_SAVE] + [p for p in BASE.glob("*.xlsx") if "kesco" in p.name.lower() and p != REF_SAVE and "mapped" not in p.name.lower()]
    for p in cands:
        if p.exists():
            try:
                STATE["matcher"] = KESCOMatcher(load_kesco_reference(str(p)), 0.45)
                STATE["ref_name"] = p.name
                return
            except Exception as e:
                print("Could not use", p.name, "->", e)
find_reference()

def read_any(f):
    name = f.filename.lower()
    return pd.read_csv(f) if name.endswith(".csv") else pd.read_excel(f)

def guess_column(cols):
    for c in cols:
        if GUESS.search(str(c)): return str(c)
    return str(cols[0])

def to_records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))

def run_map(job, column):
    job["column"] = column
    job["mapped"] = STATE["matcher"].map_dataset(job["raw"], column)
    m = job["mapped"]
    return {"id": job["id"], "name": job["name"], "column": column, "columns": list(job["raw"].columns),
            "rows": len(m), "matched": int((m["kesco_code_status"] == "MATCHED").sum()),
            "review": int((m["kesco_code_status"] == "NEEDS_REVIEW").sum()),
            "cols": list(m.columns), "data": to_records(m.head(2000))}

@app.get("/")
def index(): return Response(PAGE, mimetype="text/html")

@app.get("/api/status")
def status(): return jsonify(ref=STATE["ref_name"], public=PUBLIC)

@app.post("/api/reference")
def set_reference():
    if PUBLIC: return jsonify(error="Reference is locked on the public site."), 403
    f = request.files["file"]; f.save(REF_SAVE); STATE["matcher"] = None; find_reference()
    if not STATE["matcher"]: return jsonify(error="Could not read that as a KESCO workbook (expected columns GROUP, CODE, DESCRIPTION)."), 400
    return jsonify(ref=STATE["ref_name"])

@app.post("/api/upload")
def upload():
    if not STATE["matcher"]: return jsonify(error="Upload the KESCO reference file first."), 400
    try: df = read_any(request.files["file"])
    except Exception as e: return jsonify(error=f"Could not read file: {e}"), 400
    jid = uuid.uuid4().hex[:10]
    while len(JOBS) >= 40: JOBS.pop(next(iter(JOBS)))  # keep memory small on free hosts
    job = JOBS[jid] = {"id": jid, "name": request.files["file"].filename, "raw": df}
    return jsonify(run_map(job, guess_column(list(df.columns))))

@app.post("/api/remap")
def remap():
    d = request.json; job = JOBS[d["id"]]
    return jsonify(run_map(job, d["column"]))

@app.get("/download/<jid>")
def download(jid):
    if jid not in JOBS: return Response("Session expired - please upload the file again.", 410)
    job = JOBS[jid]; m = job["mapped"]; stem = Path(job["name"]).stem
    buf = io.BytesIO()
    if request.args.get("fmt") == "csv":
        buf.write(m.to_csv(index=False).encode("utf-8-sig")); buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=f"{stem}_KESCO_mapped.csv", mimetype="text/csv")
    m.to_excel(buf, index=False); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{stem}_KESCO_mapped.xlsx")

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KESCO Auto-Coder - Kenya occupation coding tool</title><meta name="description" content="Free online tool: upload an Excel or CSV of job titles and automatically code them to the Kenyan Standard Classification of Occupations (KESCO)."><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--mut:#667085;--acc:#0b6b4f;--warn:#b54708;--line:#e4e7ec}
*{box-sizing:border-box}body{margin:0;font:15px system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:22px 32px;background:var(--acc);color:#fff}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.85}
main{max-width:1250px;margin:24px auto;padding:0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px}
#drop{border:2px dashed #98a2b3;border-radius:12px;padding:38px;text-align:center;cursor:pointer;color:var(--mut)}#drop.on{border-color:var(--acc);background:#ecfdf3}
button,.btn,select,input[type=text]{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
.btn.p,button.p{background:var(--acc);color:#fff;border-color:var(--acc)}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.stat{padding:10px 16px;border-radius:10px;background:var(--bg)}.stat b{font-size:20px;display:block}
.wrap{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left;max-width:320px;overflow:hidden;text-overflow:ellipsis}
th{position:sticky;top:0;background:#f2f4f7}th.k{background:#d1fadf}td.k{background:#f3fbf6}tr.r td.k{background:#fef0c7}
.tag{padding:2px 8px;border-radius:99px;font-size:12px}.M{background:#d1fadf;color:#05603a}.R{background:#fef0c7;color:var(--warn)}
#msg{color:#b42318;margin-top:8px}.hide{display:none}</style></head><body>
<header><h1>KESCO Auto-Coder</h1><p>Upload an Excel/CSV of job vacancies &rarr; occupations are coded to KESCO automatically.</p></header><main>
<div class="card row" id="refbox"><span id="reftxt">Checking KESCO reference&hellip;</span>
<label class="btn">Upload / replace KESCO reference<input type="file" id="reffile" accept=".xlsx" hidden></label></div>
<div class="card"><div id="drop">Drop your Excel / CSV file here, or click to browse<input type="file" id="file" accept=".xlsx,.xls,.csv" hidden></div><div id="msg"></div></div>
<div id="out" class="hide"><div class="card"><div class="row"><div class="stat"><b id="sRows">0</b>rows</div><div class="stat"><b id="sM">0</b>matched</div>
<div class="stat"><b id="sR">0</b>need review</div><span style="flex:1"></span>
<span>Occupation column:</span><select id="col"></select><button id="remap">Re-code</button></div>
<div class="row" style="margin-top:14px"><button class="p" id="viewBtn">View results</button><a class="btn p" id="dx">Download Excel</a><a class="btn" id="dc">Download CSV</a></div></div>
<div class="card hide" id="view"><div class="row" style="margin-bottom:10px"><select id="flt"><option value="">All rows</option><option value="NEEDS_REVIEW">Needs review only</option><option value="MATCHED">Matched only</option></select>
<input type="text" id="q" placeholder="Search..." style="flex:1"><span id="cnt" style="color:var(--mut)"></span></div><div class="wrap"><table id="t"></table></div></div></div></main>
<script>
const $=id=>document.getElementById(id);let cur=null;
async function ref(){const r=await(await fetch('/api/status')).json();if(r.public)$('refbox').innerHTML='KESCO reference: <b>'+r.ref+'</b> (built in)';else $('reftxt').innerHTML=r.ref?'KESCO reference loaded: <b>'+r.ref+'</b>':'<b style="color:#b54708">No KESCO reference yet</b> - upload your KESCO structure Excel file (once; it is remembered).'}
ref();
$('reffile').onchange=async e=>{const fd=new FormData();fd.append('file',e.target.files[0]);$('reftxt').textContent='Reading reference...';
const r=await fetch('/api/reference',{method:'POST',body:fd});const j=await r.json();if(!r.ok)$('msg').textContent=j.error;else $('msg').textContent='';ref()};
$('drop').onclick=()=>$('file').click();
['dragover','dragenter'].forEach(ev=>$('drop').addEventListener(ev,e=>{e.preventDefault();$('drop').classList.add('on')}));
['dragleave','drop'].forEach(ev=>$('drop').addEventListener(ev,e=>{e.preventDefault();$('drop').classList.remove('on')}));
$('drop').addEventListener('drop',e=>send(e.dataTransfer.files[0]));$('file').onchange=e=>send(e.target.files[0]);
async function send(f){if(f&&f.size>25*1024*1024){$('msg').textContent='File too large (max 25 MB).';return}if(!f)return;$('msg').textContent='';$('drop').firstChild.textContent='Coding '+f.name+' ...';const fd=new FormData();fd.append('file',f);
const r=await fetch('/api/upload',{method:'POST',body:fd});const j=await r.json();$('drop').firstChild.textContent='Drop another Excel / CSV file here, or click to browse';
if(!r.ok){$('msg').textContent=j.error;return}show(j)}
function show(j){cur=j;$('out').classList.remove('hide');$('sRows').textContent=j.rows;$('sM').textContent=j.matched;$('sR').textContent=j.review;
$('col').innerHTML=j.columns.map(c=>'<option'+(c==j.column?' selected':'')+'>'+c+'</option>').join('');
$('dx').href='/download/'+j.id;$('dc').href='/download/'+j.id+'?fmt=csv';render()}
$('remap').onclick=async()=>{const r=await fetch('/api/remap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur.id,column:$('col').value})});show(await r.json())};
$('viewBtn').onclick=()=>{$('view').classList.toggle('hide')};$('flt').onchange=render;$('q').oninput=render;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function render(){if(!cur)return;const q=$('q').value.toLowerCase(),f=$('flt').value;
let d=cur.data.filter(r=>(!f||r.kesco_code_status==f)&&(!q||JSON.stringify(r).toLowerCase().includes(q)));$('cnt').textContent=d.length+' rows'+(cur.rows>2000?' (preview of first 2,000 - download has all '+cur.rows+')':'');
$('t').innerHTML='<tr>'+cur.cols.map(c=>'<th class="'+(c.startsWith('kesco')?'k':'')+'">'+esc(c)+'</th>').join('')+'</tr>'+
d.slice(0,500).map(r=>'<tr class="'+(r.kesco_code_status=='NEEDS_REVIEW'?'r':'')+'">'+cur.cols.map(c=>{const k=c.startsWith('kesco');let v=r[c];
if(c=='kesco_code_status')v='<span class="tag '+(v=='MATCHED'?'M':'R')+'">'+v+'</span>';else v=esc(v);return '<td class="'+(k?'k':'')+'" title="'+(k?'':v)+'">'+v+'</td>'}).join('')+'</tr>').join('')}
</script></body></html>'''

if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
