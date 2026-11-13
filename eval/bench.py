#!/usr/bin/env python3
"""Benchmark presisi retrieval: RAG (pgvector) vs grep markdown vs graph.

Dipakai untuk membandingkan tiga cara menjawab pertanyaan operasional dari
knowledge-base AI-MOP:

  1. rag   : `ai-mop-ingestion search "<query>" -k 5` (embedding bge-m3)
  2. grep  : `grep -ricE "<kata kunci query>"` di folder knowledge-base
  3. graph : hasil codebase-memory-mcp `search_graph` yang disimpan manual
             sebagai graph.json (lihat eval/README.md)

Ground truth = satu dokumen KB yang berisi jawaban, ditentukan manual dari
struktur heading dokumen.

Pemakaian:
    KB_DIR=../ai-mop/knowledge-base ./eval/bench.py
    KB_DIR=../ai-mop/knowledge-base GRAPH_JSON=/tmp/eval/graph.json ./eval/bench.py

Metrik: Hit@1/3/5 dan MRR pada level dokumen, plus latensi per query.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KB = os.environ.get("KB_DIR", os.path.join(ROOT, "..", "ai-mop", "knowledge-base"))
BIN = os.environ.get("BIN", os.path.join(ROOT, "ai-mop-ingestion"))
GRAPH_JSON = os.environ.get("GRAPH_JSON", "")
OUT = os.environ.get("EVAL_OUT", os.path.join(HERE, "out"))
K = int(os.environ.get("TOP_K", "5"))

# id, query, dokumen jawaban (relatif terhadap knowledge-base), pola grep
# (kata kunci yang diambil dari query itu sendiri).
QUERIES = [
    ("q01", "mas web saya muter-muter terus gak bisa dibuka",
     "runbooks/RB-WP-001-health-check.md", "muter|redirect"),
    ("q02", "website kena injeksi judi online",
     "learnings/2026-11-03-seo-recovery-pn-koba.md", "injeksi|judi"),
    ("q03", "plugin error setelah update perlu rollback",
     "runbooks/RB-WP-003-plugin-update.md", "plugin|rollback"),
    ("q04", "berapa biaya maintenance website per bulan",
     "learnings/2026-11-11-rancangan-layanan-maintenance-wordpress.md", "maintenance|biaya"),
    ("q05", "langkah pertama kalau situs kena malware",
     "patterns/canonical-cases/CC-WP-001-wordpress-compromise.md", "malware|langkah"),
    ("q06", "restore backup updraftplus dan verifikasi",
     "runbooks/RB-WP-002-backup-verify.md", "backup|updraftplus"),
    ("q07", "DISALLOW_FILE_MODS muncul saat update plugin",
     "runbooks/RB-WP-003-plugin-update.md", "DISALLOW_FILE_MODS"),
    ("q08", "patch PHP CVE di produksi",
     "patterns/canonical-cases/CC-WP-002-php-cve-patch.md", "CVE|patch"),
    ("q09", "kenapa versi aplikasi tidak sama dengan kode di server",
     "learnings/2026-11-04-sipp-codeigniter2-deploy-traps.md", "versi|aplikasi"),
    ("q10", "cek sertifikat TLS yang mau expired",
     "runbooks/RB-WP-001-health-check.md", "sertifikat|expired"),
    ("q11", "custom domain wordpress lewat cloudflare masih error",
     "vendors/cloudflare/2026-11-02-wordpress-custom-domain-dpanel.md", "custom|domain|cloudflare"),
    ("q12", "harga pasaran jasa pembuatan website 2026",
     "learnings/2026-11-11-survey-harga-pasaran-website-2026.md", "harga|pasaran"),
    ("q13", "valyala/fastjson untuk parsing JSON cepat",
     "learnings/2026-11-10-high-rps-json-processing-and-cache-locality.md", "fastjson"),
    ("q14", "rencana hybrid retrieval dan fase pilot",
     "learnings/2026-11-02-rag-plan-implementasi.md", "hybrid|retrieval"),
]


def rag(query):
    t0 = time.time()
    p = subprocess.run([BIN, "search", query, "-k", str(K), "-json"],
                       capture_output=True, text=True)
    ms = (time.time() - t0) * 1000
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout).strip(), "ms": ms, "docs": [], "sections": [], "scores": []}
    rows = json.loads(p.stdout).get("results", [])
    return {"ms": ms,
            "docs": [r["source_file"] for r in rows],
            "sections": [r["section_title"] for r in rows],
            "scores": [round(r["similarity"], 4) for r in rows]}


def grep(pattern):
    t0 = time.time()
    p = subprocess.run(["grep", "-ricE", pattern, "--include=*.md", "."],
                       cwd=KB, capture_output=True, text=True)
    ms = (time.time() - t0) * 1000
    hits = []
    for line in p.stdout.splitlines():
        path, _, n = line.rpartition(":")
        try:
            count = int(n)
        except ValueError:
            continue
        if count > 0:
            hits.append((count, path.lstrip("./")))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return {"ms": ms, "docs": [h[1] for h in hits[:K]], "counts": [h[0] for h in hits[:K]]}


def score(docs, gt):
    rank = next((i + 1 for i, d in enumerate(docs) if d == gt), None)
    return {
        "hit@1": 1 if rank == 1 else 0,
        "hit@3": 1 if rank and rank <= 3 else 0,
        "hit@5": 1 if rank else 0,
        "mrr": round(1.0 / rank, 4) if rank else 0.0,
    }


def main():
    if not os.path.isdir(KB):
        print(f"KB_DIR tidak ditemukan: {KB}", file=sys.stderr)
        return 2
    if not os.path.exists(BIN):
        print(f"binary tidak ditemukan: {BIN} (jalankan go build -o ai-mop-ingestion .)", file=sys.stderr)
        return 2

    graph = {}
    if GRAPH_JSON and os.path.exists(GRAPH_JSON):
        graph = json.load(open(GRAPH_JSON))
    os.makedirs(OUT, exist_ok=True)

    methods = ["rag", "grep"] + (["graph"] if graph else [])
    rows = []
    for qid, query, gt, pattern in QUERIES:
        row = {"id": qid, "query": query, "gt": gt, "grep_pattern": pattern,
               "rag": rag(query), "grep": grep(pattern)}
        if graph:
            row["graph"] = graph.get(qid, {"docs": [], "ms": 0})
        rows.append(row)

    print(f"{'id':4} {'dokumen jawaban':50} " + " ".join(f"{m:>16}" for m in methods))
    agg = {m: [] for m in methods}
    lat = {m: [] for m in methods}
    for row in rows:
        cells = []
        for m in methods:
            met = score(row[m].get("docs", []), row["gt"])
            agg[m].append(met)
            lat[m].append(row[m].get("ms", 0))
            cells.append(f"{met['hit@1']}{met['hit@3']}{met['hit@5']}/{met['mrr']:.2f}".rjust(16))
        print(f"{row['id']:4} {row['gt'][:50]:50} " + " ".join(cells))

    print(f"\n{'metode':6} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'median ms':>10}")
    summary = {}
    for m in methods:
        n = len(agg[m])
        ms = sorted(lat[m])
        s = {
            "hit@1": sum(x["hit@1"] for x in agg[m]) / n,
            "hit@3": sum(x["hit@3"] for x in agg[m]) / n,
            "hit@5": sum(x["hit@5"] for x in agg[m]) / n,
            "mrr": sum(x["mrr"] for x in agg[m]) / n,
            "median_ms": ms[n // 2],
        }
        summary[m] = s
        print(f"{m:6} {s['hit@1']*100:5.1f}% {s['hit@3']*100:5.1f}% {s['hit@5']*100:5.1f}% "
              f"{s['mrr']:6.3f} {s['median_ms']:10.0f}")

    path = os.path.join(OUT, "result.json")
    json.dump({"rows": rows, "summary": summary}, open(path, "w"), indent=2)
    print(f"\nraw: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
