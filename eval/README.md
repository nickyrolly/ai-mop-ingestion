# Eval retrieval

`bench.py` membandingkan tiga cara menjawab pertanyaan operasional dari
knowledge-base AI-MOP:

| Metode  | Cara kerja | Butuh |
|---------|------------|-------|
| `rag`   | `ai-mop-ingestion search "<query>" -k 5` — embedding `bge-m3` + pgvector cosine | Ollama + PostgreSQL |
| `grep`  | `grep -ricE "<kata kunci query>"` di folder knowledge-base, urut jumlah match | hanya file Markdown |
| `graph` | `codebase-memory-mcp search_graph` (pola nama/heading) — hasilnya disimpan sebagai `graph.json` | index codebase-memory |

Metrik: **Hit@1 / Hit@3 / Hit@5** dan **MRR** pada level dokumen (apakah
dokumen yang berisi jawaban muncul di posisi ke-1/3/5), plus latensi.

## Menjalankan

```sh
go build -o ai-mop-ingestion .
KB_DIR=../ai-mop/knowledge-base ./eval/bench.py
```

Hasil mentah: `eval/out/result.json`.

Argumen tambahan untuk `search` bisa dioper lewat `RAG_ARGS`, misalnya untuk
ablation (mematikan gerbang skor dan batas per dokumen):

```sh
RAG_ARGS="-min-score 0 -per-doc 0" KB_DIR=../ai-mop/knowledge-base ./eval/bench.py
RAG_ARGS="-per-doc 1"             KB_DIR=../ai-mop/knowledge-base ./eval/bench.py
```

Untuk ikut mengukur graph, buat `graph.json` dengan bentuk berikut (diisi dari
hasil `search_graph` per query), lalu:

```sh
KB_DIR=../ai-mop/knowledge-base GRAPH_JSON=/tmp/graph.json ./eval/bench.py
```

```json
{
  "q01": { "ms": 34, "docs": ["runbooks/RB-WP-001-health-check.md", "vendors/..."] },
  "q02": { "ms": 19, "docs": ["learnings/2026-11-03-seo-recovery-pn-koba.md"] }
}
```

## Menambah query

Edit `QUERIES` di `bench.py`: `(id, query, dokumen_jawaban, pola_grep)`.
Pola grep sebaiknya hanya berisi kata kunci yang benar-benar ada di query —
itu yang akan dipakai manusia/agent saat mengandalkan pencarian literal.

## Batasan yang perlu diingat

- Ground truth = **satu** dokumen jawaban; jawaban yang tersebar di beberapa
  dokumen tidak dihargai.
- Baseline `graph` hanya mencocokkan **nama node** (heading section, simbol
  kode), bukan isi paragraf — pertanyaan yang jawabannya hanya ada di body
  dokumen akan gagal secara struktural.
- `grep` menilai relevansi dari jumlah kemunculan kata kunci, sehingga dokumen
  panjang cenderung menang.
- Corpus kecil (46 dokumen / 406 chunk), jadi angka ini indikatif, bukan
  benchmark statistik.
- Query di luar domain tidak ada di daftar ini; uji manual yang dipakai:
  `search "resep nasi goreng spesial"` harus keluar dengan kode 3.

## Hasil rujukan (v0.3.0)

| Konfigurasi | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| hybrid + gerbang 0.50 + `-per-doc 2` | 92.9% | 100% | 100% | 0.952 |
| hybrid, `-min-score 0 -per-doc 0` | 92.9% | 92.9% | 92.9% | 0.929 |
| vektor saja (v0.2.0) | 92.9% | 100% | 100% | 0.952 |

Eksperimen `RAG_ARGS` penting: jalur kata kunci tanpa gerbang skor justru
menurunkan Hit@5, karena chunk dengan cosine rendah tapi banyak kata kunci umum
mendesak hasil yang benar keluar dari top-5 (kasus q01).
