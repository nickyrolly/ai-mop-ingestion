# ai-mop-ingestion

Aplikasi CLI Go untuk ingesting dokumen Markdown dari knowledge-base AI-MOP:
parsing frontmatter YAML + chunking per heading H2/H3 → embedding vektor via
Ollama (`bge-m3`, 1024 dim, batch `/api/embed`) → disimpan ke PostgreSQL
pgvector dengan index HNSW cosine. Pencarian memakai **hybrid retrieval**:
kandidat vektor (cosine) + kandidat kata kunci (`ts_rank_cd`) digabung dengan
Reciprocal Rank Fusion, disaring gerbang skor, dan dibatasi per dokumen.

## Prasyarat

- Go 1.24+
- Ollama berjalan di `http://localhost:11434` dengan model `bge-m3`
  (embedding length 1024, endpoint `/api/embed`)
- PostgreSQL 16 + ekstensi `vector` di Docker (port host `localhost:5433`,
  user `postgres`, pass `postgres`, database `aimop_rag`)

## Konfigurasi (environment variable)

| Variable        | Default                                                       | Keterangan                   |
|-----------------|---------------------------------------------------------------|------------------------------|
| `OLLAMA_URL`    | `http://localhost:11434`                                      | Base URL Ollama              |
| `OLLAMA_MODEL`  | `bge-m3`                                                      | Model embedding              |
| `EMBEDDING_DIM` | `1024`                                                        | Dimensi vektor (divalidasi)  |
| `PG_DSN`        | `postgres://postgres:postgres@localhost:5433/aimop_rag?sslmode=disable` | DSN PostgreSQL      |
| `AI_MOP_KB_DIR` | folder knowledge-base terdekat                                 | Sumber file Markdown         |
| `AI_MOP_TABLE`  | `knowledge_chunks`                                             | Nama tabel chunk             |
| `AI_MOP_JOBS`   | `4`                                                            | Request embedding paralel    |

## Build & test

```sh
go mod tidy
go build -o ai-mop-ingestion .
go vet ./...
go test ./...
```

## Penggunaan

```sh
# 1) Inisialisasi skema (table knowledge_chunks + index HNSW) — idempoten,
#    juga dijalankan otomatis oleh sub-perintah ingest.
./ai-mop-ingestion init

# 2) Ingest rekursif semua *.md di knowledge-base (runbooks/, learnings/,
#    vendors/, patterns/, onboarding/, ...)
./ai-mop-ingestion ingest
./ai-mop-ingestion ingest -dir ../ai-mop/knowledge-base
./ai-mop-ingestion ingest -dir ../ai-mop/knowledge-base/runbooks/RB-WP-001-health-check.md
./ai-mop-ingestion ingest -v          # log per chunk (kategori, panjang, truncate)
./ai-mop-ingestion ingest -force      # paksa re-embed walau file tidak berubah

# 3) Pencarian hybrid (vektor + kata kunci, RRF, gerbang skor)
./ai-mop-ingestion search "Mas web saya muter-muter terus" -k 3
./ai-mop-ingestion search "backup updraftplus gagal" -k 5 -json
./ai-mop-ingestion search "harga maintenance" -k 3 -category learning -tenant internal

# Ambang skor: default 0.50. Turunkan bila KB memakai istilah yang sangat khas,
# matikan dengan 0 (kembali ke perilaku vektor-only + tanpa gerbang).
./ai-mop-ingestion search "custom domain cloudflare" -min-score 0.45
./ai-mop-ingestion search "custom domain cloudflare" -min-score 0

# Diversitas konteks: default maks 2 chunk per dokumen; 1 = 5 dokumen berbeda
# di top-5 (tidak menurunkan presisi pada benchmark eval/).
./ai-mop-ingestion search "cek sertifikat TLS expired" -per-doc 1

# Keluar dengan kode 3 bila tidak ada hasil yang lolos ambang:
# query di luar domain tidak lagi dipaksa mengembalikan top-1.
./ai-mop-ingestion search "resep nasi goreng spesial"; echo $?   # → 3
```

| Kode keluar | Arti |
|---|---|
| 0 | ada hasil yang lolos gerbang skor |
| 1 | kesalahan runtime (DB, Ollama, parse) |
| 2 | argumen salah |
| 3 | tidak ada hasil relevan (skor semua kandidat < `-min-score`) |
| 130 | dibatalkan (Ctrl+C) |

Contoh output:

```
🗂  Basis: /path/knowledge-base — 46 file .md ditemukan (rekursif)
🤖 Embedding: http://localhost:11434/api/embed via bge-m3 (dim=1024, paralel=4)
⏳ Processing runbooks/RB-WP-001-health-check.md... Inserted 17 chunks ✅ (RB-WP-001)
⏳ Processing learnings/2026-11-12-rag-architecture....md... ⏭️  skip (isi tidak berubah)
⏱  46 file diproses, 45 skip, 0 gagal — 406 chunk tersimpan, 17 chunk di-embed dalam 1.0s
```

## Arsitektur

```
internal/
├── config/       — lingkungan (env vars + default)
├── markdown/     — frontmatter YAML, chunking H2/H3 + breadcrumb, unit-test
├── embedding/    — client Ollama /api/embed (batch, []float32)
├── store/        — pgx + pgvector: skema & migrasi, IngestChunks (delete+tx),
│                   Search hybrid (vektor + ts_rank_cd → RRF, gerbang skor,
│                   batas chunk per dokumen), unit-test fungsi fusi
└── cli/          — sub-perintah init/ingest/search
```

### Chunking

- Frontmatter YAML di-parse (title, id, status, owner, tenant, tags, ...).
  Bila YAML tidak valid (mis. nilai mengandung `": "`), metadata dibaca dengan
  parsing longgar dan dokumen **tetap** di-ingest (peringatan ditampilkan).
- Body dipecah per heading `## ` atau `### ` (fenced code block awareness).
- Chunk H3 menyimpan breadcrumb induknya: `Steps > 1. DNS`, sehingga chunk
  tidak kehilangan konteks runbook/dokumen asal.
- Preamble (teks sebelum heading pertama) digabung ke chunk pertama.
- Dokumen tanpa heading: satu chunk dengan judul dokumen.
- Teks embedding: `[File: <rel-path> | Doc: <judul> (<id>) | Section: <breadcrumb>]`
  + `\n` + isi section.
- Chunk yang lebih panjang dari `MaxChunkChars` dipotong di batas rune UTF-8
  dan ditandai `...[truncated]` (ingest `-v` memberi peringatan).

### Identitas dokumen & re-ingest

- Kunci dokumen adalah **path relatif terhadap root knowledge-base**
  (`runbooks/RB-WP-001-health-check.md`), bukan basename → file `index.md` di
  folder berbeda tidak saling menimpa.
- `content_hash` (SHA-256 isi file) disimpan. Ingest berikutnya melewati file
  yang isinya tidak berubah (tanpa panggilan embedding); pakai `-force` untuk
  memaksa.
- `IngestChunks` menjalankan `DELETE FROM <table> WHERE source_file=$1` lalu
  batch INSERT dalam satu transaksi, jadi re-ingest tidak menumpuk duplikasi
  dan tidak meninggalkan chunk yatim bila tenant berubah.

### Skema

```sql
CREATE TABLE knowledge_chunks (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'global',
    category VARCHAR(50),
    doc_id VARCHAR(100),
    doc_title VARCHAR(255),
    source_file VARCHAR(512) NOT NULL,
    section_title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    content_hash CHAR(64),
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- kolom turunan untuk jalur kata kunci (bobot: judul A, heading B, isi C)
    content_tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(doc_title, '')), 'A') ||
        setweight(to_tsvector('simple', section_title), 'B') ||
        setweight(to_tsvector('simple', content), 'C')
    ) STORED
);
CREATE INDEX idx_chunks_embedding ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_source ON knowledge_chunks (source_file);
CREATE INDEX idx_chunks_tenant_category ON knowledge_chunks (tenant_id, category);
CREATE INDEX idx_chunks_tsv ON knowledge_chunks USING gin (content_tsv);
```

Tabel lama dari v0.1.0/v0.2.0 dimigrasikan otomatis (`ALTER TABLE ... ADD COLUMN
IF NOT EXISTS`, termasuk `content_tsv` + index GIN-nya). Dimensi kolom
`embedding` divalidasi terhadap `EMBEDDING_DIM`; bila tidak cocok, aplikasi
berhenti dengan pesan jelas (bukan gagal INSERT). PostgreSQL 16 tidak punya
stemmer bahasa Indonesia, jadi konfigurasi teks yang dipakai `simple` (tanpa
stemming) — token dicocokkan apa adanya, dan identifier seperti
`DISALLOW_FILE_MODS` tetap utuh.

### Pencarian

Dua jalur kandidat, masing-masing `TopK × 10` baris (minimum 50):

```sql
-- 1) vektor: cosine via index HNSW
SELECT id, ..., 1 - (embedding <=> $1) AS similarity
FROM knowledge_chunks
WHERE ($2 = '' OR tenant_id = $2) AND ($3 = '' OR category = $3)
ORDER BY embedding <=> $1 ASC LIMIT $4;

-- 2) kata kunci: ts_rank_cd atas judul/heading/isi
SELECT id, ..., ts_rank_cd(content_tsv, to_tsquery('simple', $4)) AS text_rank
FROM knowledge_chunks
WHERE ($2 = '' OR tenant_id = $2) AND ($3 = '' OR category = $3)
  AND content_tsv @@ to_tsquery('simple', $4)
ORDER BY text_rank DESC, similarity DESC LIMIT $5;
```

Kemudian di Go:

1. **RRF** — `skor = Σ 1/(60 + rank)` untuk setiap jalur; chunk yang muncul di
   dua jalur otomatis naik tanpa perlu menormalkan skala cosine vs ts_rank_cd.
2. **Gerbang skor** — hasil dengan cosine < `-min-score` dibuang, termasuk hit
   jalur kata kunci (terbukti di benchmark: hit teks ber-cosine rendah justru
   derau).
3. **Batas per dokumen** — maksimum `-per-doc` chunk per `source_file`.
4. Ambil `TopK` pertama; kosong → keluar dengan kode 3.

Query kata kunci dibangun dari token huruf/angka/underscore (kata fungsi
Indonesia dibuang), di-OR, sehingga tidak ada operator `to_tsquery` yang bisa
disuntikkan dari input pengguna.

## Benchmark presisi

Harness: `eval/bench.py` (14 query operasional, k=5, metrik level dokumen).
Angka v0.3.0 vs v0.2.0 pada KB yang sama:

| Metode | Hit@1 | Hit@3 | Hit@5 | MRR | median | dok unik/top-5 |
|---|---|---|---|---|---|---|
| hybrid + gerbang + per-doc 2 (v0.3.0) | 92.9% | 100% | 100% | 0.952 | ~68 ms | 3.0 |
| hybrid, gerbang & per-doc dimatikan | 92.9% | 92.9% | 92.9% | 0.929 | ~67 ms | 5.0 |
| vektor saja (v0.2.0) | 92.9% | 100% | 100% | 0.952 | ~62 ms | 2.0 |
| grep markdown | 71.4% | 92.9% | 100% | 0.839 | ~14 ms | 5.0 |
| `search_graph` (nama heading) | 50.0% | 57.1% | 78.6% | 0.586 | ~22 ms | 4.5 |

Perbaikan yang bisa dibuktikan benchmark: query di luar domain sekarang
menghasilkan **0 hasil + exit 3** (sebelumnya 5 dokumen dengan cosine 0.41–0.48),
dan jumlah dokumen berbeda di top-5 naik dari 2.0 → 3.0 (`-per-doc 1` → 5.0).
Presisi Hit@1/MRR tidak berubah karena dua kegagalan tersisa (q01) adalah
near-tie semantik (0.5241 vs 0.5178), bukan masalah cakupan kandidat.

## Catatan

- Binary hasil build (`./ai-mop-ingestion`) tidak di-commit (lihat `.gitignore`).
- Ingest bisa dihentikan dengan `Ctrl+C`; transaksi file yang sedang berjalan
  di-rollback dan proses keluar dengan rapi.
- Bila `EMBEDDING_DIM` berubah, pakai `AI_MOP_TABLE` baru atau rebuild tabel.
