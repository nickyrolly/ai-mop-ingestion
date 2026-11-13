# ai-mop-ingestion

Aplikasi CLI Go untuk ingesting dokumen Markdown dari knowledge-base AI-MOP:
parsing frontmatter YAML + chunking per heading H2/H3 → embedding vektor via
Ollama (`bge-m3`, 1024 dim, batch `/api/embed`) → disimpan ke PostgreSQL
pgvector dengan index HNSW cosine. Disertai sub-perintah uji pencarian
similarity.

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

# 3) Uji pencarian similarity
./ai-mop-ingestion search "Mas web saya muter-muter terus" -k 3
./ai-mop-ingestion search "backup updraftplus gagal" -k 5 -json
./ai-mop-ingestion search "harga maintenance" -k 3 -category learning -tenant internal
```

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
│                   Search (cosine <=> dengan filter tenant/kategori)
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
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_chunks_embedding ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_source ON knowledge_chunks (source_file);
CREATE INDEX idx_chunks_tenant_category ON knowledge_chunks (tenant_id, category);
```

Tabel lama dari v0.1.0 dimigrasikan otomatis (`ALTER TABLE ... ADD COLUMN IF
NOT EXISTS`). Dimensi kolom `embedding` divalidasi terhadap `EMBEDDING_DIM`;
bila tidak cocok, aplikasi berhenti dengan pesan jelas (bukan gagal INSERT).

### Pencarian

```sql
SELECT source_file, doc_id, doc_title, section_title, content, category, tenant_id,
       1 - (embedding <=> $1) AS similarity, updated_at
FROM knowledge_chunks
WHERE ($2 = '' OR tenant_id = $2)
  AND ($3 = '' OR category = $3)
ORDER BY embedding <=> $1 ASC
LIMIT $4;
```

## Catatan

- Binary hasil build (`./ai-mop-ingestion`) tidak di-commit (lihat `.gitignore`).
- Ingest bisa dihentikan dengan `Ctrl+C`; transaksi file yang sedang berjalan
  di-rollback dan proses keluar dengan rapi.
- Bila `EMBEDDING_DIM` berubah, pakai `AI_MOP_TABLE` baru atau rebuild tabel.
