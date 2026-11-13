# ai-mop-ingestion

Aplikasi CLI Go untuk ingesting dokumen Markdown dari knowledge-base AI-MOP:
parsing Frontmatter YAML + chunking per heading H2/H3 -> embedding vektor via
Ollama (`bge-m3`, 1024 dim) -> disimpan ke PostgreSQL pgvector dengan index
HNSW cosine. Disertai sub-perintah uji pencarian similarity.

## Prasyarat

- Go 1.24+
- Ollama berjalan di `http://localhost:11434` dengan model `bge-m3`
  (embedding length 1024)
- PostgreSQL 16 + ekstensi `vector` di Docker (port host `localhost:5433`,
  user `postgres`, pass `postgres`, database `aimop_rag`)

## Konfigurasi (environment variable)

| Variable        | Default                                                       | Keterangan                |
|-----------------|---------------------------------------------------------------|---------------------------|
| `OLLAMA_URL`    | `http://localhost:11434`                                      | Endpoint Ollama           |
| `OLLAMA_MODEL`  | `bge-m3`                                                      | Model embedding           |
| `EMBEDDING_DIM` | `1024`                                                        | Dimensi vektor           |
| `PG_DSN`        | `postgres://postgres:postgres@localhost:5433/aimop_rag?sslmode=disable` | DSN PostgreSQL         |
| `AI_MOP_KB_DIR` | folder knowledge-base terdekat (runbooks dll)                 | Sumber file Markdown      |

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
#    diedarkan otomatis oleh sub-perintah ingestiton juga.
./ai-mop-ingestion init

# 2) Ingest semua file .md di folder knowledge-base (default: ai-mop knowledge-base/runbooks)
./ai-mop-ingestion ingest
./ai-mop-ingestion ingest -dir ./knowledge-base   # pilih folder lain
./ai-mop-ingestion ingest -v                      # log granular per chunk

# 3) Uji pencarian similarity dari CLI
./ai-mop-ingestion search "Mas web saya muter-muter terus" -k 3
./ai-mop-ingestion search "backup updraftplus gagal" -k 5 -json
```

Output ingest ala contoh yang diminta:

```
🗂  Basis: nickyrolly/ai-mop/knowledge-base/runbooks — 6 file .md ditemukan
🤖 Embedding: http://localhost:11434/api/embeddings via bge-m3 (dim=1024)
⏳ Processing RB-WP-001-health-check.md... Inserted 3 chunks ✅
...
⏱  Selesai: 6/6 file diproses, 26 chunk tersimpan, 0 gagal (12.4s)
```

## Arsitektur

```
internal/
├── config/       — lingkungan (env vars + default)
├── markdown/     — frontmatter YAML + chunking H2/H3, unit-test
├── embedding/    — client Ollama /api/embeddings (HTTP, []float32)
├── store/        — pgx + pgvector: skema, IngestChunks (delete+tx),
│                   Search (cosine <=>)
└── cli/          — sub-perintah init/ingest/search
```

### Chunking

- Frontmatter YAML di-parse (title,id,status,owner,tags,...).
- Body dipecah per heading `## ` atau `### ` (fenced code block awareness),
  sehingga 1 chunk = 1 topik/SOP.
- Preamble (teks sebelum heading pertama) digabung ke chunk pertama.
- Dokumen tanpa heading: satu chunk "Overview".
- Teks embedding: `[File: x.md | Section: judul]` + `\n` + isi.

### Dedup / re-ingest

`IngestChunks` menjalankan `DELETE FROM knowledge_chunks WHERE source_file=$1
AND tenant_id=$2` lalu batch INSERT dalam satu transaksi, jadi re-ingest file
yang sama tidak menumpuk duplikasi.

### Pencarian

Query di-embedding lalu:

```sql
SELECT source_file, section_title, content,
       1 - (embedding <=> $1) AS similarity
FROM knowledge_chunks
ORDER BY embedding <=> $1 ASC
LIMIT $2;
```
