// Package store menangani koneksi PostgreSQL, inisialisasi skema pgvector,
// ingestion (upsert per file), dan pencarian similarity.
package store

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"ai-mop-ingestion/internal/markdown"

	pgx "github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
	pgxvector "github.com/pgvector/pgvector-go/pgx"
)

// tableNameRe membatasi nama tabel yang boleh diinterpolasi ke SQL
// (nama tabel tidak bisa diparametrisasi seperti nilai biasa).
var tableNameRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Store membungkus koneksi pgx.
type Store struct {
	conn  *pgx.Conn
	table string
	dim   int
}

// Result satu baris hasil pencarian similarity.
type Result struct {
	SourceFile   string
	DocID        string
	DocTitle     string
	SectionTitle string
	Content      string
	Category     string
	TenantID     string
	Similarity   float32 // cosine similarity 0..1 (1 - cosine_distance)
	UpdatedAt    time.Time
}

// SearchOptions filter opsional untuk pencarian.
type SearchOptions struct {
	TopK     int
	Tenant   string // kosong = semua tenant
	Category string // kosong = semua kategori
}

// FileState ringkasan kondisi chunk satu dokumen di database.
type FileState struct {
	ContentHash string // hash isi file saat terakhir di-ingest
	Chunks      int
}

// Connect membuka koneksi PostgreSQL.
func Connect(ctx context.Context, dsn, table string, dim int) (*Store, error) {
	if !tableNameRe.MatchString(table) {
		return nil, fmt.Errorf("nama tabel tidak valid: %q", table)
	}
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("koneksi PostgreSQL gagal: %w", err)
	}
	// Daftarkan codec vector ke pgtype agar bisa encode/decode kolom vector.
	if err := pgxvector.RegisterTypes(ctx, conn); err != nil {
		conn.Close(context.Background())
		return nil, fmt.Errorf("register pgvector type: %w", err)
	}
	return &Store{conn: conn, table: table, dim: dim}, nil
}

// Close menutup koneksi.
func (s *Store) Close() {
	s.conn.Close(context.Background())
}

// InitSchema membuat extension vector, tabel, dan index HNSW (idempoten), lalu
// memvalidasi dimensi kolom embedding.
func (s *Store) InitSchema(ctx context.Context) error {
	ddl := fmt.Sprintf(`
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS %[1]s (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'global',
    category VARCHAR(50),
    doc_id VARCHAR(100),
    doc_title VARCHAR(255),
    source_file VARCHAR(512) NOT NULL,
    section_title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    content_hash CHAR(64),
    embedding vector(%[2]d) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- migrasi untuk tabel lama (versi v0.1.0)
ALTER TABLE %[1]s ALTER COLUMN source_file TYPE VARCHAR(512);
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS doc_id VARCHAR(100);
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS doc_title VARCHAR(255);
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS content_hash CHAR(64);
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON %[1]s USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON %[1]s (source_file);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_category
    ON %[1]s (tenant_id, category);
`, s.table, s.dim)

	if _, err := s.conn.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("inisialisasi skema: %w", err)
	}
	return s.checkDim(ctx)
}

// checkDim memastikan dimensi kolom embedding sama dengan konfigurasi. Tanpa
// ini, perubahan EMBEDDING_DIM hanya akan gagal saat INSERT dengan pesan samar.
func (s *Store) checkDim(ctx context.Context) error {
	var typ string
	err := s.conn.QueryRow(ctx, `
		SELECT format_type(atttypid, atttypmod)
		FROM pg_attribute
		WHERE attrelid = $1::regclass AND attname = 'embedding'`, s.table).Scan(&typ)
	if err != nil {
		return fmt.Errorf("cek dimensi kolom embedding: %w", err)
	}
	want := fmt.Sprintf("vector(%d)", s.dim)
	if typ != want {
		return fmt.Errorf("dimensi embedding tidak cocok: tabel %q punya %s, konfigurasi EMBEDDING_DIM=%d (%s); "+
			"pakai nama tabel lain (AI_MOP_TABLE) atau rebuild tabel", s.table, typ, s.dim, want)
	}
	return nil
}

// FileState mereturn hash dan jumlah chunk terakhir untuk satu dokumen.
func (s *Store) FileState(ctx context.Context, sourceFile string) (FileState, error) {
	var st FileState
	var hash *string
	err := s.conn.QueryRow(ctx, fmt.Sprintf(
		`SELECT max(NULLIF(content_hash, '')), count(*) FROM %s WHERE source_file = $1`, s.table),
		sourceFile).Scan(&hash, &st.Chunks)
	if err != nil {
		return st, fmt.Errorf("baca state %s: %w", sourceFile, err)
	}
	if hash != nil {
		st.ContentHash = *hash
	}
	return st, nil
}

// IngestChunks menyisip chunk beserta vektornya dalam satu transaksi: hapus
// dulu semua chunk lama dari file yang sama, lalu insert batch. contentHash
// disimpan agar ingest berikutnya bisa melewati file yang tidak berubah.
func (s *Store) IngestChunks(ctx context.Context, chunks []markdown.Chunk, embeddings [][]float32, contentHash string) (int, error) {
	if len(chunks) == 0 {
		return 0, nil
	}
	if len(chunks) != len(embeddings) {
		return 0, fmt.Errorf("chunks (%d) dan embeddings (%d) tidak seukuran",
			len(chunks), len(embeddings))
	}

	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("begin transaksi: %w", err)
	}
	defer tx.Rollback(context.Background())

	src := chunks[0].SourceFile

	// 1. Bersihkan chunk lama dari file yang sama (tanpa filter tenant, supaya
	//    perubahan tenant di frontmatter tidak meninggalkan chunk yatim).
	if _, err := tx.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE source_file = $1`, s.table), src); err != nil {
		return 0, fmt.Errorf("hapus chunk lama %s: %w", src, err)
	}

	// 2. Insert batch.
	insertSQL := fmt.Sprintf(`INSERT INTO %s
		(tenant_id, category, doc_id, doc_title, source_file, section_title, content, content_hash, embedding)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`, s.table)
	for i := range chunks {
		vec := pgvector.NewVector(embeddings[i])
		if _, err := tx.Exec(ctx, insertSQL,
			chunks[i].TenantID, chunks[i].Category, nullIfEmpty(chunks[i].DocID),
			nullIfEmpty(chunks[i].DocTitle), chunks[i].SourceFile,
			chunks[i].SectionTitle, chunks[i].Content, contentHash, vec,
		); err != nil {
			return 0, fmt.Errorf("insert chunk %q: %w", chunks[i].SectionTitle, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit transaksi: %w", err)
	}
	return len(chunks), nil
}

// Search mereturn top-k chunk paling mirip dengan vektor query (cosine).
func (s *Store) Search(ctx context.Context, q []float32, opt SearchOptions) ([]Result, error) {
	if opt.TopK <= 0 {
		opt.TopK = 3
	}
	vec := pgvector.NewVector(q)

	sql := fmt.Sprintf(`SELECT source_file, coalesce(doc_id, ''), coalesce(doc_title, ''),
			section_title, content, coalesce(category, ''), tenant_id,
			1 - (embedding <=> $1) AS similarity, updated_at
		FROM %s
		WHERE ($2 = '' OR tenant_id = $2)
		  AND ($3 = '' OR category = $3)
		ORDER BY embedding <=> $1 ASC
		LIMIT $4`, s.table)

	rows, err := s.conn.Query(ctx, sql, vec, opt.Tenant, opt.Category, opt.TopK)
	if err != nil {
		return nil, fmt.Errorf("query similarity: %w", err)
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var r Result
		var sim float64
		if err := rows.Scan(&r.SourceFile, &r.DocID, &r.DocTitle, &r.SectionTitle,
			&r.Content, &r.Category, &r.TenantID, &sim, &r.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan hasil: %w", err)
		}
		r.Similarity = float32(sim)
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

// nullIfEmpty mereturn nil untuk string kosong agar kolom nullable terisi NULL.
func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}
