// Package store menangani koneksi PostgreSQL, inisialisasi skema pgvector,
// ingestion (upsert per file), dan pencarian similarity.
package store

import (
	"context"
	"fmt"
	"time"

	"ai-mop-ingestion/internal/markdown"

	pgx "github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
	pgxvector "github.com/pgvector/pgvector-go/pgx"
)

// Store membungkus koneksi pgx.
type Store struct {
	conn  *pgx.Conn
	table string
	dim   int
}

// Result satu baris hasil pencarian similarity.
type Result struct {
	SourceFile   string
	SectionTitle string
	Content      string
	Similarity   float32 // cosine similarity 0..1 (1 - cosine_distance)
	CreatedAt    time.Time
}

// Connect membuka koneksi PostgreSQL.
func Connect(ctx context.Context, dsn, table string, dim int) (*Store, error) {
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

// InitSchema membuat extension vector, tabel, dan index HNSW (idempoten).
func (s *Store) InitSchema(ctx context.Context) error {
	sql := fmt.Sprintf(`
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS %s (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(50) DEFAULT 'global',
    category VARCHAR(50),
    source_file VARCHAR(255) NOT NULL,
    section_title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(%d) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON %s USING hnsw (embedding vector_cosine_ops);
`, s.table, s.dim, s.table)

	if _, err := s.conn.Exec(ctx, sql); err != nil {
		return fmt.Errorf("inisialisasi skema: %w", err)
	}
	return nil
}

// DeleteFile menghapus semua chunk lama dari satu source_file (untuk re-ingest
// tanpa duplikasi), dikembalikan jumlah chunk yang dihapus.
func (s *Store) DeleteFile(ctx context.Context, sourceFile, tenantID string) (int, error) {
	tag, err := s.conn.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE source_file = $1 AND tenant_id = $2`, s.table),
		sourceFile, tenantID,
	)
	if err != nil {
		return 0, fmt.Errorf("hapus chunk lama %s: %w", sourceFile, err)
	}
	return int(tag.RowsAffected()), nil
}

// IngestChunks menyisip chunk beserta vektornya dalam satu transaksi:
// hapus dulu chunk lama dari file yang sama, lalu insert batch.
func (s *Store) IngestChunks(ctx context.Context, chunks []markdown.Chunk, embeddings [][]float32) (int, error) {
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

	src, tenant := chunks[0].SourceFile, chunks[0].TenantID

	// 1. Bersihkan chunk lama dari file yang sama.
	if _, err := tx.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE source_file = $1 AND tenant_id = $2`, s.table),
		src, tenant); err != nil {
		return 0, fmt.Errorf("hapus chunk lama %s: %w", src, err)
	}

	// 2. Insert batch (tanpa prepared statement — loop Exec di dalam tx).
	insertSQL := fmt.Sprintf(`INSERT INTO %s (tenant_id, category, source_file, section_title, content, embedding)
		VALUES ($1, $2, $3, $4, $5, $6)`, s.table)
	for i := range chunks {
		vec := pgvector.NewVector(embeddings[i])
		if _, err := tx.Exec(ctx,
			insertSQL,
			chunks[i].TenantID, chunks[i].Category, chunks[i].SourceFile,
			chunks[i].SectionTitle, chunks[i].Content, vec,
		); err != nil {
			return 0, fmt.Errorf("insert chunk %q: %w", chunks[i].SectionTitle, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit transaction: %w", err)
	}
	return len(chunks), nil
}

// Search mereturn top-k chunk paling mirip dengan vektor query (cosine).
func (s *Store) Search(ctx context.Context, q []float32, topK int) ([]Result, error) {
	vec := pgvector.NewVector(q)

	sql := fmt.Sprintf(`SELECT source_file, section_title, content,
		1 - (embedding <=> $1) AS similarity, created_at
		FROM %s
		ORDER BY embedding <=> $1 ASC
		LIMIT $2`, s.table)

	rows, err := s.conn.Query(ctx, sql, vec, topK)
	if err != nil {
		return nil, fmt.Errorf("query similarity: %w", err)
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var r Result
		var sim float64
		if err := rows.Scan(&r.SourceFile, &r.SectionTitle, &r.Content, &sim, &r.CreatedAt); err != nil {
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

// Count mereturn jumlah chunk untuk source_file tertentu (untuk logging re-ingest).
func (s *Store) Count(ctx context.Context, sourceFile, tenantID string) (int, error) {
	var n int
	err := s.conn.QueryRow(ctx,
		fmt.Sprintf(`SELECT COUNT(*) FROM %s WHERE source_file = $1 AND tenant_id = $2`, s.table),
		sourceFile, tenantID).Scan(&n)
	return n, err
}
