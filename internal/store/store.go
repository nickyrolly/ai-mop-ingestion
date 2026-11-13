// Package store menangani koneksi PostgreSQL, inisialisasi skema pgvector,
// ingestion (upsert per file), dan pencarian similarity.
package store

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

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

// DefaultRRFK konstanta k pada Reciprocal Rank Fusion. 60 adalah nilai baku
// pada literatur (Cormack et al.) dan tidak perlu diubah untuk korpus kecil.
const DefaultRRFK = 60

// Result satu baris hasil pencarian.
type Result struct {
	ChunkID      int64 // id baris di tabel, dipakai untuk fusi dua jalur
	SourceFile   string
	DocID        string
	DocTitle     string
	SectionTitle string
	Content      string
	Category        string
	TenantID        string
	Status          string
	Confidentiality string
	Similarity      float32 // cosine similarity 0..1 (1 - cosine_distance)
	TextRank     float32 // skor ts_rank_cd jalur kata kunci; >0 berarti match teks
	Score        float32 // skor fusi RRF (makin besar makin relevan)
	UpdatedAt    time.Time
}

// TextMatched melaporkan apakah chunk cocok lewat jalur kata kunci (BM25-like).
// Dipakai untuk menandai hasil dan menaikkan peringkat di dalam gerbang skor:
// jalur kata kunci tidak boleh meloloskan chunk yang cosine-nya di bawah ambang.
func (r Result) TextMatched() bool { return r.TextRank > 0 }

// SearchOptions filter dan perilaku pencarian hybrid.
type SearchOptions struct {
	TopK           int     // jumlah hasil akhir (default 3)
	Tenant          string // kosong = semua tenant
	Category        string // kosong = semua kategori
	Status          string // kosong = semua status
	Confidentiality string // kosong = semua tingkat kerahasiaan
	Query           string // teks query untuk jalur kata kunci; kosong = vektor saja
	MinScore       float32 // ambang cosine 0..1; 0 = nonaktif
	MaxPerDoc      int     // maksimum chunk per source_file; 0 = tanpa batas
	CandidateLimit int     // kandidat per jalur sebelum fusi; 0 = topK*10 (min 50)
}

// normalized mengisi nilai default SearchOptions.
func (o SearchOptions) normalized() SearchOptions {
	if o.TopK <= 0 {
		o.TopK = 3
	}
	if o.CandidateLimit <= 0 {
		o.CandidateLimit = o.TopK * 10
		if o.CandidateLimit < 50 {
			o.CandidateLimit = 50
		}
	}
	return o
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
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS status VARCHAR(255);
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS confidentiality VARCHAR(255);
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100);

-- satu baris per index: model dan dimensi yang menghasilkan vektor di tabel.
-- Baris lama (sebelum kolom ini ada) tidak menyimpan model per chunk; kunci
-- ini yang menolak pencampuran model berdimensi sama.
CREATE TABLE IF NOT EXISTS %[1]s_meta (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    embedding_model TEXT NOT NULL,
    embedding_dim INT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- kolom turunan untuk jalur kata kunci (hybrid): judul dokumen diberi bobot A,
-- heading B, isi C. regconfig harus literal/specifik supaya ekspresinya
-- immutable (syarat generated column). 'simple' dipakai karena PostgreSQL 16
-- tidak punya stemmer bahasa Indonesia.
ALTER TABLE %[1]s ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(doc_title, '')), 'A') ||
        setweight(to_tsvector('simple', section_title), 'B') ||
        setweight(to_tsvector('simple', content), 'C')
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON %[1]s USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON %[1]s (source_file);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_category
    ON %[1]s (tenant_id, category);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON %[1]s USING gin (content_tsv);
`, s.table, s.dim)

	if _, err := s.conn.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("inisialisasi skema: %w", err)
	}
	return s.checkDim(ctx)
}

// EnsureEmbeddingModel mengunci index ke satu model dan satu dimensi.
// Index yang sudah terisi tetapi belum punya baris meta diadopsi sebagai model
// yang sedang dikonfigurasi (data lama sebelum kolom ini ada). Pergantian model
// ditolak supaya vektor yang tidak sebanding tidak tercampur.
func (s *Store) EnsureEmbeddingModel(ctx context.Context, model string) (adopted bool, err error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return false, fmt.Errorf("model embedding kosong")
	}
	var existing int
	if err := s.conn.QueryRow(ctx, fmt.Sprintf(`SELECT count(*) FROM %s`, s.table)).Scan(&existing); err != nil {
		return false, fmt.Errorf("hitung chunk: %w", err)
	}
	tag, err := s.conn.Exec(ctx, fmt.Sprintf(
		`INSERT INTO %s_meta (id, embedding_model, embedding_dim) VALUES (TRUE, $1, $2)
		 ON CONFLICT (id) DO NOTHING`, s.table), model, s.dim)
	if err != nil {
		return false, fmt.Errorf("kunci model embedding: %w", err)
	}
	var gotModel string
	var gotDim int
	err = s.conn.QueryRow(ctx, fmt.Sprintf(
		`SELECT embedding_model, embedding_dim FROM %s_meta WHERE id`, s.table)).Scan(&gotModel, &gotDim)
	if err != nil {
		return false, fmt.Errorf("baca kunci model embedding: %w", err)
	}
	if gotModel != model || gotDim != s.dim {
		return false, fmt.Errorf("index terkunci pada model %q dim=%d, konfigurasi OLLAMA_MODEL=%q EMBEDDING_DIM=%d; vektor tidak sebanding. Pakai AI_MOP_TABLE lain, atau kosongkan %s dan hapus %s_meta lalu ingest -force",
			gotModel, gotDim, model, s.dim, s.table, s.table)
	}
	var stray string
	err = s.conn.QueryRow(ctx, fmt.Sprintf(
		`SELECT embedding_model FROM %s
		 WHERE embedding_model IS NOT NULL AND embedding_model <> $1
		 LIMIT 1`, s.table), model).Scan(&stray)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return false, fmt.Errorf("cek model per chunk: %w", err)
	}
	if stray != "" {
		return false, fmt.Errorf("tabel %s masih menyimpan vektor model %q sementara konfigurasi %q; kosongkan tabel lalu ingest -force",
			s.table, stray, model)
	}
	return tag.RowsAffected() == 1 && existing > 0, nil
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
func (s *Store) IngestChunks(ctx context.Context, chunks []markdown.Chunk, embeddings [][]float32, contentHash, model string) (int, error) {
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
		(tenant_id, category, doc_id, doc_title, source_file, section_title, content, content_hash, embedding, status, confidentiality, embedding_model)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`, s.table)
	for i := range chunks {
		vec := pgvector.NewVector(embeddings[i])
		if _, err := tx.Exec(ctx, insertSQL,
			chunks[i].TenantID, chunks[i].Category, nullIfEmpty(chunks[i].DocID),
			nullIfEmpty(chunks[i].DocTitle), chunks[i].SourceFile,
			chunks[i].SectionTitle, chunks[i].Content, contentHash, vec,
			nullIfEmpty(chunks[i].Status), nullIfEmpty(chunks[i].Confidentiality),
			nullIfEmpty(model),
		); err != nil {
			return 0, fmt.Errorf("insert chunk %q: %w", chunks[i].SectionTitle, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit transaksi: %w", err)
	}
	return len(chunks), nil
}

// PruneStats jumlah chunk dokumen yang sudah tidak ada di disk.
type PruneStats struct {
	Files  int
	Chunks int
	Names  []string
}

// PruneSources menghapus chunk yang source_file-nya tidak ada di keep.
// keep kosong ditolak supaya panggilan yang salah tidak mengosongkan tabel.
func (s *Store) PruneSources(ctx context.Context, keep []string) (PruneStats, error) {
	var stats PruneStats
	if len(keep) == 0 {
		return stats, fmt.Errorf("prune ditolak: daftar file kosong")
	}
	rows, err := s.conn.Query(ctx, fmt.Sprintf(
		`DELETE FROM %s WHERE NOT (source_file = ANY($1::text[])) RETURNING source_file`, s.table), keep)
	if err != nil {
		return stats, fmt.Errorf("hapus chunk yatim: %w", err)
	}
	defer rows.Close()
	seen := map[string]int{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return stats, fmt.Errorf("scan prune: %w", err)
		}
		stats.Chunks++
		seen[name]++
	}
	if err := rows.Err(); err != nil {
		return stats, err
	}
	stats.Files = len(seen)
	stats.Names = make([]string, 0, len(seen))
	for name := range seen {
		stats.Names = append(stats.Names, name)
	}
	sort.Strings(stats.Names)
	return stats, nil
}

// Search menjalankan pencarian hybrid dua jalur:
//
//  1. vektor — cosine distance (embedding <=> query) via index HNSW;
//  2. kata kunci — ts_rank_cd atas content_tsv (judul/heading/isi berbobot).
//
// Kedua peringkat digabung dengan Reciprocal Rank Fusion, lalu hasilnya
// digerbang MinScore (khusus hit vektor) dan dibatasi MaxPerDoc chunk per
// dokumen. Fusi begini menjaga recall semantik sambil menstabilkan query yang
// memuat token pasti seperti "DISALLOW_FILE_MODS".
func (s *Store) Search(ctx context.Context, q []float32, opt SearchOptions) ([]Result, error) {
	opt = opt.normalized()

	// ef_search default pgvector adalah 40. Kandidat per jalur minimal 50,
	// jadi daftar HNSW harus minimal sebesar limit kandidat.
	ef := opt.CandidateLimit
	if ef < 40 {
		ef = 40
	}
	if ef > 1000 {
		ef = 1000
	}
	if _, err := s.conn.Exec(ctx, fmt.Sprintf("SET hnsw.ef_search = %d", ef)); err != nil {
		return nil, fmt.Errorf("set hnsw.ef_search: %w", err)
	}

	vec, err := s.vectorCandidates(ctx, q, opt)
	if err != nil {
		return nil, err
	}
	var txt []Result
	if tsq := buildTSQuery(opt.Query); tsq != "" {
		txt, err = s.textCandidates(ctx, q, opt, tsq)
		if err != nil {
			return nil, err
		}
	}
	return selectResults(fuseRRF(vec, txt, DefaultRRFK), opt), nil
}

// vectorCandidates mengambil kandidat terdekat secara cosine.
func (s *Store) vectorCandidates(ctx context.Context, q []float32, opt SearchOptions) ([]Result, error) {
	sql := fmt.Sprintf(`SELECT id, source_file, coalesce(doc_id, ''), coalesce(doc_title, ''),
			section_title, content, coalesce(category, ''), tenant_id,
			coalesce(status, ''), coalesce(confidentiality, ''),
			1 - (embedding <=> $1) AS similarity, 0::real AS text_rank, updated_at
		FROM %s
		WHERE ($2 = '' OR tenant_id = $2)
		  AND ($3 = '' OR category = $3)
		  AND ($4 = '' OR status = $4)
		  AND ($5 = '' OR confidentiality = $5)
		ORDER BY embedding <=> $1 ASC
		LIMIT $6`, s.table)

	return s.queryResults(ctx, sql,
		pgvector.NewVector(q), opt.Tenant, opt.Category, opt.Status, opt.Confidentiality, opt.CandidateLimit)
}

// textCandidates mengambil kandidat dengan ts_rank_cd. Cosine tetap dihitung
// supaya Similarity pada hasil fusi selalu bermakna (dipakai untuk gerbang
// skor dan tampilan).
func (s *Store) textCandidates(ctx context.Context, q []float32, opt SearchOptions, tsQuery string) ([]Result, error) {
	sql := fmt.Sprintf(`SELECT id, source_file, coalesce(doc_id, ''), coalesce(doc_title, ''),
			section_title, content, coalesce(category, ''), tenant_id,
			coalesce(status, ''), coalesce(confidentiality, ''),
			1 - (embedding <=> $1) AS similarity,
			ts_rank_cd(content_tsv, to_tsquery('simple', $4)) AS text_rank, updated_at
		FROM %s
		WHERE ($2 = '' OR tenant_id = $2)
		  AND ($3 = '' OR category = $3)
		  AND ($5 = '' OR status = $5)
		  AND ($6 = '' OR confidentiality = $6)
		  AND content_tsv @@ to_tsquery('simple', $4)
		ORDER BY text_rank DESC, similarity DESC
		LIMIT $7`, s.table)

	return s.queryResults(ctx, sql,
		pgvector.NewVector(q), opt.Tenant, opt.Category, tsQuery, opt.Status, opt.Confidentiality, opt.CandidateLimit)
}

// queryResults mengeksekusi query kandidat dan memetakan baris ke Result.
func (s *Store) queryResults(ctx context.Context, sql string, args ...any) ([]Result, error) {
	rows, err := s.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("query kandidat: %w", err)
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var r Result
		var sim float64
		var textRank float32
		if err := rows.Scan(&r.ChunkID, &r.SourceFile, &r.DocID, &r.DocTitle, &r.SectionTitle,
			&r.Content, &r.Category, &r.TenantID, &r.Status, &r.Confidentiality,
			&sim, &textRank, &r.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan hasil: %w", err)
		}
		r.Similarity = float32(sim)
		r.TextRank = textRank
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

// fuseRRF menggabungkan dua peringkat dengan Reciprocal Rank Fusion:
// skor(chunk) = Σ 1/(k + rank) untuk setiap jalur tempat chunk muncul. Chunk
// yang muncul di kedua jalur otomatis naik, tanpa perlu menormalkan skala
// cosine dan ts_rank_cd yang tidak sebanding.
func fuseRRF(vec, txt []Result, k int) []Result {
	if k <= 0 {
		k = DefaultRRFK
	}
	type ranked struct {
		res   Result
		score float64
	}
	items := make([]*ranked, 0, len(vec)+len(txt))
	pos := make(map[int64]int, len(vec)+len(txt))

	merge := func(list []Result) {
		for i := range list {
			r := list[i]
			j, ok := pos[r.ChunkID]
			if !ok {
				j = len(items)
				pos[r.ChunkID] = j
				items = append(items, &ranked{res: r})
			} else if r.TextMatched() {
				items[j].res.TextRank = r.TextRank
			}
			items[j].score += 1.0 / float64(k+i+1)
		}
	}
	merge(vec)
	merge(txt)

	// Urut skor fusi; seri dipecah cosine lalu ChunkID agar deterministik.
	sort.SliceStable(items, func(a, b int) bool {
		if items[a].score != items[b].score {
			return items[a].score > items[b].score
		}
		if items[a].res.Similarity != items[b].res.Similarity {
			return items[a].res.Similarity > items[b].res.Similarity
		}
		return items[a].res.ChunkID < items[b].res.ChunkID
	})

	out := make([]Result, len(items))
	for i, it := range items {
		it.res.Score = float32(it.score)
		out[i] = it.res
	}
	return out
}

// selectResults menerapkan gerbang skor, batas chunk per dokumen, dan TopK.
// Gerbang berlaku untuk kedua jalur: hit kata kunci yang cosine-nya jauh di
// bawah ambang hanya menambah derau (terbukti pada benchmark q01), sedangkan
// token pasti yang benar-benar relevan hampir selalu punya cosine memadai.
func selectResults(cands []Result, opt SearchOptions) []Result {
	out := make([]Result, 0, opt.TopK)
	perDoc := make(map[string]int, opt.TopK)
	for _, r := range cands {
		if opt.MinScore > 0 && r.Similarity < opt.MinScore {
			continue
		}
		if opt.MaxPerDoc > 0 && perDoc[r.SourceFile] >= opt.MaxPerDoc {
			continue
		}
		perDoc[r.SourceFile]++
		out = append(out, r)
		if len(out) >= opt.TopK {
			break
		}
	}
	return out
}

// stopwordsQuery dibuang dari query kata kunci: kata fungsi Indonesia muncul di
// hampir semua dokumen sehingga OR antar-term jadi kehilangan sinyal.
var stopwordsQuery = map[string]bool{
	"a": true, "ada": true, "adalah": true, "apa": true, "atau": true, "bagaimana": true,
	"bisa": true, "cara": true, "dalam": true, "dan": true, "dari": true, "dengan": true,
	"di": true, "gak": true, "gimana": true, "ini": true, "itu": true, "jadi": true,
	"juga": true, "kalau": true, "ke": true, "kenapa": true, "kok": true, "lagi": true,
	"mas": true, "mau": true, "nya": true, "pada": true, "per": true, "saya": true,
	"sebuah": true, "sudah": true, "tanpa": true, "tidak": true,
	"untuk": true, "yang": true,
}

// buildTSQuery mengubah teks query bebas menjadi tsquery OR antar-term yang
// aman (hanya huruf/angka/underscore, jadi tidak ada operator to_tsquery yang
// bisa disuntikkan). String kosong berarti jalur kata kunci dilewati.
func buildTSQuery(q string) string {
	seen := make(map[string]bool)
	var terms []string
	var word strings.Builder

	flush := func() {
		w := strings.ToLower(word.String())
		word.Reset()
		if len(w) < 2 || stopwordsQuery[w] || seen[w] {
			return
		}
		seen[w] = true
		terms = append(terms, w)
	}
	for _, r := range q {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			word.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return strings.Join(terms, " | ")
}

// nullIfEmpty mereturn nil untuk string kosong agar kolom nullable terisi NULL.
func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}
