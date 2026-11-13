// Package cli menerjemahkan argumen program menjadi sub-perintah
// init / ingest / search.
package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"ai-mop-ingestion/internal/config"
	"ai-mop-ingestion/internal/embedding"
	"ai-mop-ingestion/internal/markdown"
	"ai-mop-ingestion/internal/store"
)

const appVersion = "v0.2.0"

// Run mengeksekusi CLI dan mereturn kode keluar Unix; args tanpa nama program.
func Run(ctx context.Context, args []string, out io.Writer) int {
	if len(args) == 0 {
		printUsage(out)
		return 2
	}
	cfg := config.FromEnv()

	switch args[0] {
	case "version", "--version", "-v":
		fmt.Fprintf(out, "ai-mop-ingestion %s\n", appVersion)
		return 0
	case "help", "--help", "-h":
		printUsage(out)
		return 0
	case "init":
		return runInit(ctx, cfg, out)
	case "ingest":
		return runIngest(ctx, cfg, args[1:], out)
	case "search":
		return runSearch(ctx, cfg, args[1:], out)
	default:
		fmt.Fprintf(out, "error: sub-perintah tidak dikenal: %q\n", args[0])
		printUsage(out)
		return 2
	}
}

func printUsage(w io.Writer) {
	fmt.Fprint(w, `ai-mop-ingestion — ingestion Markdown ke PostgreSQL pgvector via Ollama (bge-m3)

Penggunaan:
  ai-mop-ingestion init [-dir .]                       Inisialisasi skema (tabel + index HNSW)
  ai-mop-ingestion ingest [-dir PATH] [-v] [-force]    Ingest rekursif semua *.md
  ai-mop-ingestion search "query" [-k 3] [-json]       Verifikasi hasil ingestion
                     [-tenant ID] [-category NAMA]
  ai-mop-ingestion version                             Tampilkan versi

Opsi ingest:
  -dir PATH     folder atau satu file .md (default: knowledge-base terdekat)
  -v            log per chunk (judul, kategori, panjang teks)
  -force        re-embed walau isi file tidak berubah

Konfigurasi environment variable:
  OLLAMA_URL      (default: http://localhost:11434)
  OLLAMA_MODEL    (default: bge-m3)
  EMBEDDING_DIM   (default: 1024)
  PG_DSN          (default: postgres://postgres:postgres@localhost:5433/aimop_rag?sslmode=disable)
  AI_MOP_KB_DIR   (default: folder knowledge-base terdekat)
  AI_MOP_TABLE    (default: knowledge_chunks)
  AI_MOP_JOBS     (default: 4, jumlah request embedding paralel)
`)
}

func fatalf(w io.Writer, format string, args ...any) int {
	fmt.Fprintf(w, "error: "+format+"\n", args...)
	return 1
}

// kbDirCandidates diprobe satu per satu bila tidak ada -dir / AI_MOP_KB_DIR.
var kbDirCandidates = []string{
	"nickyrolly/ai-mop/knowledge-base", // layout sesi (repo di github.com/nickyrolly/ai-mop)
	"../ai-mop/knowledge-base",         // path dari folder ai-mop-ingestion
	"knowledge-base",                   // bila dijalankan dari root repo
	"nickyrolly/ai-mop/knowledge-base/runbooks",
	"../ai-mop/knowledge-base/runbooks",
	"knowledge-base/runbooks",
}

func resolveKBDir(flagDir, envDir string) (string, error) {
	cands := []string{}
	if flagDir != "" {
		cands = append(cands, flagDir)
	}
	if envDir != "" {
		cands = append(cands, envDir)
	}
	cands = append(cands, kbDirCandidates...)

	for _, c := range cands {
		if _, err := os.Stat(c); err == nil {
			abs, _ := filepath.Abs(c)
			return abs, nil
		}
	}
	return "", fmt.Errorf("folder knowledge-base tidak ditemukan; gunakan -dir PATH atau set AI_MOP_KB_DIR")
}

// collectMarkdownFiles mengumpulkan *.md secara rekursif (melewati file/folder
// tersembunyi), atau satu file bila root menunjuk ke file .md.
func collectMarkdownFiles(root string) ([]string, error) {
	st, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("baca %s: %w", root, err)
	}
	if !st.IsDir() {
		if strings.HasSuffix(strings.ToLower(root), ".md") {
			return []string{root}, nil
		}
		return nil, fmt.Errorf("%s bukan folder dan bukan file .md", root)
	}

	var files []string
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if p != root && strings.HasPrefix(name, ".") {
				return fs.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(name, ".") {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(name), ".md") {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("telusuri %s: %w", root, err)
	}
	sort.Strings(files)
	return files, nil
}

// sourceID mereturn identitas dokumen: path relatif terhadap root KB dengan
// separator '/' (kunci dedup, tahan terhadap file bernama sama di folder beda).
func sourceID(root, file string) string {
	rel, err := filepath.Rel(root, file)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
		return filepath.ToSlash(filepath.Base(file))
	}
	return filepath.ToSlash(rel)
}

// findKBRoot mencari root knowledge-base untuk mode satu file, supaya kunci
// dokumennya tetap sama dengan hasil ingest folder penuh (tidak memicu baris
// duplikat dan kategorinya tidak jatuh ke "uncategorized").
func findKBRoot(file string) string {
	dir := filepath.Dir(file)
	// 1. folder yang namanya knowledge-base
	for p := dir; ; p = filepath.Dir(p) {
		if strings.EqualFold(filepath.Base(p), "knowledge-base") {
			return p
		}
		if parent := filepath.Dir(p); parent == p {
			break
		}
	}
	// 2. folder yang punya subfolder standar KB
	kbSubdirs := []string{"runbooks", "learnings", "vendors", "patterns", "clients"}
	for p := dir; ; p = filepath.Dir(p) {
		for _, sub := range kbSubdirs {
			if st, err := os.Stat(filepath.Join(p, sub)); err == nil && st.IsDir() {
				return p
			}
		}
		if parent := filepath.Dir(p); parent == p {
			break
		}
	}
	return dir
}

// --- init -------------------------------------------------------------------

func runInit(ctx context.Context, cfg config.Config, w io.Writer) int {
	st, err := store.Connect(ctx, cfg.PGDSN, cfg.SQLTable, cfg.EmbeddingDim)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	defer st.Close()
	if err := st.InitSchema(ctx); err != nil {
		return fatalf(w, "%v", err)
	}
	fmt.Fprintf(w, "✅ Skema siap: table %q (embedding vector(%d) + index hnsw)\n", cfg.SQLTable, cfg.EmbeddingDim)
	return 0
}

// --- ingest -----------------------------------------------------------------

type ingestStats struct {
	files    int
	skipped  int
	failed   int
	chunks   int
	embedded int
	start    time.Time
}

func runIngest(ctx context.Context, cfg config.Config, args []string, w io.Writer) int {
	fs := flag.NewFlagSet("ingest", flag.ContinueOnError)
	fs.SetOutput(w)
	dir := fs.String("dir", "", "folder atau file *.md")
	verbose := fs.Bool("v", false, "log per chunk")
	force := fs.Bool("force", false, "re-embed walau isi file tidak berubah")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	kbDir, err := resolveKBDir(*dir, cfg.KBDir)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	files, err := collectMarkdownFiles(kbDir)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	// Mode satu file: pakai root KB sebenarnya agar kunci dokumen konsisten.
	root := kbDir
	if st, statErr := os.Stat(kbDir); statErr == nil && !st.IsDir() {
		root = findKBRoot(kbDir)
	}

	st, err := store.Connect(ctx, cfg.PGDSN, cfg.SQLTable, cfg.EmbeddingDim)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	defer st.Close()

	// Skema diinisialisasi otomatis saat aplikasi dijalankan pertama kali.
	if err := st.InitSchema(ctx); err != nil {
		return fatalf(w, "%v", err)
	}

	emb := embedding.NewClient(cfg.OllamaURL, cfg.OllamaModel, cfg.EmbeddingDim)

	fmt.Fprintf(w, "🗂  Basis: %s — %d file .md ditemukan (rekursif)\n", kbDir, len(files))
	fmt.Fprintf(w, "🤖 Embedding: %s via %s (dim=%d, paralel=%d)\n",
		emb.Endpoint, cfg.OllamaModel, cfg.EmbeddingDim, cfg.Jobs)

	stats := ingestStats{start: time.Now()}
	for _, path := range files {
		if ctx.Err() != nil {
			fmt.Fprintln(w, "⚠️  dibatalkan oleh sinyal/timeout")
			break
		}
		id := sourceID(root, path)
		fmt.Fprintf(w, "⏳ Processing %s... ", id)

		res, err := ingestOne(ctx, st, emb, id, path, cfg.Jobs, *force, *verbose, w)
		if err != nil {
			fmt.Fprintf(w, "❌ %v\n", err)
			stats.failed++
			continue
		}
		stats.files++
		stats.chunks += res.inserted
		stats.embedded += res.embedded
		if res.skipped {
			stats.skipped++
			fmt.Fprintln(w, "⏭️  skip (isi tidak berubah)")
			continue
		}
		extra := ""
		if res.docID != "" {
			extra += " (" + res.docID + ")"
		}
		if res.tenant != "global" {
			extra += " [" + res.tenant + "]"
		}
		fmt.Fprintf(w, "Inserted %d chunks ✅%s\n", res.inserted, extra)
	}

	fmt.Fprintf(w, "⏱  %d file diproses, %d skip, %d gagal — %d chunk tersimpan, %d chunk di-embed dalam %s\n",
		stats.files, stats.skipped, stats.failed, stats.chunks, stats.embedded,
		time.Since(stats.start).Round(time.Millisecond))
	if stats.failed > 0 {
		return 1
	}
	return 0
}

type ingestResult struct {
	inserted int
	embedded int
	skipped  bool
	docID    string
	tenant   string
}

// ingestOne memproses satu file: hash -> skip bila tidak berubah -> parse ->
// embed (batch paralel) -> simpan dalam satu transaksi.
func ingestOne(ctx context.Context, st *store.Store, emb *embedding.Client,
	id, path string, jobs int, force, verbose bool, w io.Writer) (ingestResult, error) {

	var res ingestResult
	src, err := os.ReadFile(path)
	if err != nil {
		return res, fmt.Errorf("baca gagal: %w", err)
	}
	sum := sha256.Sum256(src)
	hash := hex.EncodeToString(sum[:])

	if !force {
		state, err := st.FileState(ctx, id)
		if err != nil {
			return res, err
		}
		if state.Chunks > 0 && state.ContentHash == hash {
			res.skipped = true
			return res, nil
		}
	}

	meta, chunks, err := markdown.ParseDocument(id, string(src))
	if err != nil {
		return res, fmt.Errorf("parse gagal: %w", err)
	}
	if len(chunks) == 0 {
		return res, fmt.Errorf("0 chunk (dilewati)")
	}
	if meta.ParseWarning != "" {
		fmt.Fprintf(w, "\n   ⚠️ frontmatter bukan YAML valid, metadata dibaca longgar: %s\n   ", meta.ParseWarning)
	}
	res.docID = meta.ID
	res.tenant = chunks[0].TenantID

	vecs, err := embedChunks(ctx, emb, chunks, jobs)
	if err != nil {
		return res, fmt.Errorf("embed gagal: %w", err)
	}
	res.embedded = len(chunks)

	if verbose {
		fmt.Fprintln(w)
		for _, ch := range chunks {
			trunc := ""
			if markdown.IsTruncated(ch.EmbeddingText()) {
				trunc = " ⚠️ terpotong"
			}
			fmt.Fprintf(w, "   • [%s] %s — %d char%s\n", ch.Category, ch.SectionTitle, len(ch.Content), trunc)
		}
		fmt.Fprint(w, "   ")
	}

	n, err := st.IngestChunks(ctx, chunks, vecs, hash)
	if err != nil {
		return res, fmt.Errorf("insert gagal: %w", err)
	}
	res.inserted = n
	return res, nil
}

// embedChunks meng-embed semua chunk dengan worker pool (batch per MaxBatch).
func embedChunks(ctx context.Context, emb *embedding.Client, chunks []markdown.Chunk, jobs int) ([][]float32, error) {
	texts := make([]string, len(chunks))
	for i, ch := range chunks {
		texts[i] = markdown.TruncateEmbeddingText(ch.EmbeddingText())
	}

	type span struct{ start, end int }
	var batches []span
	for start := 0; start < len(chunks); start += embedding.MaxBatch {
		end := start + embedding.MaxBatch
		if end > len(chunks) {
			end = len(chunks)
		}
		batches = append(batches, span{start, end})
	}

	workers := jobs
	if workers < 1 {
		workers = 1
	}
	if workers > len(batches) {
		workers = len(batches)
	}

	vecs := make([][]float32, len(chunks))
	work := make(chan span)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for b := range work {
				got, err := emb.EmbedBatch(ctx, texts[b.start:b.end])
				mu.Lock()
				if err != nil {
					if firstErr == nil {
						firstErr = fmt.Errorf("chunk %d-%d: %w", b.start+1, b.end, err)
					}
					mu.Unlock()
					continue
				}
				copy(vecs[b.start:b.end], got)
				mu.Unlock()
			}
		}()
	}

	for _, b := range batches {
		if ctx.Err() != nil {
			break
		}
		work <- b
	}
	close(work)
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return vecs, nil
}

// --- search -----------------------------------------------------------------

func runSearch(ctx context.Context, cfg config.Config, args []string, w io.Writer) int {
	// Flag boleh sebelum maupun setelah query; stdlib flag.Parse berhenti di
	// arg posisi pertama, jadi scan manual semua arg.
	opt := store.SearchOptions{TopK: 3}
	asJSON := false
	var qparts []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		next := func() (string, bool) {
			if i+1 < len(args) {
				i++
				return args[i], true
			}
			return "", false
		}
		switch {
		case a == "-k" || a == "--k":
			if v, ok := next(); ok {
				fmt.Sscanf(v, "%d", &opt.TopK)
			}
		case strings.HasPrefix(a, "-k="):
			fmt.Sscanf(strings.TrimPrefix(a, "-k="), "%d", &opt.TopK)
		case a == "-tenant":
			if v, ok := next(); ok {
				opt.Tenant = v
			}
		case strings.HasPrefix(a, "-tenant="):
			opt.Tenant = strings.TrimPrefix(a, "-tenant=")
		case a == "-category":
			if v, ok := next(); ok {
				opt.Category = v
			}
		case strings.HasPrefix(a, "-category="):
			opt.Category = strings.TrimPrefix(a, "-category=")
		case a == "-json" || a == "--json":
			asJSON = true
		case strings.HasPrefix(a, "-"):
			// flag lain: token dikeluarkan dari query, tidak dipakai
		default:
			qparts = append(qparts, a)
		}
	}
	query := strings.TrimSpace(strings.Join(qparts, " "))
	if query == "" {
		fmt.Fprintln(w, `error: butuh query. Contoh: search "Mas web saya muter-muter terus" -k 3`)
		return 2
	}
	if opt.TopK < 1 || opt.TopK > 50 {
		return fatalf(w, "-k harus antara 1 dan 50 (diberikan %d)", opt.TopK)
	}

	st, err := store.Connect(ctx, cfg.PGDSN, cfg.SQLTable, cfg.EmbeddingDim)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	defer st.Close()

	emb := embedding.NewClient(cfg.OllamaURL, cfg.OllamaModel, cfg.EmbeddingDim)
	qvec, err := emb.Embed(ctx, query)
	if err != nil {
		return fatalf(w, "embed query gagal: %v", err)
	}
	res, err := st.Search(ctx, qvec, opt)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	if asJSON {
		printJSON(w, query, res)
	} else {
		printResults(w, query, res)
	}
	return 0
}

func printResults(w io.Writer, query string, results []store.Result) {
	fmt.Fprintf(w, "\n🔎 %q → %d hasil\n\n", query, len(results))
	medals := []string{"🥇", "🥈", "🥉"}
	for i, r := range results {
		rank := fmt.Sprintf("%d.", i+1)
		if i < len(medals) {
			rank = medals[i]
		}
		doc := r.SourceFile
		if r.DocID != "" {
			doc += " (" + r.DocID + ")"
		}
		fmt.Fprintf(w, "%s [%.4f] %s — §%s [%s]\n    %s\n\n",
			rank, r.Similarity, doc, r.SectionTitle, r.Category, snippet(r.Content))
	}
}

func printJSON(w io.Writer, query string, results []store.Result) {
	type resultRow struct {
		Rank         int     `json:"rank"`
		Similarity   float64 `json:"similarity"`
		SourceFile   string  `json:"source_file"`
		DocID        string  `json:"doc_id,omitempty"`
		DocTitle     string  `json:"doc_title,omitempty"`
		SectionTitle string  `json:"section_title"`
		Category     string  `json:"category,omitempty"`
		Tenant       string  `json:"tenant_id"`
		Snippet      string  `json:"snippet"`
		UpdatedAt    string  `json:"updated_at"`
	}
	rows := make([]resultRow, 0, len(results))
	for i, r := range results {
		rows = append(rows, resultRow{
			Rank:         i + 1,
			Similarity:   float64(r.Similarity),
			SourceFile:   r.SourceFile,
			DocID:        r.DocID,
			DocTitle:     r.DocTitle,
			SectionTitle: r.SectionTitle,
			Category:     r.Category,
			Tenant:       r.TenantID,
			Snippet:      snippet(r.Content),
			UpdatedAt:    r.UpdatedAt.Format(time.RFC3339),
		})
	}
	buf, err := json.MarshalIndent(struct {
		Query   string      `json:"query"`
		Results []resultRow `json:"results"`
	}{query, rows}, "", "  ")
	if err != nil {
		fmt.Fprintf(w, "error: marshal json: %v\n", err)
		return
	}
	fmt.Fprintln(w, string(buf))
}

func snippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 220 {
		s = s[:220] + "…"
	}
	return s
}
