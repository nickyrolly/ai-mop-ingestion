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
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"ai-mop-ingestion/internal/config"
	"ai-mop-ingestion/internal/embedding"
	"ai-mop-ingestion/internal/markdown"
	"ai-mop-ingestion/internal/store"
)

const appVersion = "v0.4.0"

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
  ai-mop-ingestion search "query" [-k 3] [-json]       Pencarian hybrid (vektor + kata kunci)
                     [-tenant ID] [-category NAMA]
                     [-status NAMA] [-confidentiality NAMA]
                     [-min-score 0.50] [-per-doc 2]
  ai-mop-ingestion version                             Tampilkan versi

Opsi ingest:
  -dir PATH     folder atau satu file .md (default: knowledge-base terdekat)
  -v            log per chunk (judul, kategori, panjang teks)
  -force        re-embed walau isi file tidak berubah

Opsi search:
  -k N          jumlah hasil akhir (default 3, maks 50)
  -min-score X  ambang cosine 0..1 untuk hit vektor; 0 = nonaktif (default 0.50)
  -per-doc N    maksimum chunk per dokumen di hasil akhir; 0 = tanpa batas (default 2)
  -status NAMA          filter status frontmatter (kosong = semua)
  -confidentiality N  filter confidentiality frontmatter (kosong = semua)
  -json               keluaran JSON

Kode keluar search: 0 ada hasil, 3 tidak ada hasil yang lolos gerbang skor.

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
	// Path yang diberikan eksplisit (flag atau env) tidak boleh diam-diam
	// dialihkan ke kandidat default — itu bisa meng-ingest folder yang salah.
	if flagDir != "" {
		if _, err := os.Stat(flagDir); err != nil {
			return "", fmt.Errorf("path -dir tidak ditemukan: %s", flagDir)
		}
		abs, _ := filepath.Abs(flagDir)
		return abs, nil
	}
	if envDir != "" {
		if _, err := os.Stat(envDir); err != nil {
			return "", fmt.Errorf("AI_MOP_KB_DIR tidak ditemukan: %s", envDir)
		}
		abs, _ := filepath.Abs(envDir)
		return abs, nil
	}
	for _, c := range kbDirCandidates {
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
	if code := ensureModel(ctx, st, cfg, w); code != 0 {
		return code
	}
	fmt.Fprintf(w, "✅ Skema siap: table %q (embedding vector(%d) + index hnsw)\n", cfg.SQLTable, cfg.EmbeddingDim)
	return 0
}

func ensureModel(ctx context.Context, st *store.Store, cfg config.Config, w io.Writer) int {
	adopted, err := st.EnsureEmbeddingModel(ctx, cfg.OllamaModel)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	if adopted {
		fmt.Fprintf(w, "ℹ️  Index yang sudah ada dikunci ke model %s (dim=%d). Ganti model butuh tabel lain, atau kosongkan tabel dan hapus %s_meta.\n",
			cfg.OllamaModel, cfg.EmbeddingDim, cfg.SQLTable)
	}
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
	if code := ensureModel(ctx, st, cfg, w); code != 0 {
		return code
	}

	emb := embedding.NewClient(cfg.OllamaURL, cfg.OllamaModel, cfg.EmbeddingDim)

	fmt.Fprintf(w, "🗂  Basis: %s — %d file .md ditemukan (rekursif)\n", kbDir, len(files))
	fmt.Fprintf(w, "🤖 Embedding: %s via %s (dim=%d, paralel=%d)\n",
		emb.Endpoint, cfg.OllamaModel, cfg.EmbeddingDim, cfg.Jobs)

	stats := ingestStats{start: time.Now()}
	canceled := false
	for _, path := range files {
		if ctx.Err() != nil {
			fmt.Fprintln(w, "⚠️  dibatalkan oleh sinyal/timeout")
			canceled = true
			break
		}
		id := sourceID(root, path)
		fmt.Fprintf(w, "⏳ Processing %s... ", id)

		res, err := ingestOne(ctx, st, emb, id, path, cfg.OllamaModel, cfg.Jobs, *force, *verbose, w)
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
	if !canceled {
		pruned, err := maybePrune(ctx, st, kbDir, root, files)
		if err != nil {
			fmt.Fprintf(w, "⚠️  prune gagal: %v\n", err)
			stats.failed++
		} else if pruned.Chunks > 0 {
			names := pruned.Names
			extra := ""
			if len(names) > 8 {
				extra = fmt.Sprintf(" … +%d", len(names)-8)
				names = names[:8]
			}
			fmt.Fprintf(w, "🧹 %d dokumen yang sudah tidak ada di disk dihapus (%d chunk): %s%s\n",
				pruned.Files, pruned.Chunks, strings.Join(names, ", "), extra)
		}
	}

	if canceled {
		return 130 // konvensi SIGINT; kegagalan file akibat cancel bukan error data
	}
	if stats.failed > 0 {
		return 1
	}
	return 0
}

// maybePrune menghapus chunk milik file yang sudah tidak ada, hanya saat
// target ingest adalah folder knowledge-base (bukan satu file atau subfolder).
// Subfolder memakai kunci path yang berbeda, jadi prune di sana bisa menghapus
// dokumen yang masih ada.
func maybePrune(ctx context.Context, st *store.Store, kbDir, root string, files []string) (store.PruneStats, error) {
	var zero store.PruneStats
	info, err := os.Stat(kbDir)
	if err != nil || !info.IsDir() || !pruneEligible(kbDir) || len(files) == 0 {
		return zero, nil
	}
	keep := make([]string, 0, len(files))
	for _, path := range files {
		keep = append(keep, sourceID(root, path))
	}
	return st.PruneSources(ctx, keep)
}

// pruneEligible benar bila folder ini adalah root knowledge-base, yaitu tempat
// kunci source_file dibuat relatif terhadap seluruh korpus.
func pruneEligible(root string) bool {
	if strings.EqualFold(filepath.Base(root), "knowledge-base") {
		return true
	}
	n := 0
	for _, sub := range []string{"runbooks", "learnings", "vendors", "patterns", "clients"} {
		st, err := os.Stat(filepath.Join(root, sub))
		if err == nil && st.IsDir() {
			n++
		}
	}
	return n >= 2
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
	id, path, model string, jobs int, force, verbose bool, w io.Writer) (ingestResult, error) {

	var res ingestResult
	src, err := os.ReadFile(path)
	if err != nil {
		return res, fmt.Errorf("baca gagal: %w", err)
	}
	sum := sha256.Sum256(src)
	hash := hex.EncodeToString(sum[:])

	meta, chunks, err := markdown.ParseDocument(id, string(src))
	if err != nil {
		return res, fmt.Errorf("parse gagal: %w", err)
	}
	if len(chunks) == 0 {
		return res, fmt.Errorf("0 chunk (dilewati)")
	}

	if !force {
		state, err := st.FileState(ctx, id)
		if err != nil {
			return res, err
		}
		// Jumlah chunk ikut dibandingkan supaya section yang kini dipecah
		// tidak tertinggal sebagai satu vektor terpotong.
		if state.Chunks > 0 && state.Chunks == len(chunks) && state.ContentHash == hash {
			res.skipped = true
			return res, nil
		}
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

	n, err := st.IngestChunks(ctx, chunks, vecs, hash, model)
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
	opt := store.SearchOptions{TopK: 3, MinScore: 0.50, MaxPerDoc: 2}
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
		case a == "-min-score" || a == "--min-score":
			if v, ok := next(); ok {
				fmt.Sscanf(v, "%f", &opt.MinScore)
			}
		case strings.HasPrefix(a, "-min-score="):
			fmt.Sscanf(strings.TrimPrefix(a, "-min-score="), "%f", &opt.MinScore)
		case a == "-per-doc" || a == "--per-doc":
			if v, ok := next(); ok {
				fmt.Sscanf(v, "%d", &opt.MaxPerDoc)
			}
		case strings.HasPrefix(a, "-per-doc="):
			fmt.Sscanf(strings.TrimPrefix(a, "-per-doc="), "%d", &opt.MaxPerDoc)
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
		case a == "-status":
			if v, ok := next(); ok {
				opt.Status = v
			}
		case strings.HasPrefix(a, "-status="):
			opt.Status = strings.TrimPrefix(a, "-status=")
		case a == "-confidentiality":
			if v, ok := next(); ok {
				opt.Confidentiality = v
			}
		case strings.HasPrefix(a, "-confidentiality="):
			opt.Confidentiality = strings.TrimPrefix(a, "-confidentiality=")
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
	if opt.MinScore < 0 || opt.MinScore > 1 {
		return fatalf(w, "-min-score harus antara 0 dan 1 (diberikan %g)", opt.MinScore)
	}
	if opt.MaxPerDoc < 0 {
		return fatalf(w, "-per-doc tidak boleh negatif")
	}
	// Teks query dipakai jalur kata kunci (hybrid); tanpa ini, pencarian
	// kembali ke cosine saja.
	opt.Query = query

	st, err := store.Connect(ctx, cfg.PGDSN, cfg.SQLTable, cfg.EmbeddingDim)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	defer st.Close()
	if err := st.InitSchema(ctx); err != nil {
		return fatalf(w, "%v", err)
	}
	if code := ensureModel(ctx, st, cfg, w); code != 0 {
		return code
	}

	emb := embedding.NewClient(cfg.OllamaURL, cfg.OllamaModel, cfg.EmbeddingDim)
	qvec, err := emb.Embed(ctx, query)
	if err != nil {
		return fatalf(w, "embed query gagal: %v", err)
	}
	res, err := st.Search(ctx, qvec, opt)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	if len(res) == 0 {
		// Bedakan "tidak ada yang relevan" dari kesalahan: agent pemanggil
		// harus tahu kapan berhenti dan bertanya, bukan mengarang jawaban.
		if asJSON {
			printJSON(w, query, res, opt)
		} else {
			fmt.Fprintf(w, "\n🔎 %q → tidak ada dokumen relevan (ambang cosine %.2f)\n", query, opt.MinScore)
		}
		return 3
	}
	if asJSON {
		printJSON(w, query, res, opt)
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
		kw := ""
		if r.TextMatched() {
			kw = fmt.Sprintf(" +kata kunci %.3f", r.TextRank)
		}
		fmt.Fprintf(w, "%s [rrf %.4f | cosine %.4f%s] %s — §%s [%s]\n    %s\n\n",
			rank, r.Score, r.Similarity, kw, doc, r.SectionTitle, r.Category, snippet(r.Content))
	}
}

func printJSON(w io.Writer, query string, results []store.Result, opt store.SearchOptions) {
	type resultRow struct {
		Rank         int     `json:"rank"`
		Score        float64 `json:"score"`
		Similarity   float64 `json:"similarity"`
		TextMatch    bool    `json:"text_match"`
		TextRank     float64 `json:"text_rank,omitempty"`
		SourceFile   string  `json:"source_file"`
		DocID        string  `json:"doc_id,omitempty"`
		DocTitle     string  `json:"doc_title,omitempty"`
		SectionTitle string  `json:"section_title"`
		Category        string `json:"category,omitempty"`
		Tenant          string `json:"tenant_id"`
		Status          string `json:"status,omitempty"`
		Confidentiality string `json:"confidentiality,omitempty"`
		Snippet      string  `json:"snippet"`
		UpdatedAt    string  `json:"updated_at"`
	}
	rows := make([]resultRow, 0, len(results))
	for i, r := range results {
		rows = append(rows, resultRow{
			Rank:         i + 1,
			Score:        round4(float64(r.Score)),
			Similarity:   round4(float64(r.Similarity)),
			TextMatch:    r.TextMatched(),
			TextRank:     round4(float64(r.TextRank)),
			SourceFile:   r.SourceFile,
			DocID:        r.DocID,
			DocTitle:     r.DocTitle,
			SectionTitle: r.SectionTitle,
			Category:        r.Category,
			Tenant:          r.TenantID,
			Status:          r.Status,
			Confidentiality: r.Confidentiality,
			Snippet:      snippet(r.Content),
			UpdatedAt:    r.UpdatedAt.Format(time.RFC3339),
		})
	}
	buf, err := json.MarshalIndent(struct {
		Query    string      `json:"query"`
		Total    int         `json:"total"`
		MinScore float64     `json:"min_score"`
		PerDoc   int         `json:"per_doc"`
		Results  []resultRow `json:"results"`
	}{query, len(rows), round4(float64(opt.MinScore)), opt.MaxPerDoc, rows}, "", "  ")
	if err != nil {
		fmt.Fprintf(w, "error: marshal json: %v\n", err)
		return
	}
	fmt.Fprintln(w, string(buf))
}

// round4 membulatkan skor agar JSON mudah dibaca dan stabil dibandingkan.
func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

func snippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if utf8.RuneCountInString(s) <= 220 {
		return s
	}
	n := 0
	for i := range s {
		if n == 220 {
			return s[:i] + "…"
		}
		n++
	}
	return s
}
