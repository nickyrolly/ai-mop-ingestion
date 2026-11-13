// Package cli menerjemahkan argumen program menjadi sub-perintah
// init / ingest / search.
package cli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"ai-mop-ingestion/internal/config"
	"ai-mop-ingestion/internal/embedding"
	"ai-mop-ingestion/internal/markdown"
	"ai-mop-ingestion/internal/store"
)

const appVersion = "v0.1.0"

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
  ai-mop-ingestion init [-v]                      Inisialisasi skema (tabel + index HNSW)
  ai-mop-ingestion ingest [-dir PATH] [-v]        Ingest semua *.md di folder knowledge-base
  ai-mop-ingestion search "query anda" [-k 3] [-json]  Verifikasi hasil ingestion
  ai-mop-ingestion version                        Tampilkan versi

Konfigurasi environment variable:
  OLLAMA_URL      (default: http://localhost:11434)
  OLLAMA_MODEL    (default: bge-m3)
  EMBEDDING_DIM   (default: 1024)
  PG_DSN          (default: postgres://postgres:postgres@localhost:5433/aimop_rag?sslmode=disable)
  AI_MOP_KB_DIR   (default: folder knowledge-base terdekat)
`)
}

func fatalf(w io.Writer, format string, args ...any) int {
	fmt.Fprintf(w, "error: "+format+"\n", args...)
	return 1
}

// kbDirCandidates diprobe satu per satu bila tidak ada -dir / AI_MOP_KB_DIR.
var kbDirCandidates = []string{
	"nickyrolly/ai-mop/knowledge-base/runbooks", // layout sesi (repo di github.com/nickyrolly/ai-mop)
	"../ai-mop/knowledge-base/runbooks",         // path dari spesifikasi
	"knowledge-base/runbooks",                   // bila dijalankan dari root repo
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
		st, err := os.Stat(c)
		if err == nil && st.IsDir() {
			abs, _ := filepath.Abs(c)
			return abs, nil
		}
	}
	return "", fmt.Errorf("folder knowledge-base tidak ditemukan; gunakan -dir PATH atau set AI_MOP_KB_DIR")
}

func collectMarkdownFiles(root string) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("baca folder %s: %w", root, err)
	}
	var files []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".md") {
			continue
		}
		files = append(files, filepath.Join(root, e.Name()))
	}
	sort.Strings(files)
	return files, nil
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

func runIngest(ctx context.Context, cfg config.Config, args []string, w io.Writer) int {
	fs := flag.NewFlagSet("ingest", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dir := fs.String("dir", "", "folder berisi file *.md")
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

	st, err := store.Connect(ctx, cfg.PGDSN, cfg.SQLTable, cfg.EmbeddingDim)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	defer st.Close()

	emb := embedding.NewClient(cfg.OllamaEmbeddingsURL(), cfg.OllamaModel, cfg.EmbeddingDim)

	// Skema diinisialisasi otomatis saat aplikasi dijalankan pertama kali.
	if err := st.InitSchema(ctx); err != nil {
		return fatalf(w, "%v", err)
	}

	fmt.Fprintf(w, "🗂  %d file Markdown ditemukan di %s\n", len(files), kbDir)
	var (
		done   int
		total  int
		start  time.Time
		failed bool
	)
	if len(files) > 0 {
		start = time.Now()
	}

	for _, path := range files {
		base := filepath.Base(path)
		fmt.Fprintf(w, "⏳ Processing %s... ", base)

		src, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(w, "❌ baca gagal: %v\n", err)
			failed = true
			continue
		}
		meta, chunks, err := markdown.ParseDocument(base, string(src))
		if err != nil {
			fmt.Fprintf(w, "❌ parse gagal: %v\n", err)
			failed = true
			continue
		}
		if len(chunks) == 0 {
			fmt.Fprintf(w, "⚠️ 0 chunk (lewat)\n")
			continue
		}

		vecs := make([][]float32, len(chunks))
		ok := true
		for i, ch := range chunks {
			e, err := emb.Embed(ctx, markdown.TruncateEmbeddingText(ch.EmbeddingText()))
			if err != nil {
				fmt.Fprintf(w, "❌ embed gagal (%q): %v\n", ch.SectionTitle, err)
				ok = false
				break
			}
			vecs[i] = e
		}
		if !ok {
			failed = true
			continue
		}

		n, err := st.IngestChunks(ctx, chunks, vecs)
		if err != nil {
			fmt.Fprintf(w, "❌ insert gagal: %v\n", err)
			failed = true
			continue
		}
		done++
		total += n
		re := ""
		if meta.ID != "" {
			re = fmt.Sprintf(" (%s)", meta.ID)
		}
		fmt.Fprintf(w, "Inserted %d chunks ✅%s\n", n, re)
	}

	if len(files) > 0 {
		fmt.Fprintf(w, "⏱  %d/%d file ingested (%d chunks) dalam %s\n",
			done, len(files), total, time.Since(start).Round(time.Millisecond))
	}
	if failed {
		return 1
	}
	return 0
}

// --- search -----------------------------------------------------------------

func runSearch(ctx context.Context, cfg config.Config, args []string, w io.Writer) int {
	// Flag boleh sebelum/maupun setelah query; stdlib flag.Parse berhenti di
	// arg posisi pertama, jadi scan manual semua arg (gabung: query + flag).
	topK := 3
	asJSON := false
	var qparts []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-k" && i+1 < len(args):
			if _, err := fmt.Sscanf(args[i+1], "%d", &topK); err != nil {
				topK = 3
			}
			i++ // lewati nilai flag
		case strings.HasPrefix(a, "-k="):
			fmt.Sscanf(strings.TrimPrefix(a, "-k="), "%d", &topK)
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

	st, err := store.Connect(ctx, cfg.PGDSN, cfg.SQLTable, cfg.EmbeddingDim)
	if err != nil {
		return fatalf(w, "%v", err)
	}
	defer st.Close()

	emb := embedding.NewClient(cfg.OllamaEmbeddingsURL(), cfg.OllamaModel, cfg.EmbeddingDim)
	qvec, err := emb.Embed(ctx, query)
	if err != nil {
		return fatalf(w, "embed query gagal: %v", err)
	}
	res, err := st.Search(ctx, qvec, topK)
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
		fmt.Fprintf(w, "%s [%.4f] %s — §%s (saved %s)\n    %s\n\n",
			rank, r.Similarity, r.SourceFile, r.SectionTitle, r.CreatedAt.Format("2006-01-02 15:04"), snippet(r.Content))
	}
}

func printJSON(w io.Writer, query string, results []store.Result) {
	type resultRow struct {
		Rank         int     `json:"rank"`
		Similarity   float64 `json:"similarity"`
		SourceFile   string  `json:"source_file"`
		SectionTitle string  `json:"section_title"`
		Snippet      string  `json:"snippet"`
		StoredAt     string  `json:"stored_at"`
	}
	rows := make([]resultRow, 0, len(results))
	for i, r := range results {
		rows = append(rows, resultRow{i + 1, float64(r.Similarity), r.SourceFile, r.SectionTitle, snippet(r.Content), r.CreatedAt.Format(time.RFC3339)})
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
