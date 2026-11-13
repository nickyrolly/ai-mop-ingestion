// Package markdown mem-parsing dokumen Markdown (frontmatter YAML + chunking
// per heading H2/H3) menjadi unit teks yang siap di-embedding.
package markdown

import (
	"fmt"
	"path"
	"strings"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// MaxChunkChars batas panjang teks yang dikirim ke model embedding
// (bge-m3 punya konteks 8192 token). Section yang lebih panjang dipecah,
// bukan dipotong diam-diam.
const MaxChunkChars = 7000

// ChunkOverlapChars jumlah byte yang diulang di batas potongan supaya konsep
// yang jatuh di tengah section tetap ada di salah satu vektor.
const ChunkOverlapChars = 500

// sectionPartReserve ruang judul untuk sufiks " (bagian i/n)" saat menghitung
// anggaran isi, supaya potongan tidak melebihi MaxChunkChars setelah diberi judul.
const sectionPartReserve = 32

// bk3 adalah tiga backtick ("```") sebagai konstanta, agar tidak ada literal
// backtick di dalam source Go.
const bk3 = "\x60\x60\x60"

// headingLevel mereturn level heading ATX (2 atau 3) atau 0 bila baris bukan
// heading yang kita pakai sebagai batas chunk.
func headingLevel(line string) int {
	t := strings.TrimRight(line, "\r")
	n := 0
	for n < len(t) && t[n] == '#' {
		n++
	}
	if n != 2 && n != 3 {
		return 0
	}
	if n >= len(t) {
		return 0
	}
	if t[n] != ' ' && t[n] != '\t' {
		return 0
	}
	return n
}

// isHeadingLine melaporkan apakah baris heading ATX level 2 atau 3.
func isHeadingLine(line string) bool { return headingLevel(line) != 0 }

// headingTitle mereturn teks heading tanpa '#' dan whitespace.
func headingTitle(line string) string {
	t := strings.TrimRight(line, "\r")
	i := 0
	for i < len(t) && t[i] == '#' {
		i++
	}
	t = strings.TrimSpace(t[i:])
	// buang closed-ATX '#' di akhir ("## Judul ##").
	for strings.HasSuffix(t, "#") {
		t = strings.TrimSpace(t[:len(t)-1])
	}
	return t
}

// isFenceLine melaporkan apakah baris adalah fence code block (```/~~~).
func isFenceLine(line string) bool {
	t := strings.TrimLeft(line, " \t")
	return strings.HasPrefix(t, bk3) || strings.HasPrefix(t, "~~~")
}

// section satu potongan isi dokumen: heading + body.
type section struct {
	level   int
	title   string
	content strings.Builder
}

// Meta metadata YAML frontmatter dokumen.
type Meta struct {
	Title           string   `yaml:"title"`
	ID              string   `yaml:"id"`
	Status          string   `yaml:"status"`
	Owner           string   `yaml:"owner"`
	Tenant          string   `yaml:"tenant"`
	Confidentiality string   `yaml:"confidentiality"`
	Tags            []string `yaml:"tags"`

	// ParseWarning berisi error YAML bila frontmatter tidak valid dan metadata
	// diambil dengan parsing longgar (dokumen tetap di-ingest).
	ParseWarning string `yaml:"-"`
}

// Chunk satu unit teks untuk embedding: satu topik/SOP utuh.
type Chunk struct {
	SourceFile   string // path relatif terhadap root knowledge-base (kunci dedup)
	DocID        string // frontmatter id (mis. RB-WP-001)
	DocTitle     string // judul dokumen
	SectionTitle string // heading chunk, sudah berbentuk breadcrumb "H2 > H3"
	Content      string // isi section (tanpa baris heading)
	Category        string // inferensi dari path file
	TenantID        string // dari frontmatter, default "global"
	Status          string // frontmatter status; kosong bila tidak ada
	Confidentiality string // frontmatter confidentiality; kosong bila tidak ada
}

// EmbeddingText mereturn representasi teks chunk untuk model embedding:
//
//	[File: <path> | Doc: <judul> (<id>) | Section: <heading>]
//	<isi_section>
//
// Judul dokumen dan breadcrumb heading ikut masuk supaya chunk H3 seperti
// "1. DNS" tidak kehilangan konteks runbook induknya.
func (c Chunk) EmbeddingText() string {
	head := fmt.Sprintf("[File: %s", c.SourceFile)
	if c.DocTitle != "" {
		head += " | Doc: " + c.DocTitle
	}
	if c.DocID != "" {
		head += " (" + c.DocID + ")"
	}
	head += " | Section: " + c.SectionTitle + "]"
	if c.Category != "" {
		head += " [" + c.Category + "]"
	}
	return head + "\n" + strings.TrimSpace(c.Content)
}

// ParseDocument mem-parsing isi Markdown mentah menjadi (metadata, chunk).
// relPath adalah path file relatif terhadap root knowledge-base (dipakai
// sebagai identitas dokumen sekaligus sumber kategori).
func ParseDocument(relPath, src string) (Meta, []Chunk, error) {
	var meta Meta
	body := strings.TrimPrefix(src, "\ufeff")
	relPath = normalizePath(relPath)

	if hasFrontmatter(body) {
		rest, fm, err := splitFrontmatter(body)
		if err != nil {
			return meta, nil, fmt.Errorf("parsing frontmatter %s: %w", relPath, err)
		}
		meta = fm
		body = rest
	}

	tenantID := strings.TrimSpace(meta.Tenant)
	if tenantID == "" {
		tenantID = "global"
	}
	category := InferCategory(relPath)

	sections, preamble, err := splitSections(body)
	if err != nil {
		return meta, nil, fmt.Errorf("chunking %s: %w", relPath, err)
	}

	// H1 dipakai sebagai judul fallback, lalu dihilangkan dari preamble.
	docTitle := strings.TrimSpace(meta.Title)
	if docTitle == "" {
		docTitle = firstH1(preamble)
	}
	preamble = removeH1Line(preamble)

	var chunks []Chunk
	appendChunk := func(title, content string) {
		content = strings.TrimSpace(content)
		if content == "" {
			return
		}
		title = strings.TrimSpace(title)
		if title == "" {
			title = docTitle
		}
		if title == "" {
			title = "Overview"
		}
		chunks = append(chunks, Chunk{
			SourceFile:      relPath,
			DocID:           strings.TrimSpace(meta.ID),
			DocTitle:        docTitle,
			SectionTitle:    title,
			Content:         content,
			Category:        category,
			TenantID:        tenantID,
			Status:          strings.TrimSpace(meta.Status),
			Confidentiality: strings.TrimSpace(meta.Confidentiality),
		})
	}

	if len(sections) == 0 {
		// Tanpa heading H2/H3: satu chunk utuh.
		appendChunk(docTitle, preamble)
		return meta, splitOversizedChunks(chunks), nil
	}

	parentH2 := ""
	for i, s := range sections {
		title := s.title
		switch s.level {
		case 2:
			parentH2 = s.title
		case 3:
			// Breadcrumb: H3 tetap membawa konteks H2 induknya.
			if parentH2 != "" {
				title = parentH2 + " > " + s.title
			}
		}

		content := s.content.String()
		// Preamble (paragraf pembuka sebelum heading pertama) digabung ke
		// chunk pertama, supaya konteks tidak hilang.
		if i == 0 && strings.TrimSpace(preamble) != "" {
			content = preamble + "\n" + content
		}
		appendChunk(title, content)
	}
	return meta, splitOversizedChunks(chunks), nil
}

// TruncateEmbeddingText memangkas teks ke MaxChunkChars dengan penanda,
// tanpa membelah rune UTF-8.
func TruncateEmbeddingText(t string) string {
	if len(t) <= MaxChunkChars {
		return t
	}
	cut := MaxChunkChars
	for cut > 0 && !utf8.RuneStart(t[cut]) {
		cut--
	}
	return t[:cut] + "\n...[truncated]"
}

// IsTruncated melaporkan apakah teks chunk akan terpotong sebelum embedding.
func IsTruncated(t string) bool { return len(t) > MaxChunkChars }

// splitOversizedChunks memecah chunk yang teks embedding-nya melewati
// MaxChunkChars. Isi yang disimpan adalah potongan itu sendiri, jadi vektor
// dan cuplikan pencarian sama-sama melihat bagian yang relevan.
func splitOversizedChunks(chunks []Chunk) []Chunk {
	out := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, splitOne(c, 0)...)
	}
	return out
}

func splitOne(c Chunk, depth int) []Chunk {
	c.Content = strings.TrimSpace(c.Content)
	if c.Content == "" {
		return nil
	}
	if !IsTruncated(c.EmbeddingText()) || depth > 8 {
		return []Chunk{c}
	}
	budget := embeddingBudget(c.SectionTitle+strings.Repeat("x", sectionPartReserve), c)
	parts := splitText(c.Content, budget, ChunkOverlapChars)
	if len(parts) <= 1 {
		parts = splitText(c.Content, budget/2, budget/10)
	}
	if len(parts) <= 1 {
		return []Chunk{c}
	}
	out := make([]Chunk, 0, len(parts))
	for i, p := range parts {
		ch := c
		ch.Content = strings.TrimSpace(p)
		ch.SectionTitle = fitSectionTitle(c.SectionTitle, i+1, len(parts))
		if ch.Content == "" {
			continue
		}
		if IsTruncated(ch.EmbeddingText()) {
			out = append(out, splitOne(ch, depth+1)...)
			continue
		}
		out = append(out, ch)
	}
	return out
}

// embeddingBudget menghitung panjang isi yang masih muat di MaxChunkChars
// setelah header embedding (path, judul, section) ditambahkan.
func embeddingBudget(sectionTitle string, sample Chunk) int {
	probe := sample
	probe.SectionTitle = sectionTitle
	probe.Content = "x"
	headerLen := len(probe.EmbeddingText()) - 1
	budget := MaxChunkChars - headerLen
	if budget < 800 {
		budget = 800
	}
	return budget
}

// fitSectionTitle menambahkan sufiks bagian tanpa melewati VARCHAR(255).
func fitSectionTitle(base string, i, n int) string {
	suffix := fmt.Sprintf(" (bagian %d/%d)", i, n)
	maxBase := 255 - len(suffix)
	if maxBase < 1 {
		if len(suffix) <= 255 {
			return suffix
		}
		return suffix[:255]
	}
	if len(base) <= maxBase {
		return base + suffix
	}
	cut := maxBase
	for cut > 0 && !utf8.RuneStart(base[cut]) {
		cut--
	}
	return strings.TrimSpace(base[:cut]) + suffix
}

// splitText memotong s per anggaran byte, dengan overlap, di batas paragraf,
// baris, atau spasi bila ada. Indeks selalu jatuh di awal rune.
func splitText(s string, budget, overlap int) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	if budget < 200 {
		budget = 200
	}
	if len(s) <= budget {
		return []string{s}
	}
	if overlap < 0 || overlap >= budget/2 {
		overlap = budget / 10
	}
	var parts []string
	start := 0
	for start < len(s) {
		if len(s)-start <= budget {
			rest := strings.TrimSpace(s[start:])
			if rest != "" {
				parts = append(parts, rest)
			}
			break
		}
		end := snapEnd(s, start, start+budget)
		if end <= start {
			end = runeBoundary(s, start+budget)
		}
		part := strings.TrimSpace(s[start:end])
		if part != "" {
			parts = append(parts, part)
		}
		next := end - overlap
		if next <= start {
			next = end
		}
		next = runeBoundaryForward(s, next)
		if next <= start {
			next = end
		}
		if next <= start {
			next = start + 1
			next = runeBoundaryForward(s, next)
		}
		start = next
	}
	return parts
}

func snapEnd(s string, start, end int) int {
	if end > len(s) {
		end = len(s)
	}
	end = runeBoundary(s, end)
	floor := start + (end-start)/2
	if floor < start {
		floor = start
	}
	window := s[floor:end]
	if i := strings.LastIndex(window, "\n\n"); i >= 0 {
		return floor + i
	}
	if i := strings.LastIndex(window, "\n"); i >= 0 {
		return floor + i
	}
	if i := strings.LastIndex(window, " "); i >= 0 {
		return floor + i
	}
	return end
}

func runeBoundary(s string, i int) int {
	if i > len(s) {
		i = len(s)
	}
	if i < 0 {
		return 0
	}
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	return i
}

func runeBoundaryForward(s string, i int) int {
	if i < 0 {
		return 0
	}
	if i >= len(s) {
		return len(s)
	}
	for i < len(s) && !utf8.RuneStart(s[i]) {
		i++
	}
	return i
}

// hasFrontmatter memastikan dokumen benar-benar dibuka dengan blok '---' pada
// baris pertama, bukan horizontal rule biasa.
func hasFrontmatter(src string) bool {
	if !strings.HasPrefix(src, "---") {
		return false
	}
	rest := src[3:]
	return strings.HasPrefix(rest, "\n") || strings.HasPrefix(rest, "\r\n")
}

// splitFrontmatter memisah blok YAML di awal (---...---) dari sisa body.
func splitFrontmatter(src string) (string, Meta, error) {
	var meta Meta
	if !hasFrontmatter(src) {
		return src, meta, nil
	}
	body0 := src[3:]
	// Hilangkan newline(s) setelah '---' pembuka.
	body0 = strings.TrimLeft(body0, "\r\n")
	// cari '---' penutup pada barisnya sendiri.
	lines := strings.Split(body0, "\n")
	end := -1
	for i, l := range lines {
		if strings.TrimRight(l, "\r") == "---" {
			end = i
			break
		}
	}
	if end < 0 {
		return "", meta, fmt.Errorf("frontmatter tidak ditutup (tanpa ---)")
	}
	raw := strings.Join(lines[:end], "\n")
	body := strings.Join(lines[end+1:], "\n")
	if err := yaml.Unmarshal([]byte(raw), &meta); err != nil {
		// Frontmatter bisa saja tidak valid YAML (mis. nilai mengandung ": ").
		// Isi dokumen tetap berharga, jadi metadata diambil secara longgar dan
		// peringatannya diteruskan ke pemanggil.
		meta = parseFrontmatterLenient(raw)
		meta.ParseWarning = err.Error()
	}
	return body, meta, nil
}

// parseFrontmatterLenient mengambil field metadata penting dari frontmatter
// yang bukan YAML valid: hanya baris "key: value" level atas dan list inline.
func parseFrontmatterLenient(raw string) Meta {
	var m Meta
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimRight(line, "\r")
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") ||
			strings.HasPrefix(trimmed, "- ") || line != trimmed {
			continue // kosong, komentar, item list, atau lanjutan nested
		}
		key, val, ok := strings.Cut(trimmed, ":")
		if !ok {
			continue
		}
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "title":
			m.Title = val
		case "id":
			m.ID = val
		case "status":
			m.Status = val
		case "owner":
			m.Owner = val
		case "tenant":
			m.Tenant = val
		case "confidentiality":
			m.Confidentiality = val
		case "tags":
			m.Tags = parseInlineList(val)
		}
	}
	return m
}

// parseInlineList mem-parsing list YAML inline sederhana: [a, b, c].
func parseInlineList(s string) []string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
		if s == "" {
			return nil
		}
		return []string{s}
	}
	s = strings.TrimSuffix(strings.TrimPrefix(s, "["), "]")
	var out []string
	for _, part := range strings.Split(s, ",") {
		part = strings.Trim(strings.TrimSpace(part), `"'`)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

// splitSections memecah body per heading H2/H3; fence-aware.
func splitSections(body string) ([]section, string, error) {
	var sections []section
	var pre strings.Builder
	cur := -1
	inFence := false

	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimRight(line, "\r")
		if isFenceLine(trimmed) {
			inFence = !inFence
			appendLine(&sections, cur, &pre, trimmed)
			continue
		}
		if !inFence {
			if level := headingLevel(trimmed); level != 0 {
				sections = append(sections, section{level: level, title: headingTitle(trimmed)})
				cur = len(sections) - 1
				continue
			}
		}
		appendLine(&sections, cur, &pre, trimmed)
	}
	if inFence {
		return nil, "", fmt.Errorf("fenced code block tidak ditutup")
	}
	return sections, pre.String(), nil
}

// appendLine menulis baris ke section terakhir (atau preamble bila belum ada
// section).
func appendLine(sections *[]section, cur int, preamble *strings.Builder, line string) {
	if cur >= 0 {
		(*sections)[cur].content.WriteString(line + "\n")
	} else {
		preamble.WriteString(line + "\n")
	}
}

// firstH1 mereturn teks heading H1 pertama di teks ("# title").
func firstH1(s string) string {
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimRight(line, "\r")
		if len(t) >= 2 && t[0] == '#' && (t[1] == ' ' || t[1] == '\t') {
			return stripATXClosing(t[1:])
		}
	}
	return ""
}

// stripATXClosing menghapus trailing '##' (closed ATX).
func stripATXClosing(s string) string {
	s = strings.TrimSpace(s)
	for strings.HasSuffix(s, "#") {
		s = strings.TrimSpace(s[:len(s)-1])
	}
	return s
}

// removeH1Line menghapus baris H1 pertama dari teks (jika ada).
func removeH1Line(s string) string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	removed := false
	for _, line := range lines {
		if !removed {
			t := strings.TrimRight(line, "\r")
			if len(t) >= 2 && t[0] == '#' && (t[1] == ' ' || t[1] == '\t') {
				removed = true
				continue
			}
		}
		out = append(out, line)
	}
	if !removed {
		return s
	}
	return strings.Join(out, "\n")
}

// normalizePath menyamakan separator path jadi '/' agar kunci dedup konsisten
// di macOS/Linux/Windows.
func normalizePath(p string) string {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	return path.Clean(p)
}

// InferCategory menebak kategori chunk dari path relatif file.
func InferCategory(relPath string) string {
	s := strings.ToLower(relPath)
	switch {
	case strings.Contains(s, "runbook"):
		return "runbook"
	case strings.Contains(s, "learn"):
		return "learning"
	case strings.Contains(s, "client"):
		return "client"
	case strings.Contains(s, "vendor"):
		return "vendor"
	case strings.Contains(s, "infra"):
		return "infra"
	case strings.Contains(s, "pattern"):
		return "pattern"
	case strings.Contains(s, "onboarding"):
		return "onboarding"
	default:
		return "uncategorized"
	}
}
