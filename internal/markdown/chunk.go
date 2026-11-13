// Package markdown mem-parsing dokumen Markdown (frontmatter YAML + chunking
// per heading H2/H3) menjadi unit teks yang siap di-embedding.
package markdown

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

// MaxChunkChars batas lunak panjang teks chunk sebelum dikirim ke model
// embedding (bge-m3 punya konteks 8192 token).
const MaxChunkChars = 7000

// bk3 adalah tiga backtick ("```") sebagai konstanta, agar tidak ada literal
// backtick di dalam source Go.
const bk3 = "\x60\x60\x60"

// isHeadingLine melaporkan apakah baris heading ATX level 2 atau 3
// (CommonMark menuliskan '# ' atau '##'/'### ' + spasi).
func isHeadingLine(line string) bool {
	t := strings.TrimRight(line, "\r")
	if len(t) < 3 || t[0] != '#' || t[1] != '#' {
		return false
	}
	level := 2
	if t[2] == '#' {
		level = 3
		if len(t) < 4 {
			return false
		}
		return t[3] == ' ' || t[3] == '\t'
	}
	return t[2] == ' ' || t[2] == '\t' && level >= 2
}

// headingTitle mereturn teks heading tanpa '#' dan whitespace.
func headingTitle(line string) string {
	t := strings.TrimRight(line, "\r")
	i := 0
	for i < len(t) && t[i] == '#' {
		i++
	}
	t = strings.TrimSpace(t[i:])
	// strip trailing closed-ATX '#'
	for strings.HasSuffix(t, "#") {
		t = strings.TrimSpace(t[:len(t)-1])
	}
	return t
}

// isFenceLine bagitu apakah baris adalah fence code block (```/~~~).
func isFenceLine(line string) bool {
	t := strings.TrimLeft(line, " \t")
	return strings.HasPrefix(t, bk3) || strings.HasPrefix(t, "~~~")
}

// section satu potongan isi dokumen: heading + body.
type section struct {
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
}

// Chunk satu unit teks untuk embedding: satu topik/SOP utuh.
type Chunk struct {
	SourceFile   string // basename file — kunci dedup saat re-ingest
	SectionTitle string // teks heading pembuka chunk
	Content      string // isi section (tanpa baris heading)
	Category     string // inferensi dari path file
	TenantID     string // dari frontmatter, default "global"
}

// EmbeddingText mereturn representasi teks chunk untuk model embedding:
//
//	[File: <source_file> | Section: <heading>]
//	<isi_section>
func (c Chunk) EmbeddingText() string {
	return fmt.Sprintf("[File: %s | Section: %s]\n%s",
		c.SourceFile, c.SectionTitle, strings.TrimSpace(c.Content))
}

// ParseDocument mem-parse isi Markdown mentah menjadi (metadata, chunk).
func ParseDocument(sourceFile, src string) (Meta, []Chunk, error) {
	var meta Meta
	body := src

	if strings.HasPrefix(body, "---") {
		rest, fm, err := splitFrontmatter(body)
		if err != nil {
			return meta, nil, fmt.Errorf("parsing frontmatter %s: %w", sourceFile, err)
		}
		meta = fm
		body = rest
	}

	tenantID := strings.TrimSpace(meta.Tenant)
	if tenantID == "" {
		tenantID = "global"
	}
	category := InferCategory(sourceFile)

	sections, preamble, err := splitSections(body)
	if err != nil {
		return meta, nil, fmt.Errorf("chunking %s: %w", sourceFile, err)
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
			SourceFile:   sourceFile,
			SectionTitle: title,
			Content:      content,
			Category:     category,
			TenantID:     tenantID,
		})
	}

	if len(sections) == 0 {
		// Tanpa heading H2/H3: satu chunk utuh.
		appendChunk(docTitle, preamble)
		return meta, chunks, nil
	}

	for i, s := range sections {
		content := s.content.String()
		// Preamble (paragraf pembuka sebelum heading pertama) digabung ke
		// chunk section pertama + deux, supaya konteks tidak hilang.
		if i == 0 && strings.TrimSpace(preamble) != "" {
			content = preamble + "\n" + content
		}
		// Fallback judul bila section tanpa heading以内.
		appendChunk(s.title, content)
	}
	return meta, chunks, nil
}

// TruncateEmbeddingText memangkas teks ke MaxChunkChars dengan penanda.
func TruncateEmbeddingText(t string) string {
	if len(t) <= MaxChunkChars {
		return t
	}
	return t[:MaxChunkChars] + "\n...[truncated]"
}

// splitFrontmatter memisah blok YAML di awal (---...---) dari sisa body.
func splitFrontmatter(src string) (string, Meta, error) {
	var meta Meta
	if !strings.HasPrefix(src, "---") {
		return src, meta, nil
	}
	body0 := src[3:]
	// Hilangkan newline(s) setelah '---' pembuka.
	body0 = strings.TrimLeft(body0, "\r\n")
	// finding closing '---' on its own line.
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
		return "", meta, fmt.Errorf("YAML frontmatter tidak valid: %w", err)
	}
	return body, meta, nil
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
		if !inFence && isHeadingLine(trimmed) {
			sections = append(sections, section{title: headingTitle(trimmed)})
			cur = len(sections) - 1
			continue
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
	first := ""
	found := false
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimRight(line, "\r")
		if len(t) >= 2 && t[0] == '#' && (t[1] == ' ' || t[1] == '\t') {
			first = strings.TrimSpace(stripATXClosing(t[1:]))
			found = true
			break
		}
	}
	if !found {
		return ""
	}
	return first
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
	out := make([]string, 0, len(strings.Split(s, "\n")))
	removed := false
	for _, line := range strings.Split(s, "\n") {
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
	default:
		return "uncategorized"
	}
}
