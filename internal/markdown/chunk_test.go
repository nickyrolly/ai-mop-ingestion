package markdown

import (
	"strings"
	"testing"
	"unicode/utf8"
)

const fence = "\x60\x60\x60"

const sampleRunbook = `---
title: Health Check WordPress (Publik + Internal)
id: RB-WP-001
status: draft
tags: [runbook, wordpress, health-check]
---

# RB-WP-001 — Health Check WordPress (Publik + Internal)

Paragraf pembuka.

## Scope

Berlaku untuk aplikasi WordPress di server dpanel-wordpress.

## Steps

### 1. DNS

dig +short www.okcarlomboktransport.com A

### 2. HTTP body

periksa ukuran body halaman fall-back.

` + "```" + `bash
curl -s https://www.okcarlomboktransport.com | wc -c
` + "```" + `

## Changelog

- 2026-10-31 publish.
`

func TestParseRunbook(t *testing.T) {
	meta, chunks, err := ParseDocument("runbooks/RB-WP-001-health-check.md", sampleRunbook)
	if err != nil {
		t.Fatalf("ParseDocument: %v", err)
	}
	if meta.ID != "RB-WP-001" {
		t.Errorf("meta.ID = %q", meta.ID)
	}
	if len(meta.Tags) != 3 {
		t.Errorf("meta.Tags = %v", meta.Tags)
	}
	if len(chunks) != 4 {
		t.Fatalf("chunk count = %d, want 4: %v", len(chunks), titles(chunks))
	}

	// H3 harus membawa breadcrumb H2 induknya.
	want := []string{"Scope", "Steps > 1. DNS", "Steps > 2. HTTP body", "Changelog"}
	for i, w := range want {
		if chunks[i].SectionTitle != w {
			t.Errorf("chunks[%d].SectionTitle = %q, want %q", i, chunks[i].SectionTitle, w)
		}
	}

	// Isi chunk H3 utuh, termasuk fenced code block.
	if !strings.Contains(chunks[2].Content, "curl -s https://www.okcarlomboktransport.com") {
		t.Errorf("chunk 2. HTTP body kehilangan fenced code block: %q", snippet(chunks[2].Content))
	}
	// Preamble menyatu dengan chunk pertama.
	if !strings.Contains(chunks[0].Content, "Paragraf pembuka") {
		t.Errorf("preamble hilang di chunk pertama: %q", snippet(chunks[0].Content))
	}
	// Identitas dokumen: path relatif + kategori dari path.
	if chunks[0].SourceFile != "runbooks/RB-WP-001-health-check.md" {
		t.Errorf("SourceFile = %q", chunks[0].SourceFile)
	}
	if chunks[0].Category != "runbook" {
		t.Errorf("Category = %q, want runbook", chunks[0].Category)
	}
	if chunks[0].DocID != "RB-WP-001" || chunks[0].DocTitle == "" {
		t.Errorf("metadata dokumen hilang: %+v", chunks[0])
	}

	// Teks embedding membawa judul dokumen + breadcrumb section.
	et := chunks[1].EmbeddingText()
	for _, must := range []string{
		"[File: runbooks/RB-WP-001-health-check.md",
		"Health Check WordPress",
		"RB-WP-001",
		"Section: Steps > 1. DNS]",
		"dig +short",
	} {
		if !strings.Contains(et, must) {
			t.Errorf("EmbeddingText tidak memuat %q:\n%s", must, et)
		}
	}
}

func TestH2H3Breadcrumb(t *testing.T) {
	src := "## A\nau\n### B\nbu\n## C\ncu\n"
	_, chunks, err := ParseDocument("x.md", src)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(chunks), 3; got != want {
		t.Fatalf("count = %d, want %d", got, want)
	}
	if chunks[1].SectionTitle != "A > B" {
		t.Fatalf("breadcrumb salah: %+v", chunks[1].SectionTitle)
	}
	if chunks[1].Content != "bu" {
		t.Fatalf("isi chunk B salah: %q", chunks[1].Content)
	}
	// H2 baru mereset induk: C tidak boleh jadi "A > C".
	if chunks[2].SectionTitle != "C" {
		t.Fatalf("H2 setelah H3 tidak reset: %q", chunks[2].SectionTitle)
	}
}

func TestFenceNotSplit(t *testing.T) {
	src := "## B\n\na\n" + fence + "\n## intermediate\n" + fence + "\n\nb\n## C\n\nc\n"
	_, chunks, err := ParseDocument("x.md", src)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("fence-awareness salah: %v", titles(chunks))
	}
	if !strings.Contains(chunks[0].Content, "## intermediate") {
		t.Errorf("heading di dalam fence tidak boleh jadi batas chunk")
	}
}

func TestUnclosedFence(t *testing.T) {
	if _, _, err := ParseDocument("bad.md", "## A\n"+fence+"\nkode\n"); err == nil {
		t.Fatal("harus error: fenced code block tidak ditutup")
	}
}

func TestFallbackNoHeading(t *testing.T) {
	_, chunks, err := ParseDocument("memo.md", "---\ntitle: Catatan\n---\nHanya teks bebas.\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].SectionTitle != "Catatan" {
		t.Fatalf("fallback salah: %+v", chunks)
	}
}

func TestUnclosedFrontmatter(t *testing.T) {
	if _, _, err := ParseDocument("bad.md", "---\ntitle: x\n\nbody"); err == nil {
		t.Fatal("harus error: frontmatter tidak ditutup")
	}
}

// TestLenientFrontmatter menjaga kasus nyata di knowledge-base: frontmatter
// dengan ': ' di dalam nilai membuat YAML invalid, tapi isi dokumen tetap harus
// ter-ingest.
func TestLenientFrontmatter(t *testing.T) {
	src := `---
title: Aturan Bahasa Respons AI
id: KB-L-001
status: active — dicek setiap pesan
tenant: internal
tags: [learning, bahasa]
source: observasi model (contoh nyata di sesi 2026-11-01: "pervention")
---

## Dasar

isi
`
	meta, chunks, err := ParseDocument("learnings/x.md", src)
	if err != nil {
		t.Fatalf("ParseDocument tidak boleh gagal: %v", err)
	}
	if meta.ParseWarning == "" {
		t.Error("ParseWarning harus terisi untuk frontmatter invalid")
	}
	if meta.ID != "KB-L-001" || meta.Title != "Aturan Bahasa Respons AI" {
		t.Errorf("metadata longgar salah: %+v", meta)
	}
	if meta.Tenant != "internal" {
		t.Errorf("tenant = %q, want internal", meta.Tenant)
	}
	if len(meta.Tags) != 2 {
		t.Errorf("tags = %v", meta.Tags)
	}
	if len(chunks) != 1 || chunks[0].SectionTitle != "Dasar" {
		t.Fatalf("chunk salah: %+v", chunks)
	}
}

func TestMidDocumentRuleIsNotFrontmatter(t *testing.T) {
	_, chunks, err := ParseDocument("x.md", "## S\n\n---\n\nisi\n")
	if err != nil {
		t.Fatalf("'---' di tengah dokumen tidak boleh dianggap frontmatter: %v", err)
	}
	if len(chunks) != 1 || !strings.Contains(chunks[0].Content, "---") {
		t.Fatalf("chunk S salah: %+v", chunks)
	}
}

func TestTenantDefaultAndOverride(t *testing.T) {
	_, chunks, err := ParseDocument("t.md", "## S\nisi\n")
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].TenantID != "global" {
		t.Errorf("TenantID default = %q, want global", chunks[0].TenantID)
	}

	_, chunks, err = ParseDocument("t.md", "---\ntenant: okcar\n---\n## S\nisi\n")
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].TenantID != "okcar" {
		t.Errorf("TenantID = %q, want okcar", chunks[0].TenantID)
	}
}

func TestInferCategory(t *testing.T) {
	cases := map[string]string{
		"runbooks/a.md":                "runbook",
		"learnings/2026/x.md":          "learning",
		"clients/acme/profile.md":      "client",
		"vendors/aws.md":               "vendor",
		"infrastructure/servers/x.md":  "infra",
		"patterns/p1.md":               "pattern",
		"onboarding/start.md":          "onboarding",
		"random/other.md":              "uncategorized",
		"knowledge-base/runbooks/x.md": "runbook",
	}
	for in, want := range cases {
		if got := InferCategory(in); got != want {
			t.Errorf("InferCategory(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPathNormalized(t *testing.T) {
	_, chunks, err := ParseDocument("runbooks\\sub\\..\\a.md", "## S\nisi\n")
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].SourceFile != "runbooks/a.md" {
		t.Errorf("SourceFile = %q, want runbooks/a.md", chunks[0].SourceFile)
	}
	if chunks[0].Category != "runbook" {
		t.Errorf("Category = %q, want runbook", chunks[0].Category)
	}
}

func TestTruncateRuneSafe(t *testing.T) {
	// 'é' 2 byte jatuh tepat di batas MaxChunkChars.
	long := strings.Repeat("a", MaxChunkChars-1) + "é" + strings.Repeat("b", 100)
	out := TruncateEmbeddingText(long)
	if !utf8.ValidString(out) {
		t.Fatalf("hasil truncate bukan UTF-8 valid: %.40q", out[max(0, len(out)-40):])
	}
	if !strings.HasSuffix(out, "...[truncated]") {
		t.Errorf("tanpa penanda truncate")
	}
	if len(out) > MaxChunkChars+len("\n...[truncated]") {
		t.Errorf("terlalu panjang: %d", len(out))
	}
	if !IsTruncated(long) {
		t.Error("IsTruncated harus true untuk teks panjang")
	}

	short := "halo"
	if TruncateEmbeddingText(short) != short || IsTruncated(short) {
		t.Error("string pendek tidak boleh berubah")
	}
}

func TestIsHeadingLine(t *testing.T) {
	cases := map[string]bool{
		"## Scope":   true,
		"### 1. DNS": true,
		"# H1":       false,
		"#### H4":    false,
		"##":         false,
		"##\tjudul":  true,
		"teks biasa": false,
		"####### x":  false,
	}
	for in, want := range cases {
		if got := isHeadingLine(in); got != want {
			t.Errorf("isHeadingLine(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestCRLF(t *testing.T) {
	src := "---\r\nid: X\r\n---\r\n## A\r\nisi\r\n"
	meta, chunks, err := ParseDocument("x.md", src)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ID != "X" {
		t.Errorf("meta.ID = %q", meta.ID)
	}
	if len(chunks) != 1 || chunks[0].SectionTitle != "A" {
		t.Errorf("CRLF tidak ditangani: %+v", chunks)
	}
}

func titles(chunks []Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.SectionTitle
	}
	return out
}

func snippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 80 {
		return s[:80]
	}
	return s
}
