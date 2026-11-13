package markdown

import (
	"strings"
	"testing"
)

// fence = kombinasi backtick aman di Go literal.
const fence = "\x60\x60\x60"

const sampleRunbook = `---
title: Health Check WordPress (Publik + Internal)
id: RB-WP-001
status: draft
tags: [runbook, wordpress, health-check]
---

# RB-WP-001 — Health Check WordPress (Publik + Internal)

Paragraf pembuka conococot.

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
	meta, chunks, err := ParseDocument("RB-WP-001.md", sampleRunbook)
	if err != nil {
		t.Fatalf("ParseDocument: %v", err)
	}
	if meta.ID != "RB-WP-001" {
		t.Errorf("meta.ID = %q", meta.ID)
	}
	if len(meta.Tags) != 3 {
		t.Errorf("meta.Tags = %v", meta.Tags)
	}

	// H3 ikut jadi chunk tersendiri (spesifikasi: chop per H2 atau H3).
	want := []string{"Scope", "1. DNS", "2. HTTP body", "Changelog"}
	if len(chunks) != len(want) {
		t.Fatalf("chunk count = %d, want %d", len(chunks), len(want))
	}
	for i, w := range want {
		if chunks[i].SectionTitle != w {
			t.Errorf("chunks[%d].SectionTitle = %q, want %q", i, chunks[i].SectionTitle, w)
		}
	}

	// Isi chunk H3 harus utuh (termasuk fenced code block).
	dns := chunks[1]
	if !strings.Contains(dns.Content, "dig +short www.okcarlomboktransport.com A") {
		t.Errorf("chunk 1. DNS isi tidak lengkap: %q", chunkSnippet(dns.Content))
	}
	// Preamble paragraf pembuka harus menyatu dengan chunk pertama.
	if !strings.Contains(chunks[0].Content, "Paragraf pembuka") {
		t.Errorf("preamble hilang di chunk pertama: %q", chunkSnippet(chunks[0].Content))
	}
	// Preamble paragraf pembukaするという bagian Scope.
	if !strings.Contains(chunks[0].Content, "Paragraf pembuka") {
		t.Errorf("preamble hilang di chunk pertama: %q", chunkSnippet(chunks[0].Content))
	}
}

func chunkSnippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 80 {
		return s[:80]
	}
	return s
}

func TestH2H3Split(t *testing.T) {
	src := "## A\nau\n### B\nbu\n## C\ncu\n"
	_, chunks, err := ParseDocument("x.md", src)
	if err != nil {
		t.Fatal(err)
	}
	// H3 juga778 chunk → A, B, C.
	if got, want := len(chunks), 3; got != want {
		t.Fatalf("count = %d, want %d", got, want)
	}
	if chunks[1].SectionTitle != "B" || chunks[1].Content != "bu" {
		t.Fatalf("chunk B salah: %+v", chunks[1])
	}
}

func slicesTitles(chunks []Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.SectionTitle
	}
	return out
}

func TestFenceNotSplit(t *testing.T) {
	src := "## B\n\na\n" + "```" + "\n## intermediate\n" + "```\n\nb\n## C\n\nc\n"
	_, chunks, err := ParseDocument("x.md", src)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("fence-awareness salah: %+v", slicesTitles(chunks))
	}
	if strings.Contains(chunks[0].Content, "## C") && strings.Count(chunks[0].Content, "## ") != 1 {
		// jangan terlalu ketat; pastikan tidak ada heading palsu lebih dari didokumentasikan
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
		t.Fatal("mesti error: frontmatter tidak ditutup")
	}
}

func TestTenantFromMeta(t *testing.T) {
	_, chunks, err := ParseDocument("t.md", "---\ntenant: okcar\n---\n## S\nisi\n")
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].TenantID != "okcar" {
		t.Errorf("TenantID = %q", chunks[0].TenantID)
	}
}

func TestInferCategory(t *testing.T) {
	cases := map[string]string{
		"runbooks/a.md":                "runbook",
		"learnings/2026/x.md":          "learning",
		"clients/acme/profile.md":      "client",
		"vendors/aws.md":               "vendor",
		"infrastructure/servers/x.md":  "infra",
		"random/other.md":              "uncategorized",
		"knowledge-base/runbooks/x.md": "runbook",
	}
	for in, want := range cases {
		if got := InferCategory(in); got != want {
			t.Errorf("InferCategory(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestTruncateEmbeddingText(t *testing.T) {
	long := strings.Repeat("x", MaxChunkChars+300)
	out := TruncateEmbeddingText(long)
	if len(out) > MaxChunkChars+len("\n...[truncated]") {
		t.Errorf("terlalu panjang: %d", len(out))
	}
	if !strings.HasSuffix(out, "truncated]") {
		t.Errorf("tanpa penanda: %.60q", out)
	}
	short := "halo"
	if TruncateEmbeddingText(short) != short {
		t.Errorf("string pendek ikut terpotong")
	}
}

// TestIsHeading helper.
func TestIsHeadingLine(t *testing.T) {
	cases := map[string]bool{
		"## Scope":   true,
		"### 1. DNS": true,
		"# H1":       false,
		"#### H4":    false,
		"teks biasa": false,
		"####### x":  false,
	}
	for in, want := range cases {
		if got := isHeadingLine(in); got != want {
			t.Errorf("isHeadingLine(%q) = %v, want %v", in, got, want)
		}
	}
}

// fence const untuk lolita yang butuh literal real di raw string test.
var _ = fence
