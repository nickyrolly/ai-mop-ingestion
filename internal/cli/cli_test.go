package cli

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCollectMarkdownFilesRecursive menjaga bug v0.1.0: ingest hanya membaca
// level atas folder, sehingga learnings/, vendors/, patterns/ terlewat.
func TestCollectMarkdownFilesRecursive(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "README.md"), "# root\n")
	mustWrite(t, filepath.Join(root, "runbooks", "RB-1.md"), "## a\n")
	mustWrite(t, filepath.Join(root, "learnings", "2026", "catatan.md"), "## b\n")
	mustWrite(t, filepath.Join(root, "vendors", "aws.md"), "## c\n")
	mustWrite(t, filepath.Join(root, "vendors", "note.txt"), "bukan markdown\n")
	mustWrite(t, filepath.Join(root, ".hidden", "secret.md"), "## d\n")
	mustWrite(t, filepath.Join(root, ".dotfile.md"), "## e\n")

	got, err := collectMarkdownFiles(root)
	if err != nil {
		t.Fatalf("collectMarkdownFiles: %v", err)
	}
	want := []string{
		filepath.Join(root, "README.md"),
		filepath.Join(root, "learnings", "2026", "catatan.md"),
		filepath.Join(root, "runbooks", "RB-1.md"),
		filepath.Join(root, "vendors", "aws.md"),
	}
	if len(got) != len(want) {
		t.Fatalf("jumlah file = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("files[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestCollectMarkdownFilesSingleFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "satu.md")
	mustWrite(t, p, "## a\n")
	got, err := collectMarkdownFiles(p)
	if err != nil {
		t.Fatalf("collectMarkdownFiles(file): %v", err)
	}
	if len(got) != 1 || got[0] != p {
		t.Fatalf("got %v", got)
	}
}

// TestSourceID menjaga bug v0.1.0: kunci dedup memakai basename sehingga
// clients/acme/index.md menimpa clients/beta/index.md.
func TestSourceID(t *testing.T) {
	root := filepath.Join(t.TempDir(), "knowledge-base")
	cases := map[string]string{
		filepath.Join(root, "runbooks", "RB-1.md"):         "runbooks/RB-1.md",
		filepath.Join(root, "clients", "acme", "index.md"): "clients/acme/index.md",
		filepath.Join(root, "clients", "beta", "index.md"): "clients/beta/index.md",
		filepath.Join(root, "README.md"):                   "README.md",
	}
	seen := map[string]bool{}
	for in, want := range cases {
		got := sourceID(root, in)
		if got != want {
			t.Errorf("sourceID(%q) = %q, want %q", in, got, want)
		}
		if seen[got] {
			t.Errorf("kunci dokumen duplikat: %q", got)
		}
		seen[got] = true
	}

	// root menunjuk file itu sendiri.
	one := filepath.Join(root, "runbooks", "RB-1.md")
	if got := sourceID(one, one); got != "RB-1.md" {
		t.Errorf("sourceID untuk -dir file = %q, want RB-1.md", got)
	}
}

// TestFindKBRoot menjaga agar ingest satu file memakai kunci & kategori yang
// sama dengan ingest folder penuh.
func TestFindKBRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "knowledge-base")
	file := filepath.Join(root, "runbooks", "RB-WP-001.md")
	mustWrite(t, file, "## a\n")

	if got := findKBRoot(file); got != root {
		t.Errorf("findKBRoot = %q, want %q", got, root)
	}
	if got := sourceID(findKBRoot(file), file); got != "runbooks/RB-WP-001.md" {
		t.Errorf("kunci dokumen = %q", got)
	}

	// Tanpa folder bernama knowledge-base: deteksi lewat subfolder runbooks.
	alt := filepath.Join(t.TempDir(), "kb")
	altFile := filepath.Join(alt, "runbooks", "x.md")
	mustWrite(t, altFile, "## a\n")
	if got := findKBRoot(altFile); got != alt {
		t.Errorf("findKBRoot(alt) = %q, want %q", got, alt)
	}
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
