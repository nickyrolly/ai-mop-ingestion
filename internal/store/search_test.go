package store

import (
	"strings"
	"testing"
)

// TestBuildTSQuery menjaga perilaku jalur kata kunci: kata fungsi dibuang,
// duplikat dihapus, dan token pasti (identifier ber-underscore) tetap utuh.
func TestBuildTSQuery(t *testing.T) {
	cases := map[string]string{
		"mas web saya muter-muter terus gak bisa dibuka": "web | muter | terus | dibuka",
		"DISALLOW_FILE_MODS muncul saat update plugin":   "disallow_file_mods | muncul | saat | update | plugin",
		"kenapa versi aplikasi tidak sama dengan kode":   "versi | aplikasi | sama | kode",
		"cek sertifikat TLS yang mau expired":            "cek | sertifikat | tls | expired",
		"muter muter muter":                              "muter",
		"":                                               "",
		"dan atau yang":                                  "",
	}
	for in, want := range cases {
		if got := buildTSQuery(in); got != want {
			t.Errorf("buildTSQuery(%q) = %q, want %q", in, got, want)
		}
	}

	// Token yang bukan huruf/angka/underscore dibuang, sehingga tidak ada
	// operator to_tsquery (&|!<>():*) yang bisa lolos dari input pengguna.
	got := buildTSQuery(`'; DROP TABLE knowledge_chunks; -- &|!():*`)
	for _, term := range strings.Split(got, " | ") {
		if term == "" {
			continue
		}
		if strings.Trim(term, "abcdefghijklmnopqrstuvwxyz0123456789_") != "" {
			t.Errorf("token berisi karakter operator: %q (dari %q)", term, got)
		}
	}
}

// TestFuseRRF memastikan chunk yang muncul di kedua jalur naik, dan hasilnya
// deterministik saat skor seri.
func TestFuseRRF(t *testing.T) {
	vec := []Result{
		{ChunkID: 1, SectionTitle: "vektor-1", Similarity: 0.70},
		{ChunkID: 2, SectionTitle: "vektor-2", Similarity: 0.60},
		{ChunkID: 3, SectionTitle: "vektor-3", Similarity: 0.55},
	}
	txt := []Result{
		{ChunkID: 3, SectionTitle: "vektor-3", Similarity: 0.55, TextRank: 0.9},
		{ChunkID: 4, SectionTitle: "teks-1", Similarity: 0.40, TextRank: 0.5},
	}

	got := fuseRRF(vec, txt, DefaultRRFK)
	if len(got) != 4 {
		t.Fatalf("jumlah hasil = %d, want 4", len(got))
	}

	// Chunk 3 ada di dua jalur (rank 3 vektor + rank 1 teks) sehingga harus
	// mengalahkan chunk 1 yang hanya rank 1 vektor.
	if got[0].ChunkID != 3 {
		t.Errorf("peringkat 1 = chunk %d (%s), want chunk 3 (dua jalur)", got[0].ChunkID, got[0].SectionTitle)
	}
	if !got[0].TextMatched() {
		t.Error("chunk yang cocok jalur kata kunci harus TextMatched")
	}
	wantScore := 1.0/63 + 1.0/61
	if diff := float64(got[0].Score) - wantScore; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("skor RRF chunk 3 = %v, want %v", got[0].Score, wantScore)
	}

	// Chunk 4 hanya muncul di jalur teks dan tetap masuk hasil.
	var found bool
	for _, r := range got {
		if r.ChunkID == 4 {
			found = true
			if !r.TextMatched() {
				t.Error("chunk teks-saja harus TextMatched")
			}
		}
	}
	if !found {
		t.Error("chunk yang hanya muncul di jalur kata kunci hilang dari hasil fusi")
	}

	// Skor harus monoton menurun.
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Errorf("skor tidak monoton pada peringkat %d: %v > %v", i+1, got[i].Score, got[i-1].Score)
		}
	}
}

// TestFuseRRFStable memastikan urutan hasil deterministik saat semua skor sama.
func TestFuseRRFStable(t *testing.T) {
	vec := []Result{{ChunkID: 7}, {ChunkID: 5}, {ChunkID: 6}}
	got := fuseRRF(vec, nil, DefaultRRFK)
	for i, want := range []int64{7, 5, 6} {
		if got[i].ChunkID != want {
			t.Fatalf("hasil[%d] = chunk %d, want %d", i, got[i].ChunkID, want)
		}
	}
}

// TestSelectResults mencakup tiga perbaikan sekaligus: gerbang skor, batas
// chunk per dokumen, dan hit kata kunci yang tetap harus lolos gerbang.
func TestSelectResults(t *testing.T) {
	cands := []Result{
		{ChunkID: 1, SourceFile: "a.md", Similarity: 0.72},
		{ChunkID: 2, SourceFile: "a.md", Similarity: 0.70},
		{ChunkID: 3, SourceFile: "a.md", Similarity: 0.68},                // kena batas per-doc
		{ChunkID: 4, SourceFile: "b.md", Similarity: 0.45},                // di bawah ambang
		{ChunkID: 5, SourceFile: "c.md", Similarity: 0.30, TextRank: 0.4}, // derau jalur teks
		{ChunkID: 6, SourceFile: "d.md", Similarity: 0.62, TextRank: 0.2}, // teks + cosine cukup
	}
	opt := SearchOptions{TopK: 5, MinScore: 0.50, MaxPerDoc: 2}

	got := selectResults(cands, opt)
	ids := make([]int64, len(got))
	for i, r := range got {
		ids[i] = r.ChunkID
	}
	want := []int64{1, 2, 6}
	if len(ids) != len(want) {
		t.Fatalf("hasil = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("hasil = %v, want %v", ids, want)
		}
	}

	// Ambang 0 = gerbang nonaktif (dipakai untuk evaluasi pembanding).
	opt.MinScore = 0
	opt.MaxPerDoc = 0
	opt.TopK = 10
	if got := selectResults(cands, opt); len(got) != len(cands) {
		t.Errorf("tanpa gerbang dan tanpa batas per-doc, hasil = %d, want %d", len(got), len(cands))
	}
}

// TestSelectResultsEmpty memastikan query di luar domain berakhir tanpa hasil
// (CLI memetakannya ke exit code 3), bukan memaksa mengembalikan top-1.
func TestSelectResultsEmpty(t *testing.T) {
	cands := []Result{
		{ChunkID: 1, SourceFile: "a.md", Similarity: 0.48},
		{ChunkID: 2, SourceFile: "b.md", Similarity: 0.41},
	}
	if got := selectResults(cands, SearchOptions{TopK: 3, MinScore: 0.50, MaxPerDoc: 2}); len(got) != 0 {
		t.Fatalf("hasil = %d, want 0", len(got))
	}
}

// TestSearchOptionsNormalized memastikan default kandidat cukup untuk fusi.
func TestSearchOptionsNormalized(t *testing.T) {
	o := SearchOptions{TopK: 5}.normalized()
	if o.CandidateLimit != 50 {
		t.Errorf("CandidateLimit = %d, want 50", o.CandidateLimit)
	}
	o = SearchOptions{TopK: 30}.normalized()
	if o.CandidateLimit != 300 {
		t.Errorf("CandidateLimit = %d, want 300", o.CandidateLimit)
	}
	if o.MinScore != 0 || o.MaxPerDoc != 0 {
		t.Errorf("normalized tidak boleh mengisi MinScore/MaxPerDoc: %+v", o)
	}
}

// TestTableNameValidation memastikan sanitasi nama tabel menolak karakter berbahaya
// seperti spasi, titik koma, kutip, tanda hubung, atau operator SQL injection.
func TestTableNameValidation(t *testing.T) {
	valid := []string{
		"knowledge_chunks",
		"knowledge_chunks_2",
		"kb_chunks",
		"_private_chunks",
		"Chunks",
	}
	for _, name := range valid {
		if !tableNameRe.MatchString(name) {
			t.Errorf("nama tabel valid ditolak: %q", name)
		}
	}

	invalid := []string{
		"knowledge-chunks",              // tanda hubung
		"knowledge chunks",              // spasi
		"knowledge_chunks; DROP TABLE",  // SQL injection
		"chunks' OR 1=1",                // kutip tunggal
		`chunks"`,                       // kutip ganda
		"123chunks",                     // diawali angka
		"",                              // kosong
		"chunks--comment",               // komentar SQL
		"chunks/*block*/",               // komentar blok
	}
	for _, name := range invalid {
		if tableNameRe.MatchString(name) {
			t.Errorf("nama tabel tidak valid lolos: %q", name)
		}
	}
}
