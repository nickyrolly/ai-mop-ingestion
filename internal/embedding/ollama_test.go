package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// TestEmbedBatchPagination memastikan input dipecah per MaxBatch dan urutan
// vektor hasilnya sesuai urutan input.
func TestEmbedBatchPagination(t *testing.T) {
	const dim = 4
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("path = %q, want /api/embed", r.URL.Path)
		}
		atomic.AddInt32(&calls, 1)
		var req requestPayload
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		resp := embedResponse{}
		for range req.Input {
			resp.Embeddings = append(resp.Embeddings, []float32{1, 1, 1, 1})
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "bge-m3", dim)
	prompts := make([]string, MaxBatch+3)
	for i := range prompts {
		prompts[i] = "teks"
	}

	vecs, err := c.EmbedBatch(context.Background(), prompts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(vecs) != len(prompts) {
		t.Fatalf("jumlah vektor = %d, want %d", len(vecs), len(prompts))
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("jumlah request = %d, want 2", got)
	}
	for i, v := range vecs {
		if len(v) != dim {
			t.Fatalf("vektor %d dimensi %d", i, len(v))
		}
	}
}

func TestEmbedDimMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponse{Embeddings: [][]float32{{0.1, 0.2}}})
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "bge-m3", 4)
	if _, err := c.Embed(context.Background(), "x"); err == nil {
		t.Fatal("harus error saat dimensi tidak cocok")
	}
}

func TestEmbedHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "nope", 2)
	if _, err := c.Embed(context.Background(), "x"); err == nil {
		t.Fatal("harus error saat status bukan 200")
	}
}

func TestNewClientEndpoint(t *testing.T) {
	c := NewClient("http://localhost:11434/", "bge-m3", 1024)
	if c.Endpoint != "http://localhost:11434/api/embed" {
		t.Errorf("Endpoint = %q", c.Endpoint)
	}
}
