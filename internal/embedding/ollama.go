// Package embedding menyediakan client HTTP ke endpoint embedding Ollama
// (/api/embed), dengan dukungan batch agar tidak perlu satu request per chunk.
package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// MaxBatch jumlah teks maksimum per request ke Ollama.
const MaxBatch = 16

// Client membungkus endpoint embedding Ollama.
type Client struct {
	Endpoint string // URL penuh /api/embed
	Model    string
	Dim      int
	hc       *http.Client
}

// NewClient membuat client embedding Ollama. baseURL contoh: http://localhost:11434
func NewClient(baseURL, model string, dim int) *Client {
	return &Client{
		Endpoint: strings.TrimRight(baseURL, "/") + "/api/embed",
		Model:    model,
		Dim:      dim,
		hc:       &http.Client{Timeout: 300 * time.Second},
	}
}

type requestPayload struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed mengirim satu prompt ke Ollama dan mereturn vektor embedding.
func (c *Client) Embed(ctx context.Context, prompt string) ([]float32, error) {
	vecs, err := c.EmbedBatch(ctx, []string{prompt})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch mengirim sekumpulan prompt (dipecah per MaxBatch) dan mereturn
// vektor untuk masing-masing prompt, berurutan sesuai input.
func (c *Client) EmbedBatch(ctx context.Context, prompts []string) ([][]float32, error) {
	if len(prompts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(prompts))
	for start := 0; start < len(prompts); start += MaxBatch {
		end := start + MaxBatch
		if end > len(prompts) {
			end = len(prompts)
		}
		vecs, err := c.embedOnce(ctx, prompts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *Client) embedOnce(ctx context.Context, prompts []string) ([][]float32, error) {
	payload, err := json.Marshal(requestPayload{Model: c.Model, Input: prompts})
	if err != nil {
		return nil, fmt.Errorf("encode payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("membangun request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request ke Ollama gagal: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("Ollama status %s: %s", resp.Status, strings.TrimSpace(string(msg)))
	}

	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode respons Ollama: %w", err)
	}
	if len(out.Embeddings) != len(prompts) {
		return nil, fmt.Errorf("Ollama mereturn %d vektor untuk %d input",
			len(out.Embeddings), len(prompts))
	}
	for i, v := range out.Embeddings {
		if len(v) != c.Dim {
			return nil, fmt.Errorf("dimensi embedding ke-%d = %d, tidak sama dengan konfigurasi %d",
				i+1, len(v), c.Dim)
		}
	}
	return out.Embeddings, nil
}
