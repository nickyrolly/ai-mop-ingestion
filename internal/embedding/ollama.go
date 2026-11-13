// Package embedding menyediakan client HTTP ke Ollama /api/embeddings.
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

// Client mightyembedding Ollama.
type Client struct {
	Endpoint string // URL penuh /api/embeddings
	Model    string
	Dim      int
	hc       *http.Client
}

// NewClient membuat client embedding Ollama.
func NewClient(endpoint, model string, dim int) *Client {
	return &Client{
		Endpoint: endpoint,
		Model:    model,
		Dim:      dim,
		hc:       &http.Client{Timeout: 120 * time.Second},
	}
}

type requestPayload struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type embeddingsResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Embed mengirim prompt ke Ollama dan mereturn vektor embedding.
// Error dikembalikan bila HTTP gagal, status bukan 200, JSON tidak bisa
// didecode, atau dimensi vektor berbeda dari konfigurasi.
func (c *Client) Embed(ctx context.Context, prompt string) ([]float32, error) {
	payload, err := json.Marshal(requestPayload{Model: c.Model, Prompt: prompt})
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

	var out embeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode respons Ollama: %w", err)
	}
	if len(out.Embedding) != c.Dim {
		return nil, fmt.Errorf("dimensi embedding %d tidak sama dengan konfigurasi %d",
			len(out.Embedding), c.Dim)
	}
	return out.Embedding, nil
}
