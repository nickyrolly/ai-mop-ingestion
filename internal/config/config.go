// Package config resolusi konfigurasi runtime dari environment variable.
package config

import (
	"os"
	"strconv"
)

const (
	DefaultOllamaURL   = "http://localhost:11434"
	DefaultOllamaModel = "bge-m3"
	DefaultDim         = 1024
	DefaultDSN         = "postgres://postgres:postgres@localhost:5433/aimop_rag?sslmode=disable"
	DefaultTable       = "knowledge_chunks"
)

// Nama environment variable.
const (
	EnvOllamaURL    = "OLLAMA_URL"
	EnvOllamaModel  = "OLLAMA_MODEL"
	EnvEmbeddingDim = "EMBEDDING_DIM"
	EnvPGDSN        = "PG_DSN"
	EnvKBDir        = "AI_MOP_KB_DIR"
)

// Config mengontrol perilaku ingestion, embedding, dan penyimpanan.
type Config struct {
	OllamaURL    string
	OllamaModel  string
	EmbeddingDim int
	PGDSN        string
	KBDir        string
	SQLTable     string
}

// OllamaEmbeddingsURL mereturn endpoint /api/embeddings Ollama.
func (c Config) OllamaEmbeddingsURL() string {
	return c.OllamaURL + "/api/embeddings"
}

// FromEnv membangun Config dari environment variable, pakai default jika kosong.
func FromEnv() Config {
	cfg := Config{
		OllamaURL:    DefaultOllamaURL,
		OllamaModel:  DefaultOllamaModel,
		EmbeddingDim: DefaultDim,
		PGDSN:        DefaultDSN,
		SQLTable:     DefaultTable,
	}
	if v := os.Getenv(EnvOllamaURL); v != "" {
		cfg.OllamaURL = v
	}
	if v := os.Getenv(EnvOllamaModel); v != "" {
		cfg.OllamaModel = v
	}
	if v := os.Getenv(EnvEmbeddingDim); v != "" {
		if d, err := strconv.Atoi(v); err == nil && d > 0 {
			cfg.EmbeddingDim = d
		}
	}
	if v := os.Getenv(EnvPGDSN); v != "" {
		cfg.PGDSN = v
	}
	if v := os.Getenv(EnvKBDir); v != "" {
		cfg.KBDir = v
	}
	return cfg
}
