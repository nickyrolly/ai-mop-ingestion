// Package config resolusi konfigurasi runtime dari environment variable.
package config

import (
	"os"
	"strconv"
	"strings"
)

const (
	DefaultOllamaURL   = "http://localhost:11434"
	DefaultOllamaModel = "bge-m3"
	DefaultDim         = 1024
	DefaultDSN         = "postgres://postgres:postgres@localhost:5433/aimop_rag?sslmode=disable"
	DefaultTable       = "knowledge_chunks"
	DefaultJobs        = 4
	DefaultCTLMode     = false
)

// Nama environment variable.
const (
	EnvOllamaURL    = "OLLAMA_URL"
	EnvOllamaModel  = "OLLAMA_MODEL"
	EnvEmbeddingDim = "EMBEDDING_DIM"
	EnvPGDSN        = "PG_DSN"
	EnvKBDir        = "AI_MOP_KB_DIR"
	EnvSQLTable     = "AI_MOP_TABLE"
	EnvJobs         = "AI_MOP_JOBS"
	EnvCTL          = "AI_MOP_CTL"
)

// Config mengontrol perilaku ingestion, embedding, dan penyimpanan.
type Config struct {
	OllamaURL    string
	OllamaModel  string
	EmbeddingDim int
	PGDSN        string
	KBDir        string
	SQLTable     string
	Jobs         int
	CTLMode      bool
}

// OllamaEmbedURL mereturn endpoint /api/embed Ollama.
func (c Config) OllamaEmbedURL() string {
	return c.OllamaURL + "/api/embed"
}

// FromEnv membangun Config dari environment variable, pakai default jika kosong.
func FromEnv() Config {
	cfg := Config{
		OllamaURL:    DefaultOllamaURL,
		OllamaModel:  DefaultOllamaModel,
		EmbeddingDim: DefaultDim,
		PGDSN:        DefaultDSN,
		SQLTable:     DefaultTable,
		Jobs:         DefaultJobs,
		CTLMode:      DefaultCTLMode,
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
	if v := os.Getenv(EnvSQLTable); v != "" {
		cfg.SQLTable = v
	}
	if v := os.Getenv(EnvJobs); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Jobs = n
		}
	}
	if v := os.Getenv(EnvCTL); v != "" {
		v = strings.ToLower(strings.TrimSpace(v))
		cfg.CTLMode = (v == "1" || v == "true" || v == "yes" || v == "on")
	}
	return cfg
}
