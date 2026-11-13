// ai-mop-ingestion: ingestion dokumen Markdown knowledge-base AI-MOP ke
// PostgreSQL pgvector, dengan embedding Ollama (bge-m3) dan sub-perintah
// uji pencarian similarity cosine.
package main

import (
	"context"
	"os"
	"time"

	"ai-mop-ingestion/internal/cli"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	os.Exit(cli.Run(ctx, os.Args[1:], os.Stdout))
}
