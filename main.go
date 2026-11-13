// ai-mop-ingestion: ingestion dokumen Markdown knowledge-base AI-MOP ke
// PostgreSQL pgvector, dengan embedding Ollama (bge-m3) dan sub-perintah
// uji pencarian similarity cosine.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"ai-mop-ingestion/internal/cli"
)

func main() {
	// Ctrl+C / SIGTERM membatalkan ingest dengan rapi (transaksi per file
	// di-rollback), tanpa deadline buatan yang memotong ingest besar.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	os.Exit(cli.Run(ctx, os.Args[1:], os.Stdout))
}
