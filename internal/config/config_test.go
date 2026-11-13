package config

import (
	"os"
	"testing"
)

func TestFromEnvCTLMode(t *testing.T) {
	// Default
	os.Unsetenv(EnvCTL)
	cfg := FromEnv()
	if cfg.CTLMode {
		t.Error("default CTLMode should be false")
	}

	// Enabled via true
	os.Setenv(EnvCTL, "true")
	cfg = FromEnv()
	if !cfg.CTLMode {
		t.Error("CTLMode should be true for 'true'")
	}

	// Enabled via 1
	os.Setenv(EnvCTL, "1")
	cfg = FromEnv()
	if !cfg.CTLMode {
		t.Error("CTLMode should be true for '1'")
	}

	// Disabled via false
	os.Setenv(EnvCTL, "false")
	cfg = FromEnv()
	if cfg.CTLMode {
		t.Error("CTLMode should be false for 'false'")
	}

	os.Unsetenv(EnvCTL)
}
