package binpath

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func exe(t *testing.T, mode os.FileMode) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "tool")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Chmod after create: the umask would otherwise hide group/other bits.
	if err := os.Chmod(p, mode); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestAcceptsAnExecutableOwnedByTheCaller(t *testing.T) {
	if err := Check(exe(t, 0o755)); err != nil {
		t.Fatal(err)
	}
}

func TestRejectsABareNameBecauseTheEpilogHasNoPATH(t *testing.T) {
	for _, p := range []string{"nvidia-smi", "./nvidia-smi", "bin/scontrol"} {
		if err := Check(p); !errors.Is(err, ErrNotAbsolute) {
			t.Errorf("%q: want ErrNotAbsolute, got %v", p, err)
		}
	}
}

func TestRejectsUnsafeOrMissingFiles(t *testing.T) {
	cases := map[string]string{
		"missing":        filepath.Join(t.TempDir(), "absent"),
		"directory":      t.TempDir(),
		"not executable": exe(t, 0o644),
		"group-writable": exe(t, 0o775),
		"world-writable": exe(t, 0o757),
	}
	for name, p := range cases {
		if err := Check(p); err == nil {
			t.Errorf("%s: accepted %s", name, p)
		}
	}
}

func TestFollowsSymlinksToTheRealFile(t *testing.T) {
	target := exe(t, 0o755)
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(target, link); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := Check(link); err != nil {
		t.Fatal(err)
	}
}

func TestOwnerRule(t *testing.T) {
	if checkOwner(0, 1000) != nil || checkOwner(1000, 1000) != nil {
		t.Error("root and the invoking user must be accepted")
	}
	if err := checkOwner(1001, 0); err == nil || !strings.Contains(err.Error(), "uid 1001") {
		t.Errorf("a file owned by another user must be refused when running as root: %v", err)
	}
}
