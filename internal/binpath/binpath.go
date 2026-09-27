// Package binpath checks that an external tool is safe to run as root.
//
// The Epilog runs "as user root" (slurm.conf, Epilog) and with no search
// path set ("for security reasons, these programs do not have a search path
// set", https://slurm.schedmd.com/prolog_epilog.html). Both facts point the
// same way: name every external binary by absolute path, and refuse one that
// someone other than root (or the invoking user) could have replaced.
package binpath

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// ErrNotAbsolute is returned for a bare name or relative path.
var ErrNotAbsolute = errors.New("must be an absolute path (the Epilog has no PATH, and a PATH lookup as root is a hijack risk)")

// Check reports why path should not be executed, or nil. It follows
// symlinks and checks the final file: absolute path, regular file,
// executable, owned by root or the effective user, and not writable by group
// or others. It does not check the permissions of parent directories.
func Check(path string) error {
	if !filepath.IsAbs(path) {
		return fmt.Errorf("%q: %w", path, ErrNotAbsolute)
	}
	fi, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if !fi.Mode().IsRegular() {
		return fmt.Errorf("%s: not a regular file", path)
	}
	perm := fi.Mode().Perm()
	if perm&0o111 == 0 {
		return fmt.Errorf("%s: not executable (mode %v)", path, perm)
	}
	if perm&0o022 != 0 {
		return fmt.Errorf("%s: writable by group or others (mode %v); refusing to run it as root", path, perm)
	}
	if uid, ok := ownerUID(fi); ok {
		if err := checkOwner(uid, uint32(os.Geteuid())); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

// checkOwner accepts root, or the user running the tool (so tests and
// non-root dry runs work).
func checkOwner(uid, euid uint32) error {
	if uid == 0 || uid == euid {
		return nil
	}
	return fmt.Errorf("owned by uid %d, not root or the invoking user (uid %d)", uid, euid)
}
