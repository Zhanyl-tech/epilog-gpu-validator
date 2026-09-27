//go:build !unix

package binpath

import "os"

// ownerUID is unavailable off unix; the tool only targets Linux nodes.
func ownerUID(os.FileInfo) (uint32, bool) { return 0, false }
