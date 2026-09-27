//go:build !windows && !plan9

package main

import "log/syslog"

// syslogSink is the part of *syslog.Writer this tool uses.
type syslogSink interface {
	Err(string) error
	Warning(string) error
	Info(string) error
	Close() error
}

// openSyslog connects to the local syslog socket. The tag is what to search
// for: `grep epilog-gpu-validator` on the syslog file, or `journalctl -t
// epilog-gpu-validator` where journald collects syslog.
func openSyslog() (syslogSink, error) {
	return syslog.New(syslog.LOG_DAEMON|syslog.LOG_INFO, "epilog-gpu-validator")
}
