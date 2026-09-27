//go:build windows || plan9

package main

import "errors"

type syslogSink interface {
	Err(string) error
	Warning(string) error
	Info(string) error
	Close() error
}

func openSyslog() (syslogSink, error) {
	return nil, errors.New("syslog is not available on this platform")
}
