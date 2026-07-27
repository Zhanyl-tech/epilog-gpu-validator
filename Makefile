BINARY  := epilog-gpu-validator
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: help build test vet lint scenarios install clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

build: ## Build the binary
	go build -ldflags '$(LDFLAGS)' -o bin/$(BINARY) ./cmd/$(BINARY)

test: ## Run tests
	go test -race -cover ./...

vet: ## go vet
	go vet ./...

lint: vet test ## vet + test

scenarios: build ## Every fault scenario, with the decision and exit code
	@printf "  %-15s %-9s %-6s %-4s %s\n" SCENARIO SEVERITY DRAIN EXIT REASON
	@printf "  %s\n" "$$(printf '─%.0s' $$(seq 1 92))"
	@for s in healthy remap-pending thermal pcie-degraded hw-slowdown ecc remap-failure missing; do \
		out=$$(SLURM_JOB_ID=42 SLURMD_NODENAME=gpu001 ./bin/$(BINARY) --simulate $$s --json --enforce 2>/dev/null); code=$$?; \
		printf "  %-15s " "$$s"; \
		echo "$$out" | python3 -c "\
import json,sys;\
raw=sys.stdin.read().strip();\
print(f'{\"n/a\":<9} {\"no\":<6} $$code    query failed — safe no-op') if not raw else None;\
d=json.loads(raw) if raw else None;\
print(f\"{d['worst_severity']:<9} {('yes' if d['drain'] else 'no'):<6} $$code    {d.get('reason','')[:44]}\") if d else None"; \
	done

install: build ## Install to /usr/local/bin (needs root)
	install -m 0755 bin/$(BINARY) /usr/local/bin/$(BINARY)
	@echo "now reference deploy/epilog.sh from slurm.conf Epilog="

clean:
	rm -rf bin
