BINARY  := epilog-gpu-validator
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: help build test vet fmt-check shellcheck integration check scenarios readme-table install clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

build: ## Build the binary
	go build -ldflags '$(LDFLAGS)' -o bin/$(BINARY) ./cmd/$(BINARY)

test: ## Run tests (race detector + coverage)
	go test -race -cover ./...

vet: ## go vet
	go vet ./...

fmt-check: ## Fail if any Go file is not gofmt-formatted
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "not gofmt-formatted:"; echo "$$out"; exit 1; fi

shellcheck: ## Lint the shell scripts
	shellcheck deploy/epilog.sh scripts/integration.sh

integration: ## Run the wrapper + binary under env -i with fake nvidia-smi/scontrol
	bash scripts/integration.sh

check: fmt-check vet test integration ## Everything CI runs except shellcheck/staticcheck/govulncheck

scenarios: ## The README scenario table (simulated; touches no hardware and no Slurm)
	@go run ./cmd/$(BINARY) --scenario-table

readme-table: scenarios ## Alias: print the table to paste between the README markers

install: build ## Install to /usr/local/bin (needs root)
	install -m 0755 bin/$(BINARY) /usr/local/bin/$(BINARY)
	@echo "then: install -d -m 0755 /etc/slurm/epilog.d && install -m 0755 deploy/epilog.sh /etc/slurm/epilog.d/50-gpu-validate"
	@echo "and run: /usr/local/bin/$(BINARY) --check-config (see README, Install)"

clean:
	rm -rf bin
