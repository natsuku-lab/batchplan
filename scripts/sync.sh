#!/usr/bin/env bash
# Sync this checkout to ~/batchplan on another machine (e.g. a GPU box) and run a command there.
#   REMOTE=user@host scripts/sync.sh 'cargo test --release'
# Extra ssh flags go in SSH_OPTS, e.g. SSH_OPTS='-4 -i ~/.ssh/id_ed25519'.
set -euo pipefail
: "${REMOTE:?set REMOTE=user@host}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-}"
cd "$(dirname "$0")/.."
# Local-only paths (see .gitignore): excluded paths neither sync nor get deleted on the remote,
# so its build cache and generated data survive.
rsync -az --delete --exclude .git --exclude target --exclude data --exclude '.venv*' --exclude CLAUDE.local.md \
  -e "$SSH" ./ "$REMOTE:batchplan/"
$SSH "$REMOTE" "source ~/.cargo/env && cd ~/batchplan && ${*:-cargo build --release}"
