#!/bin/sh
# Entrypoint wrapper for the opt-in, arbitrary-UID `vllm-dev-nonroot` CPU dev
# image. Same $HOME/$USER/passwd fixup as
# docker/entrypoints/vllm-nonroot-entrypoint.sh (vllm-openai-nonroot), but
# ends in an interactive shell instead of `vllm serve`.

set -eu

if [ -z "${HOME:-}" ] || [ ! -w "${HOME}" ]; then
    if [ -w /home/dev ]; then
        export HOME=/home/dev
    else
        if _h="$(mktemp -d /tmp/dev-home.XXXXXX 2>/dev/null)"; then
            export HOME="$_h"
            chmod 0700 "$HOME" 2>/dev/null || true
        else
            export HOME=/tmp
        fi
        unset _h
    fi
fi

if ! cd . 2>/dev/null; then
    cd "$HOME"
fi

if [ -z "${USER:-}" ]; then
    export USER=dev
fi
if [ -z "${LOGNAME:-}" ]; then
    export LOGNAME="$USER"
fi

_passwd_file="${VLLM_PASSWD_FILE:-/etc/passwd}"
_uid="$(id -u)"
if [ -w "$_passwd_file" ] \
    && ! awk -F: -v u="$_uid" '$3==u {found=1; exit} END {exit !found}' "$_passwd_file" 2>/dev/null; then
    printf 'dev:x:%s:%s:dev:%s:/bin/bash\n' \
        "$_uid" "$(id -g)" "$HOME" >> "$_passwd_file"
fi
unset _uid _passwd_file

exec bash "$@"
