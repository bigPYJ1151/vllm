#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# `as` wrapper for Triton-CPU on hosts with AVX10.2 / APX (e.g. Diamond Rapids).
# LLVM prints a few instructions in forms that the pinned binutils rejects; this
# rewrites them in the generated assembly before calling the real assembler:
#  - vcmp<pred>bf16 aliases -> canonical `vcmpbf16 $imm, ...`
#  - APX NDD shift-by-one printed without the count (`sarl %eax, %ecx`)
#    -> `sarl $1, %eax, %ecx`
# It is found through COMPILER_PATH (see docker/Dockerfile.cpu) and only touches
# `*.s` arguments.
args=()
for a in "$@"; do
	if [[ "$a" == *.s && -f "$a" ]]; then
		# shellcheck disable=SC2016  # `$0`..`$7`, `$1` are assembler immediates
		sed -i -E \
			-e 's/vcmpeqbf16[[:space:]]+/vcmpbf16 $0, /;s/vcmpltbf16[[:space:]]+/vcmpbf16 $1, /;s/vcmplebf16[[:space:]]+/vcmpbf16 $2, /;s/vcmpunordbf16[[:space:]]+/vcmpbf16 $3, /' \
			-e 's/vcmpneqbf16[[:space:]]+/vcmpbf16 $4, /;s/vcmpnltbf16[[:space:]]+/vcmpbf16 $5, /;s/vcmpnlebf16[[:space:]]+/vcmpbf16 $6, /;s/vcmpordbf16[[:space:]]+/vcmpbf16 $7, /' \
			-e '/^[[:space:]]*(\{nf\}[[:space:]]*)?(sar|shr|shl|sal|rol|ror|rcl|rcr)[bwlq]?[[:space:]]+%cl,/!s/^([[:space:]]*(\{nf\}[[:space:]]*)?(sar|shr|shl|sal|rol|ror|rcl|rcr)[bwlq]?)[[:space:]]+(%[a-z0-9]+),[[:space:]]*(%[a-z0-9]+)[[:space:]]*(#.*)?$/\1 $1, \4, \5/' \
			"$a"
	fi
	args+=("$a")
done
exec /opt/binutils/bin/as "${args[@]}"
