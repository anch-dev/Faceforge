#!/usr/bin/env bash
# Re-publish compiler / test errors from a log as GitHub annotations, so they
# are visible through the API and on the run page without opening raw logs.
# Usage: annotate.sh LOGFILE TITLE
log="$1"; title="${2:-Build errors}"
[ -f "$log" ] || exit 0
errors=$(grep -E "error:|failed \(|Fatal error|BUILD FAILED|Test Suite .* failed|\*\* .* FAILED \*\*" "$log" \
  | grep -v "warning:" | awk '!seen[$0]++' | head -60)
[ -z "$errors" ] && errors=$(tail -40 "$log")
# Annotations are one line: encode newlines (%0A) and escape % first.
msg=$(printf '%s' "$errors" | sed -e 's/%/%25/g' | awk 'BEGIN{ORS="%0A"} {print}')
echo "::error title=${title}::${msg}"
