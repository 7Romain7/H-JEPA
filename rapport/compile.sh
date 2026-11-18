#!/bin/bash
# Compile rapport.tex sur une machine de l'X (pas de LaTeX en local) et rapatrie rapport.pdf.
#   bash compile.sh [machine]
set -u
H=${1:-Lycaste}
HERE=$(cd "$(dirname "$0")" && pwd)
CL="$HERE/../x-cluster/cluster.py"
T=$(mktemp -d)
cd "$HERE" && tar czf - rapport.tex fig_*.pdf | base64 -w0 > "$T/src.b64"
split -b 7000 -d -a 3 "$T/src.b64" "$T/part_"
ex() { PYTHONIOENCODING=utf-8 python "$CL" exec --hosts "$H" --timeout 180 "$1" 2>&1 | grep -v -i "warning: conn\|quantum\|vulnerable\|upgraded"; }
ex 'rm -rf /tmp/$USER/rapport; mkdir -p /tmp/$USER/rapport' > /dev/null
for f in "$T"/part_*; do ex "printf %s '$(cat "$f")' >> /tmp/\$USER/rapport/src.b64" > /dev/null; done
ex 'cd /tmp/$USER/rapport && base64 -d src.b64 | tar xzf - && pdflatex -interaction=nonstopmode -halt-on-error rapport.tex > /dev/null; pdflatex -interaction=nonstopmode -halt-on-error rapport.tex > /dev/null; pdflatex -interaction=nonstopmode -halt-on-error rapport.tex > /dev/null; echo RC=$?; grep -E "^!|LaTeX Warning|Overfull" rapport.log | head -30; grep -A3 "^!" rapport.log | head; echo PDF64; base64 -w0 rapport.pdf' > "$T/out.txt"
sed -n '1,/PDF64/p' "$T/out.txt" | cut -c1-220
sed -n '/PDF64/,$p' "$T/out.txt" | tail -n +2 | tr -d '\n\r' | base64 -d > "$HERE/rapport.pdf" 2>/dev/null
ls -la "$HERE/rapport.pdf"
rm -rf "$T"
