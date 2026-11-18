# Sonde exécutée sur chaque machine par snapshot.py (via `bash -s`). Lecture seule.
echo "UP=$(cut -d' ' -f1 /proc/uptime)"
echo "LOAD=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "NPROC=$(nproc)"
echo "MEM=$(free -m | awk '/^Mem/{print $2","$2-$7}')"
who | awk '{printf "WHO=%s,%s,%s\n", $1, $2, $NF}'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^/GPU=/'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
  while IFS=', ' read -r p m; do
    echo "GPROC=$p,$m,$(ps -o user:32= -p "$p" | tr -d ' '),$(ps -o args= -p "$p" | cut -c1-120 | tr ',' ' ')"
  done
# CPU instantané (2e trame de top, ~0,7 s), noms complets via ps ; on ignore la sonde elle-même
ps -eo pid=,user:40= > /tmp/.psc_probe_users.$$
top -b -n 2 -d 0.7 -w 200 | awk '/^top -/{f++} f==2 && $1 ~ /^[0-9]+$/ {print $1, $9, $10, $12}' |
  awk -v me="$USER" -v map=/tmp/.psc_probe_users.$$ '
    BEGIN {while ((getline l < map) > 0) {split(l, a, " "); U[a[1]] = a[2]}}
    {u = U[$1]; if (u == "") next
     if (u == me && $4 ~ /^(top|ps|awk|bash|sshd|sed|cut|sort|tr|nvidia-smi)$/) next
     c[u] += $2; m[u] += $3; n[u]++; if ($2 >= mx[u]) {mx[u] = $2; t[u] = $4}}
    END {for (u in c) if (c[u] > 0.5 || m[u] > 0.5) printf "PROC=%s,%.1f,%.1f,%d,%s\n", u, c[u], m[u], n[u], t[u]}'
rm -f /tmp/.psc_probe_users.$$
# Jobs PSC réellement vivants sur ce poste (runs_state.json local peut être périmé)
for h in $(grep -lx "$(hostname)" ~/psc/runs/*/host.txt 2>/dev/null); do
  d=${h%/host.txt}
  [ -f "$d/exit_code" ] && continue
  p=$(cat "$d/pid" 2>/dev/null) && kill -0 "$p" 2>/dev/null &&
    echo "PSCJOB=${d##*/},$(date -d @"$(stat -c %Y "$d/pid")" '+%Y-%m-%d %H:%M:%S')"
done
