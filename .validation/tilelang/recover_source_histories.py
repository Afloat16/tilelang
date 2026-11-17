"""Rebuild complete reachable source histories, including intermediate commits."""
import base64
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
manifest = pathlib.Path(sys.argv[3]).resolve()
output.mkdir(parents=True, exist_ok=True)

def git(*args, check=True):
    p = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr)
    return p

initial = [line.split() for line in manifest.read_text().splitlines() if line.strip()]
missing = []
for i, (ref, sha) in enumerate(initial):
    exists = git("cat-file", "-e", f"{sha}^{{commit}}", check=False)
    if exists.returncode:
        remote = "https://github.com/tile-ai/tilelang.git" if "upstream-pr" in ref else "https://github.com/Afloat16/tilelang.git"
        result = git("fetch", "--filter=blob:none", "--no-tags", remote, sha, check=False)
        if result.returncode:
            missing.append({"ref": ref, "sha": sha, "error": result.stderr})
            continue
    git("update-ref", f"refs/remotes/initial-snapshot/{i}", sha)

names = set(git("log", "--all", "--format=", "--name-only", "--", "src", "tilelang").stdout.splitlines())
groups = {
    "template_header": sorted(p for p in names if "tl_templates/" in p and p.endswith("/common.h") and any(part in p for part in ("/cpu/", "/cpp/"))),
    "cpu_codegen": sorted(p for p in names if p.endswith("/codegen_c.cc")),
    "cython_adapter": sorted(p for p in names if "/cython/" in p and p.endswith(("/adapter.py", "/cython_wrapper.pyx"))),
}
histories = {}
for name, paths in groups.items():
    commits = list(dict.fromkeys(git("log", "--all", "--full-history", "--format=%H", "--", *paths).stdout.splitlines()))
    blobs = {}
    versions = []
    for commit in commits:
        entries = []
        for line in git("ls-tree", "-r", commit, "--", *paths).stdout.splitlines():
            meta, path = line.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind != "blob":
                continue
            entries.append({"path": path, "blob": sha})
            if sha not in blobs:
                blobs[sha] = git("cat-file", "blob", sha).stdout
        versions.append({"commit": commit, "entries": entries})
    histories[name] = {"paths": paths, "commits": versions, "blobs": blobs}
    print(json.dumps({"history": name, "paths": paths, "commits": len(commits), "unique_blobs": len(blobs)}), flush=True)
report = {
    "base": git("rev-parse", "HEAD").stdout.strip(),
    "initial_snapshot_refs": len(initial),
    "missing_initial_commits": missing,
    "current_refs": git("for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/upstream-pr", "refs/remotes/user-fork").stdout,
    "histories": histories,
}
raw = (json.dumps(report, separators=(",", ":")) + "\n").encode()
compressed = gzip.compress(raw, mtime=0)
(output / "complete-source-histories.json.gz").write_bytes(compressed)
summary = {
    "source_base": report["base"],
    "initial_snapshot_refs": len(initial),
    "missing_initial_commits": missing,
    "source_json_sha256": hashlib.sha256(raw).hexdigest(),
    "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
    "groups": {name: {"paths": value["paths"], "commits": len(value["commits"]), "blobs": len(value["blobs"])} for name, value in histories.items()},
}
(output / "history-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("HISTORY_SUMMARY=" + json.dumps(summary), flush=True)
encoded = base64.b64encode(compressed).decode()
print("HISTORY_GZIP_BASE64_BEGIN", flush=True)
for i in range(0, len(encoded), 4096):
    print(encoded[i:i+4096], flush=True)
print("HISTORY_GZIP_BASE64_END", flush=True)
if missing:
    raise SystemExit("Initial snapshot coverage incomplete; do not claim complete historical deduplication.")
