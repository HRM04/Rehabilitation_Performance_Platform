#!/usr/bin/env python3
"""B02 / R02 build freeze: record exactly which software produced a session, and check it has not changed.

  python tools/build_freeze.py            write a freeze (do this once, just before formal Session B)
  python tools/build_freeze.py --check    compare the code now against the newest freeze

The code hash covers src/ and tools/ Python files, skipping backup copies. The freeze folder holds a copy of
those files, the model weights hash, the runtime settings, package versions and the git state.
"""
import argparse, glob, hashlib, json, os, platform, re, subprocess, sys, tarfile, time

SKIP_DIRS = {"backups", ".venv", "venv", "__pycache__", ".git", "node_modules", "data", "reports"}
KNOBS = ["YOLO_MODEL", "GRASP_VOTE_FRAMES", "HAND_DEDUP", "HANDEDNESS_SWAP", "LABEL_STABILISE", "HAND_SMOOTHING",
         "FAMILY_CONTACT_RULE", "ASSOC_PERMITTED_ONLY", "AMBIGUOUS_APPROPRIATENESS", "GRASP_OVERLAY"]
PKGS = ["ultralytics", "mediapipe", "opencv-python", "torch", "numpy", "streamlit", "pytest", "pandas", "plotly"]
OUT = os.path.join("data", "build_freezes")


def code_files():
    out = []
    for top in ("src", "tools"):
        for d, ds, fs in os.walk(top):
            ds[:] = sorted(x for x in ds if x not in SKIP_DIRS)
            for f in sorted(fs):
                low = f.lower()
                if f.endswith(".py") and "backup" not in low and ".pre_" not in low:
                    out.append(os.path.join(d, f))
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.replace(os.sep, "/").encode()); h.update(b"\0"); h.update(sha(f).encode()); h.update(b"\n")
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def weights():
    found = []
    for d, ds, fs in os.walk("."):
        ds[:] = [x for x in ds if x not in {".venv", "venv", ".git", "node_modules", "__pycache__", "data", "reports", "backups"}]
        for f in fs:
            if f.endswith(".pt") and "backup" not in f.lower():
                p = os.path.join(d, f)
                found.append({"file": p, "bytes": os.path.getsize(p), "sha256": sha(p)})
    return found


def code_defaults(files):
    out = {}
    for k in KNOBS:
        rx = re.compile(r"environ\.get\(\s*[\"']%s[\"']\s*(?:,\s*([^)]*))?\)" % k)
        for f in files:
            m = rx.search(open(f, errors="ignore").read())
            if m:
                out[k] = {"default_in_code": (m.group(1) or "None").strip(), "file": f}; break
    return out


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PKGS:
        try:
            from importlib.metadata import version
            out[p] = version(p)
        except Exception:
            out[p] = None
    return out


def make():
    files = code_files()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(OUT, stamp)
    os.makedirs(folder, exist_ok=True)
    cfgs = sorted((f for f in glob.glob("**/session_config.json", recursive=True) if "volunteer_testing" not in f and "backup" not in f),
                  key=os.path.getmtime)
    rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "code_hash": code_hash(files), "code_files": len(files),
           "git_head": run(["git", "rev-parse", "--short", "HEAD"]) or None,
           "git_uncommitted_files": len(run(["git", "status", "--porcelain"]).splitlines()) if run(["git", "rev-parse", "HEAD"]) else None,
           "weights": weights(), "env_now": {k: os.environ.get(k) for k in KNOBS}, "env_code_defaults": code_defaults(files),
           "versions": versions(), "newest_session_config": cfgs[-1] if cfgs else None,
           "file_hashes": {f.replace(os.sep, "/"): sha(f) for f in files}}
    if cfgs:
        rec["session_config_content"] = json.load(open(cfgs[-1]))
    with tarfile.open(os.path.join(folder, "code_snapshot.tar.gz"), "w:gz") as tar:
        for f in files:
            tar.add(f)
    json.dump(rec, open(os.path.join(folder, "build_freeze.json"), "w"), indent=1)
    lines = ["BUILD FREEZE %s" % rec["frozen_at"], "code hash   %s  (%d files, backups skipped)" % (rec["code_hash"][:16], rec["code_files"]),
             "git         HEAD %s, uncommitted files %s" % (rec["git_head"], rec["git_uncommitted_files"])]
    lines += ["weights     %s  sha256 %s" % (w["file"], w["sha256"][:16]) for w in rec["weights"]] or ["weights     none found"]
    lines += ["env set now %s" % ({k: v for k, v in rec["env_now"].items() if v is not None} or "none (code defaults apply)"),
              "versions    " + ", ".join("%s %s" % (k, v) for k, v in rec["versions"].items() if k not in ("platform",)),
              "config      %s" % rec["newest_session_config"], "", "Saved in %s:" % folder, "  build_freeze.json", "  code_snapshot.tar.gz"]
    open(os.path.join(folder, "build_freeze.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check():
    dirs = sorted(glob.glob(os.path.join(OUT, "*", "build_freeze.json")))
    if not dirs:
        sys.exit("No freeze yet. Run: python tools/build_freeze.py")
    rec = json.load(open(dirs[-1]))
    files = code_files()
    now = {f.replace(os.sep, "/"): sha(f) for f in files}
    old = rec["file_hashes"]
    changed = sorted(f for f in now if f in old and now[f] != old[f])
    added, removed = sorted(set(now) - set(old)), sorted(set(old) - set(now))
    wnow = {w["file"]: w["sha256"] for w in weights()}
    wchanged = sorted(f for f, h in ((w["file"], w["sha256"]) for w in rec["weights"]) if wnow.get(f) != h)
    env_now = {k: os.environ.get(k) for k in KNOBS}
    envchanged = sorted(k for k in KNOBS if env_now[k] != rec["env_now"].get(k))
    print("Checking against %s (frozen %s)" % (dirs[-1], rec["frozen_at"]))
    ok = not (changed or added or removed or wchanged or envchanged)
    print("code hash   %s" % ("UNCHANGED " + rec["code_hash"][:16] if code_hash(files) == rec["code_hash"] else "CHANGED"))
    for label, items in (("changed files", changed), ("new files", added), ("removed files", removed), ("changed weights", wchanged), ("changed settings", envchanged)):
        if items: print("%-16s %s" % (label, ", ".join(items[:12])))
    print("RESULT: " + ("MATCHES THE FREEZE" if ok else "DOES NOT MATCH THE FREEZE. Record why, or refreeze before Session B."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    check() if ap.parse_args().check else make()
