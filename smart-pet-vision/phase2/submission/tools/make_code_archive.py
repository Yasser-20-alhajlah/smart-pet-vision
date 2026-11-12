"""Build reviewed_source.zip: the executable candidate files of the committed revision (git HEAD).

The same members, byte for byte, are written under code/ in SmartPetVision_Phase2_source.zip by
build.sh, so the report source archive and the code archive contain identical candidate code.

Excluded from the code archive (listed in the archive's CODE_ARCHIVE_INDEX.json):
  * *.pdf: documents, not executable. The report PDF is delivered separately, and private instructor PDFs may contain student identifiers.
The tool refuses to write the archive if any member contains a private student identifier read
from private_ids.tex (when that file is present). Standard library only.

Usage: python3 tools/make_code_archive.py [OUT.zip]   (default: submission/reviewed_source.zip)
"""
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
EXCLUDED_SUFFIXES = {".pdf": "document, not executable; the report PDF is delivered separately and "
                             "instructor reports are packaged separately"}
INDEX_NAME = "CODE_ARCHIVE_INDEX.json"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*args, cwd=SUB) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout


def head_members(repo: Path) -> dict:
    """All files of the committed revision (HEAD), as {path: bytes}. Uncommitted edits are not included."""
    data = git("archive", "--format=zip", "HEAD", cwd=repo)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist() if not n.endswith("/")}


def select(members: dict):
    kept, excluded = {}, {}
    for name, data in members.items():
        reason = EXCLUDED_SUFFIXES.get(Path(name).suffix.lower())
        if reason:
            excluded[name] = reason
        else:
            kept[name] = data
    return kept, excluded


def private_tokens(path: Path) -> set:
    """Values of \\newcommand{\\id...}{VALUE} lines in private_ids.tex (never printed)."""
    if not path.is_file():
        return set()
    vals = re.findall(r"\\newcommand\{\\id\w*\}\{([^}]+)\}", path.read_text(encoding="utf-8"))
    return {v.strip() for v in vals if len(v.strip()) >= 4}


def find_private(members: dict, tokens: set) -> list:
    """Names of members that contain any token as a standalone alphanumeric run."""
    if not tokens:
        return []
    pats = [re.compile(rb"(?<![0-9A-Za-z])" + re.escape(t.encode()) + rb"(?![0-9A-Za-z])") for t in tokens]
    return sorted(n for n, b in members.items() if any(p.search(b) for p in pats))


def index(kept: dict, excluded: dict, commit: str, tree: str) -> dict:
    return {"source_commit": commit, "source_tree": tree,
            "members": {n: sha(b) for n, b in sorted(kept.items())},
            "excluded": dict(sorted(excluded.items())),
            "note": "Identical members appear under code/ in SmartPetVision_Phase2_source.zip."}


def _put(z: zipfile.ZipFile, name: str, data: bytes) -> None:
    """Fixed timestamp and permissions, so the same inputs give a byte-identical archive."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    z.writestr(info, data)


def write_zip(out: Path, kept: dict, idx: dict, prefix: str = "") -> None:
    with zipfile.ZipFile(out, "w") as z:
        for n in sorted(kept):
            _put(z, prefix + n, kept[n])
        _put(z, prefix + INDEX_NAME, (json.dumps(idx, indent=1) + "\n").encode())


def build(out: Path, private_ids: Path):
    repo = Path(git("rev-parse", "--show-toplevel").decode().strip())
    commit = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    kept, excluded = select(head_members(repo))
    hits = find_private(kept, private_tokens(private_ids))
    if hits:
        raise SystemExit("refusing to write code archive: private identifiers found in " + ", ".join(hits))
    idx = index(kept, excluded, commit, tree)
    write_zip(out, kept, idx)
    return kept, idx


def write_source_zip(out: Path, code_zip: Path, report_files: list) -> None:
    """Report source (main.tex, figures, private_ids.tex) plus every member of code_zip under code/."""
    with zipfile.ZipFile(code_zip) as c, zipfile.ZipFile(out, "w") as z:
        for f in report_files:
            _put(z, Path(f).as_posix(), Path(f).read_bytes())
        for n in sorted(c.namelist()):
            _put(z, "code/" + n, c.read(n))


def code_members_identical(source_zip: Path, code_zip: Path) -> bool:
    """True when code/ in the source ZIP holds exactly the code archive's members with equal bytes."""
    with zipfile.ZipFile(source_zip) as s, zipfile.ZipFile(code_zip) as c:
        src = {n[len("code/"):]: sha(s.read(n)) for n in s.namelist() if n.startswith("code/")}
        return src == {n: sha(c.read(n)) for n in c.namelist()}


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "--source-zip":
        out, code_zip = Path(sys.argv[2]), Path(sys.argv[3])
        write_source_zip(out, code_zip, sys.argv[4:])
        print(f"wrote {out.name}: report source + code/ from {code_zip.name} "
              f"(identical members: {code_members_identical(out, code_zip)})")
        return
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else SUB / "reviewed_source.zip"
    kept, idx = build(out, SUB / "private_ids.tex")
    print(f"wrote {out.name}: {len(kept)} files from {idx['source_commit'][:7]}, "
          f"{len(idx['excluded'])} excluded; no private identifiers found")


if __name__ == "__main__":
    main()
