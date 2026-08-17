"""Extract per-culture licensing from stellarium-skycultures into machine-readable form.

Writes two files next to the project root:

  licenses.json   full parse: every license clause of every sky culture, normalised
  allowlist.json  the subset safe to derive new text from, plus the rejects and why

Run:  python -m scripts.scan_licenses
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The module rather than its names: this file already has a `LICENSES` of its own -- the
# licence *terms* -- and importing the path under that name would shadow it silently.
from skylore import paths  # noqa: E402

CULTURES_DIR = paths.CORPUS_DIR

# Terms of each licence that appears in the corpus. `derivatives` and `commercial`
# are what actually gate a text-generating tool; the rest is for attribution UX.
LICENSES = {
    "CC-BY-SA": dict(name="Creative Commons Attribution-ShareAlike", derivatives=True,
                     commercial=True, attribution=True, share_alike=True),
    "CC-BY-SA-2.0": dict(name="Creative Commons Attribution-ShareAlike 2.0", derivatives=True,
                         commercial=True, attribution=True, share_alike=True),
    "CC-BY-ND-4.0": dict(name="Creative Commons Attribution-NoDerivatives 4.0", derivatives=False,
                         commercial=True, attribution=True, share_alike=False),
    "CC-BY-NC": dict(name="Creative Commons Attribution-NonCommercial", derivatives=True,
                     commercial=False, attribution=True, share_alike=True),
    "CC-BY-NC-ND-4.0": dict(name="Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
                            derivatives=False, commercial=False, attribution=True, share_alike=False),
    "GPL-2.0": dict(name="GNU General Public License v2.0", derivatives=True,
                    commercial=True, attribution=True, share_alike=True),
    "LAL": dict(name="Free Art License", derivatives=True,
                commercial=True, attribution=True, share_alike=True),
    "public-domain": dict(name="Public Domain", derivatives=True,
                          commercial=True, attribution=False, share_alike=False),
}

# Ordered: first match wins, so the longer variants must precede their prefixes.
LICENSE_PATTERNS = [
    (re.compile(r"CC[- ]BY[- ]NC[- ]ND[- ]?4\.0", re.I), "CC-BY-NC-ND-4.0"),
    (re.compile(r"CC[- ]BY[- ]ND[- ]?4\.0", re.I), "CC-BY-ND-4.0"),
    (re.compile(r"CC[- ]BY[- ]?SA[- ]?2\.0", re.I), "CC-BY-SA-2.0"),
    (re.compile(r"CC[- ]BY[- ]?SA", re.I), "CC-BY-SA"),
    (re.compile(r"CC[- ]BY[- ]?NC", re.I), "CC-BY-NC"),
    (re.compile(r"GNU GPL v?2\.0|GPL[- ]?v?2", re.I), "GPL-2.0"),
    (re.compile(r"Free Art License", re.I), "LAL"),
    (re.compile(r"Public[- ]domain", re.I), "public-domain"),
]

# A clause's scope decides what it constrains. Most cultures license their prose and
# their artwork separately -- typically CC-BY-SA text alongside Free Art License
# illustrations -- so the two have to be resolved independently.
TEXT_SCOPE_WORDS = ("text", "data", "line", "lines", "all")
IMAGE_SCOPE_WORDS = ("illustration", "illustrations", "image", "images", "map", "maps",
                     "art", "artwork", "all")

# Grants Stellarium negotiated for itself. They do not extend to downstream users,
# so a culture carrying one must be judged on its stated licence alone.
SPECIAL_PERMISSION = re.compile(r"special permission", re.I)

# Clauses no pattern can decide are escalated to a human. Recording the verdict here
# keeps the output reproducible: rerunning the scan must not silently undo a review.
MANUAL_REVIEWS = {
    "japanese_moon_stations": {
        "verdict": "allow",
        "reviewed": "2026-08-08",
        "clause": "Yasui's Map detail[4] used with authorisation from Steven Renshaw",
        "rationale": (
            "The permission covers one scanned map, not the sky culture as a whole; "
            "text and data are stated separately as CC BY-SA. Allowed for text and data "
            "only, on condition that the map is neither ingested nor served."
        ),
        "excluded_assets": ["chart.webp"],
    },
}


def read_section(markdown: str, heading: str) -> list[str]:
    """Return the non-empty lines under `## <heading>`, up to the next `## ` heading."""
    lines: list[str] = []
    inside = False
    for line in markdown.splitlines():
        if re.match(rf"^##\s+{heading}\s*$", line, re.I):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside and line.strip():
            lines.append(line.strip())
    return lines


def parse_clause(raw: str) -> dict:
    """Split one licence line into scope + licence id, or mark it as a note."""
    scope, _, rest = raw.partition(":")
    if rest.strip():
        scope, text = scope.strip(), rest.strip()
    else:
        scope, text = "all", raw

    for pattern, license_id in LICENSE_PATTERNS:
        if pattern.search(text):
            # Footnote markers such as "[#5]" leak in from the markdown body.
            return {"raw": raw, "scope": scope, "license": license_id}

    kind = "special_permission" if SPECIAL_PERMISSION.search(raw) else "unrecognised"
    return {"raw": raw, "scope": scope, "license": None, "note": kind}


def covers(scope: str, scope_words: tuple[str, ...]) -> bool:
    lowered = scope.lower()
    return any(word in lowered for word in scope_words)


def combine(clauses: list[dict], scope_words: tuple[str, ...] | None = None) -> dict:
    """Most restrictive terms across the given clauses — one veto is enough.

    `scope_words` narrows to the clauses governing one kind of material; None keeps
    every clause, which answers "what covers this culture taken as a whole".
    """
    considered = [
        c for c in clauses
        if c["license"] and (scope_words is None or covers(c["scope"], scope_words))
    ]
    if not considered:
        return {"licenses": [], "derivatives": None, "commercial": None,
                "share_alike": None, "attribution": None}

    terms = [LICENSES[c["license"]] for c in considered]
    return {
        "licenses": sorted({c["license"] for c in considered}),
        "derivatives": all(t["derivatives"] for t in terms),
        "commercial": all(t["commercial"] for t in terms),
        "share_alike": any(t["share_alike"] for t in terms),
        "attribution": any(t["attribution"] for t in terms),
    }


def find_images(directory: Path, excluded: list[str]) -> dict:
    """Inventory the artwork on disk, minus anything a manual review carved out."""
    files = sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"} and "/po/" not in p.as_posix()
    )
    kept = [f for f in files if f not in excluded and Path(f).name not in excluded]
    return {"count": len(kept), "files": kept,
            "excluded": [f for f in files if f not in kept]}


def scan_culture(directory: Path) -> dict:
    markdown = (directory / "description.md").read_text(encoding="utf-8")
    clauses = [parse_clause(line) for line in read_section(markdown, "License")]

    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    review = MANUAL_REVIEWS.get(directory.name)
    images = find_images(directory, review["excluded_assets"] if review else [])

    culture = {
        "id": directory.name,
        "region": index.get("region"),
        "constellations": len(index.get("constellations", [])),
        "authors": read_section(markdown, "Authors"),
        "clauses": clauses,
        "effective": combine(clauses),
        "text_effective": combine(clauses, TEXT_SCOPE_WORDS),
        "image_effective": combine(clauses, IMAGE_SCOPE_WORDS),
        "images": images,
        "has_special_permission_to_stellarium": any(
            c.get("note") == "special_permission" for c in clauses
        ),
        "unrecognised_clauses": [c["raw"] for c in clauses if c.get("note") == "unrecognised"],
    }
    if review:
        culture["manual_review"] = review
    return culture


def image_entry(culture: dict) -> dict:
    """Artwork terms for a culture, resolved independently of its text terms."""
    image, inventory = culture["image_effective"], culture["images"]
    review = culture.get("manual_review", {})
    return {
        "count": inventory["count"],
        "licenses": image["licenses"],
        "commercial": image["commercial"],
        "share_alike": image["share_alike"],
        # False where a no-derivatives or unparsed clause governs the artwork even
        # though the prose is free; such a culture is text-only for us.
        "usable": bool(inventory["count"]) and image["derivatives"] is True,
        "excluded": inventory["excluded"],
        **({"excluded_because": review["rationale"]} if inventory["excluded"] else {}),
    }


def build_allowlist(cultures: list[dict]) -> dict:
    allowed, rejected = [], []
    for culture in cultures:
        text = culture["text_effective"]
        review = culture.get("manual_review", {})
        reasons = []
        if text["derivatives"] is False:
            reasons.append("no-derivatives clause on the text/data")
        if text["derivatives"] is None:
            reasons.append("no machine-readable licence found for text/data")
        # A review can only resolve the clauses it was raised for; it never overrides a
        # no-derivatives verdict, which no amount of reading makes go away.
        if review.get("verdict") != "allow":
            for clause in culture["unrecognised_clauses"]:
                reasons.append(f"clause needs manual review: {clause!r}")

        entry = {
            "id": culture["id"],
            "licenses": text["licenses"],
            "commercial": text["commercial"],
            "share_alike": text["share_alike"],
            "images": image_entry(culture),
        }
        if review and not reasons:
            entry["manual_review"] = {k: review[k] for k in ("reviewed", "clause", "rationale")}
        if reasons:
            rejected.append({**entry, "reasons": reasons})
        else:
            allowed.append(entry)

    return {
        "policy": (
            "Cultures whose text/data licence permits derivative works. "
            "share_alike=true means output derived from that culture must carry the same "
            "licence; commercial=false means non-commercial use only. Special permissions "
            "granted to Stellarium in the source do not extend to this project. "
            "Artwork is licensed separately from prose in most cultures, so `images` "
            "carries its own terms: serve an illustration only where images.usable is "
            "true, and never serve a file listed in images.excluded."
        ),
        # Derived from the real location rather than written out, so a moved corpus
        # cannot leave a false provenance trail in a generated licence record.
        "generated_from": f"{CULTURES_DIR.relative_to(paths.ROOT)}/*/description.md",
        "allowed": allowed,
        "allowed_commercial": [e["id"] for e in allowed if e["commercial"]],
        "allowed_images": [e["id"] for e in allowed if e["images"]["usable"]],
        "rejected": rejected,
    }


def main() -> int:
    if not CULTURES_DIR.is_dir():
        print(f"missing corpus: {CULTURES_DIR}", file=sys.stderr)
        return 1

    cultures = sorted(
        (scan_culture(d) for d in CULTURES_DIR.iterdir()
         if d.is_dir() and (d / "description.md").exists()),
        key=lambda c: c["id"],
    )

    licenses_path = paths.LICENSES
    allowlist_path = paths.ALLOWLIST
    corpus_path = CULTURES_DIR.relative_to(paths.ROOT)
    payload = {
        "source": paths.CORPUS_NAME,
        "repository_license": f"AGPL-3.0 (see {corpus_path}/LICENSE-AGPL-3.0.txt)",
        "note": (
            "The repository-level AGPL does not override the per-culture licence declared "
            "in each description.md; the per-culture terms below govern the content."
        ),
        "license_terms": LICENSES,
        "cultures": cultures,
    }
    licenses_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    allowlist = build_allowlist(cultures)
    allowlist_path.write_text(json.dumps(allowlist, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    images = sum(e["images"]["count"] for e in allowlist["allowed"] if e["images"]["usable"])
    print(f"{len(cultures)} cultures -> {licenses_path.name}, {allowlist_path.name}")
    print(f"  derivable: {len(allowlist['allowed'])} "
          f"({len(allowlist['allowed_commercial'])} of them commercially)")
    print(f"  artwork:   {images} illustrations across "
          f"{len(allowlist['allowed_images'])} cultures")
    print(f"  excluded:  {len(allowlist['rejected'])}")
    for entry in allowlist["rejected"]:
        print(f"    {entry['id']}: {'; '.join(entry['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
