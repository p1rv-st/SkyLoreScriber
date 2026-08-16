"""`compare_across_cultures`: who else drew a figure through these same stars.

The one question the corpus exists to answer and no other tool reaches. `lookup_star`
works from a star outwards -- what is known about Aldebaran. This works from a *set*:
give it a figure, or a list of HIPs, and it finds every other figure overlapping that
patch of sky and reports what each tradition saw there. Western Orion comes back as the
Tupi Old Man, the Navajo First Slim One, the Egyptian Sah, the Belarusian Throne of
Jesus.

Needs no model. `constellation_lines` (11637 rows) and the `ix_lines_hip` index exist
for exactly this join.

Two properties of the data shape it:

**Naming and drawing are different relations,** as everywhere else in this codebase. A
culture can run a line through a star it never names. Overlap is computed on the lines,
because that is what "seeing a figure here" means; names come along for what to call it.

**Absolute overlap flatters large figures, measured.** Ranking Orion's overlaps by raw
count put the Egyptian *Sah* -- 8 shared stars out of a 26-star figure reaching well
outside Orion -- above the Belarusian *Throne of Jesus*, Chinese *Three Stars*, Hawaiian
*Cat's Cradle* and Korean *Saam*, every one of which lies **entirely** inside Orion.
Those are complete matches from their own side and were ranked seventh and below.

Ranking by the other fraction alone fails the opposite way: it promotes any small figure
above `western_SnT`'s near-identical Orion. So the score is the harmonic mean of the
two, which is what "sees the same figure" actually means -- how much of the asked-about
sky the figure covers, *and* how much of the figure that sky explains. Both fractions
and the raw count stay in the output; only the ordering uses the mean.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from . import lang, names

# Below this many shared stars the result is noise: Orion crosses dozens of figures at
# a single star. Deliberately a flat number for now rather than a function of the target
# size -- worth revisiting once real answers have been read.
MIN_SHARED = 2


@dataclass(frozen=True)
class Overlap:
    """One other culture's figure drawn through some of the same stars."""
    constellation_id: str
    culture_id: str
    names: names.NameSet
    shared_hips: list[int]
    figure_size: int
    target_size: int
    attribution: str

    @property
    def shared(self) -> int:
        return len(self.shared_hips)

    @property
    def of_target(self) -> float:
        """How much of the asked-about sky this figure covers."""
        return self.shared / self.target_size if self.target_size else 0.0

    @property
    def of_figure(self) -> float:
        """How much of this figure the asked-about sky explains."""
        return self.shared / self.figure_size if self.figure_size else 0.0

    @property
    def score(self) -> float:
        """Harmonic mean of the two. Neither alone orders these sensibly -- see the
        module docstring for what each gets wrong."""
        a, b = self.of_target, self.of_figure
        return 2 * a * b / (a + b) if a + b else 0.0


@dataclass(frozen=True)
class Comparison:
    target_hips: list[int]
    target: names.Constellation | None       # None when called with a bare HIP list
    overlaps: list[Overlap] = field(default_factory=list)


def figure_hips(connection: sqlite3.Connection, constellation_id: str) -> list[int]:
    return [row[0] for row in connection.execute(
        "SELECT DISTINCT hip FROM constellation_lines WHERE constellation_id = ?"
        " ORDER BY hip", (constellation_id,))]


def compare_across_cultures(
    connection: sqlite3.Connection,
    *,
    constellation_id: str | None = None,
    hips: list[int] | None = None,
    locale: str = lang.SOURCE_LANG,
    min_shared: int = MIN_SHARED,
    limit: int = 20,
) -> Comparison:
    """Figures overlapping the given stars, most shared stars first.

    Takes either a constellation to compare outwards from, or a bare list of HIPs for
    "who sees a figure in the Pleiades". The source figure is excluded from its own
    results; other figures of the same culture are not, because a culture genuinely can
    draw two overlapping ones.
    """
    if constellation_id and hips is None:
        hips = figure_hips(connection, constellation_id)
    hips = sorted(set(hips or []))
    if not hips:
        return Comparison(target_hips=[], target=None)

    placeholders = ",".join("?" * len(hips))
    rows = connection.execute(f"""
        SELECT l.constellation_id, c.culture_id, group_concat(DISTINCT l.hip),
               (SELECT count(DISTINCT hip) FROM constellation_lines w
                 WHERE w.constellation_id = l.constellation_id)
          FROM constellation_lines l
          JOIN constellations c ON c.id = l.constellation_id
         WHERE l.hip IN ({placeholders})
           AND (? IS NULL OR l.constellation_id <> ?)
         GROUP BY l.constellation_id
        HAVING count(DISTINCT l.hip) >= ?
    """, [*hips, constellation_id, constellation_id, min_shared]).fetchall()

    available = lang.available_langs(connection)
    overlaps: list[Overlap] = []
    for other_id, culture_id, shared_csv, figure_size in rows:
        shared = sorted(int(value) for value in shared_csv.split(","))
        nameset = names._grouped_names(
            connection, "constellation_id = :id", {"id": other_id}, locale)
        attribution, = connection.execute(
            "SELECT attribution FROM cultures WHERE id = ?", (culture_id,)).fetchone()
        overlaps.append(Overlap(
            constellation_id=other_id,
            culture_id=culture_id,
            names=nameset[0] if nameset else names.NameSet(
                culture_id, None, None, {}, None),
            shared_hips=shared,
            figure_size=figure_size,
            target_size=len(hips),
            attribution=attribution,
        ))

    overlaps.sort(key=lambda o: (-o.score, -o.shared, o.culture_id))
    del available  # resolution happens inside _grouped_names

    target = (names._constellation(connection, constellation_id, locale, with_prose=True)
              if constellation_id else None)
    return Comparison(target_hips=hips, target=target, overlaps=overlaps[:limit])
