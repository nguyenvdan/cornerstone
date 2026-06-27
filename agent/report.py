"""Synthesize a structured, grounded report from tool outputs.

Every number here is read straight from a tool result dict — nothing is
invented. Used by the scripted orchestrator, and also as the offline renderer.
"""

from __future__ import annotations


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def synthesize(query: str, results: dict[str, dict]) -> str:
    lines: list[str] = []
    prof = results.get("lookup_prospect")
    comps = results.get("find_comparables")
    proj = results.get("project_development")
    fit = results.get("evaluate_roster_fit")

    subject = (prof or {}).get("player_name") or (proj or {}).get("prospect") or "the prospect"
    lines.append(f"# Scouting & strategy report: {subject}")
    lines.append(f"_Question: {query}_\n")

    if prof and "error" not in prof:
        bits = []
        if prof.get("college_name"):
            bits.append(f"{prof['college_name']}")
        if prof.get("age_at_draft"):
            bits.append(f"age {prof['age_at_draft']}")
        if prof.get("height_in"):
            ft, inch = divmod(int(prof["height_in"]), 12)
            bits.append(f"{ft}'{inch}\"")
        if prof.get("coll_pts_per_g") is not None:
            bits.append(f"{prof['coll_pts_per_g']} pts / {prof.get('coll_trb_per_g','?')} reb / "
                        f"{prof.get('coll_ast_per_g','?')} ast")
        lines.append("## Profile  _(from lookup_prospect)_")
        lines.append(", ".join(str(b) for b in bits) + ".\n")

    if comps and comps.get("comparables"):
        lines.append("## Historical comparables  _(from find_comparables)_")
        top = comps["comparables"][:5]
        for c in top:
            why = ", ".join(c.get("why_alike", [])[:2])
            lines.append(f"- **{c['player_name']}** ({c['draft_year']}) — similarity "
                         f"{c['similarity']}, became a {c['outcome_tier']} "
                         f"(career VORP {c['career_vorp']}); alike on {why}.")
        tiers = [c["outcome_tier"] for c in comps["comparables"]]
        lines.append(f"\nAcross the analog set the outcomes range widely "
                     f"({', '.join(sorted(set(tiers)))}), which is the basis for the "
                     f"uncertainty below.\n")

    if proj and "error" not in proj:
        lines.append("## Development projection  _(from project_development)_")
        tp = proj["tier_probabilities"]
        lines.append(f"- **P(starter or better): {_pct(proj['p_starter_plus'])}**, "
                     f"P(all-star or better): {_pct(proj['p_star_plus'])}, "
                     f"P(bust): {_pct(tp['bust'])}.")
        band = proj["career_vorp_band"]
        lines.append(f"- Expected career VORP **{proj['expected_career_vorp']}**, with an "
                     f"honest 80% range of {band['p10']} to {band['p90']} "
                     f"(median {band['p50']}).")
        if proj.get("swing_factors"):
            sw = ", ".join(f"{s['factor']} ({'↑' if s['direction']=='raises' else '↓'})"
                           for s in proj["swing_factors"][:3])
            lines.append(f"- Biggest swing factors: {sw}.")
        lines.append(f"\n> {proj['key_uncertainties'][0]}\n")

    if fit and "error" not in fit:
        lines.append("## Roster fit & build path  _(from evaluate_roster_fit)_")
        lines.append(f"- {fit['team']} fit around {fit['cornerstone']}: "
                     f"**{fit['fit_score']}/100**.")
        gaps = ", ".join(g.replace("_", " ") for g in fit["biggest_gaps"])
        lines.append(f"- Biggest unmet needs: **{gaps}**.")
        def _rec(r):
            fills = ", ".join(f.replace("_", " ") for f in r["fills"])
            return f"{r['archetype']} (fills {fills})"
        rec_str = "; ".join(_rec(r) for r in fit["recommended_archetypes"][:3])
        lines.append(f"- Recommended complementary archetypes: {rec_str}.\n")

    # Bottom line — grounded synthesis
    lines.append("## Bottom line")
    bl = []
    if proj and "error" not in proj:
        bl.append(f"{subject} profiles as a {_pct(proj['p_starter_plus'])}-likely "
                  f"starter-or-better with real star upside (P(all-star+) "
                  f"{_pct(proj['p_star_plus'])}) but genuine bust risk "
                  f"({_pct(proj['tier_probabilities']['bust'])}) — a wide distribution, "
                  f"not a sure thing.")
    if fit and "error" not in fit:
        bl.append(f"To maximize his value, {fit['team']} should prioritize "
                  f"{', '.join(g.replace('_',' ') for g in fit['biggest_gaps'][:2])} — "
                  f"a {fit['recommended_archetypes'][0]['archetype']} is the highest-leverage add.")
    lines.append(" ".join(bl) if bl else "Insufficient tool output to synthesize.")
    return "\n".join(lines)
