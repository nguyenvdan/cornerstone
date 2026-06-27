export default function Comparables({ c }: { c: any }) {
  const comps = c.comparables ?? [];
  return (
    <section className="view">
      <h2 className="h2">Historical comparables</h2>
      <p className="sub">
        The most similar pre-draft profiles from {c.n_universe} historical prospects, with the
        features that drove each match. Similarity is explainable, not a black box.
      </p>
      <div className="card">
        {comps.map((m: any, i: number) => (
          <div className="comp" key={i}>
            <div className="top">
              <div>
                <span className="name">{m.player_name}</span>{" "}
                <span style={{ color: "var(--muted-solid)", fontSize: 13 }}>
                  ({m.draft_year}{m.draft_pick ? `, #${m.draft_pick}` : ""})
                </span>{" "}
                <span className={`pill ${m.outcome_tier}`}>{m.outcome_tier.replace("_", "-")}</span>
              </div>
              <div className="sim">{m.similarity}</div>
            </div>
            <div className="why">
              Alike on {m.most_alike.map((d: any) => `${d.display} (${d.comp_value})`).join(", ")}.
              {m.biggest_gaps?.length ? (
                <> Differs on {m.biggest_gaps.map((d: any) =>
                  `${d.display} ${d.direction}`).join(", ")}.</>
              ) : null}
              {typeof m.career_vorp === "number" ? <> Career VORP {m.career_vorp}.</> : null}
            </div>
          </div>
        ))}
      </div>
      <p className="sub" style={{ marginTop: 14 }}>
        Sanity check: Anthony Davis's analogs are all shot-blocking bigs; Dybantsa's are
        high-usage scoring wings — the similarity space clusters the way scouts would.
      </p>
    </section>
  );
}
