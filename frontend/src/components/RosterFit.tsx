import { titleCase } from "../data";
import { Bar } from "./Bits";

export default function RosterFit({ fit }: { fit: any }) {
  const r = fit.report;
  const roster = fit.roster ?? [];
  return (
    <section className="view">
      <h2 className="h2">Wizards build path</h2>
      <p className="sub">
        How well Washington's current roster complements a Dybantsa cornerstone — and the
        highest-leverage gaps to fill. Skills are real 2025-26 NBA league percentiles.
      </p>

      <div className="grid two">
        <div className="card">
          <h3>Roster fit around {fit.cornerstone}</h3>
          <div className="kpi">{r.fit_score}<small>/100</small></div>
          {fit.calibration && (
            <p className="sub" style={{ marginTop: 8, marginBottom: 0 }}>
              Ranks <b>{fit.calibration.rank} of {fit.calibration.n_teams}</b> teams league-wide
              {" "}({fit.calibration.percentile}th pctile · median {fit.calibration.league_median}
              {" "}· best {fit.calibration.league_best.team} {fit.calibration.league_best.fit_score}).
              Trajectory-adjusted for the roster's youth.
            </p>
          )}
          <p className="sub" style={{ marginTop: 12 }}>Need vs. roster supply, by skill:</p>
          {r.skills.map((s: any) => (
            <div key={s.skill} style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>
                {titleCase(s.skill)}
                {s.gap >= 20 && <span className="pill bust" style={{ marginLeft: 8 }}>gap {s.gap}</span>}
              </div>
              <Bar label="need" value={s.need} color="var(--faint)" />
              <Bar label="roster" value={s.roster_supply} color="var(--ink)" />
            </div>
          ))}
        </div>

        <div className="card">
          <h3>Recommended complementary archetypes</h3>
          <p className="sub" style={{ marginTop: 0 }}>
            Biggest unmet needs: <b>{r.biggest_gaps.map(titleCase).join(", ")}</b>.
          </p>
          {r.recommended_archetypes.map((a: any, i: number) => (
            <div className="comp" key={a.archetype}>
              <div className="top">
                <span className="name">{i + 1}. {a.archetype}</span>
                <span className="sim">{a.score}</span>
              </div>
              <div className="why">Fills {a.fills.map(titleCase).join(", ")}.</div>
            </div>
          ))}
          <div className="note">
            Cornerstone supply (role tendency): {Object.entries(r.cornerstone_supply)
              .sort((a: any, b: any) => b[1] - a[1])
              .slice(0, 3)
              .map(([k, v]: any) => `${titleCase(k)} ${Math.round(v)}`)
              .join(", ")}.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <h3>Current rotation</h3>
        <table className="tbl">
          <thead>
            <tr>
              <th>Player</th><th className="num">MPG</th><th>Primary skill</th>
              <th className="num">Spacing</th><th className="num">Rim prot.</th>
              <th className="num">Playmaking</th><th className="num">Perim. D</th>
            </tr>
          </thead>
          <tbody>
            {roster.slice(0, 12).map((p: any) => (
              <tr key={p.player_name}>
                <td>{p.player_name}</td>
                <td className="num">{p.mp_per_g.toFixed?.(1) ?? p.mp_per_g}</td>
                <td>{titleCase(p.primary_skill)}</td>
                <td className="num">{Math.round(p.spacing)}</td>
                <td className="num">{Math.round(p.rim_protection)}</td>
                <td className="num">{Math.round(p.playmaking)}</td>
                <td className="num">{Math.round(p.perimeter_defense)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
