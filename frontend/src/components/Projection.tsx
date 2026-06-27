import { inchesToFeet, pct, TIER_LABELS, TIER_ORDER, vorpLabel } from "../data";
import { Bar, Radar, SeasonCurve } from "./Bits";

// Map the 6 internal skills to scout-friendly radar categories (clockwise).
const RADAR: { key: string; label: string }[] = [
  { key: "shot_creation", label: "Scoring" },
  { key: "playmaking", label: "Playmaking" },
  { key: "spacing", label: "Shooting" },
  { key: "perimeter_defense", label: "Perimeter D" },
  { key: "rim_protection", label: "Rim Protect" },
  { key: "rebounding", label: "Rebounding" },
];

// AJ's measured combine metrics, as height/position-adjusted percentiles.
const COMBINE_RADAR: { key: string; label: string }[] = [
  { key: "vertical_pct", label: "Explosive" },
  { key: "wingspan_pct", label: "Length" },
  { key: "reach_pct", label: "Reach" },
  { key: "height_pct", label: "Pos. Size" },
];

function RadarLegend() {
  return (
    <div style={{ display: "flex", gap: 18, justifyContent: "center", marginTop: 2,
      fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-display)",
      textTransform: "uppercase", letterSpacing: "0.04em" }}>
      <span><span style={{ display: "inline-block", width: 14, height: 2,
        background: "var(--ink)", verticalAlign: "middle", marginRight: 5 }} />AJ</span>
      <span><span style={{ display: "inline-block", width: 14, height: 0,
        borderTop: "2px dashed var(--avg)", verticalAlign: "middle", marginRight: 5 }} />Average</span>
    </div>
  );
}

// Color only carries meaning: green = star upside, red = bust, gray = the middle.
const TIER_COLOR: Record<string, string> = {
  superstar: "var(--pos)", all_star: "var(--pos-soft)", starter: "var(--ink-2)",
  rotation: "var(--faint)", bust: "var(--neg)",
};

export default function Projection({ p }: { p: any }) {
  const tp = p.tier_probabilities;
  const band = p.career_vorp_band;
  return (
    <section className="view">
      <h2 className="h2">
        Development projection{" "}
        {p.model_mode === "scouting-informed" && (
          <span className="pill all_star" style={{ fontSize: 12, verticalAlign: "middle" }}>
            scouting-informed
          </span>
        )}
      </h2>
      <p className="sub">
        A probability distribution over outcomes — derived from how {p.n_comparables} historical
        comparables actually developed, with real pre-draft signals (draft capital, playstyle
        archetype, competition, age) layered in. No point estimates, no false precision.
      </p>

      {/* Best / expected / worst career outcomes, mapped to the VORP scale */}
      <div className="grid three" style={{ marginBottom: 10 }}>
        <Scenario label="Worst case (10th pct)" vorp={band.p10} accent="var(--accent-2)" />
        <Scenario label="Expected" vorp={p.expected_career_vorp} accent="var(--navy-2)" />
        <Scenario label="Ceiling (99th pct)" vorp={band.p99} accent="var(--ok)" />
      </div>
      {p.ceiling_comparable?.player_name && (
        <p className="sub" style={{ marginBottom: 18 }}>
          <b>Reading the range.</b> A <b>strong</b> outcome (75th pct) is multiple-All-Star
          territory ({band.p75} VORP). The <b>ceiling</b> (99th pct, {band.p99} VORP) lands right
          at his single best comparable — a <b>{p.ceiling_comparable.player_name}</b>–level,
          all-time-great career ({p.ceiling_comparable.career_vorp} VORP). Durant and Harden are
          genuinely among his analogs; that ceiling is a low-probability (~1-in-100) dream, but a
          real, data-grounded one.
        </p>
      )}

      {p.projected_peak_per36?.pts && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3>Projected peak production <span style={{ color: "var(--muted)", fontWeight: 400,
            textTransform: "none", letterSpacing: 0 }}>— per 36 minutes, at his peak</span></h3>
          <div style={{ display: "flex", gap: 0, flexWrap: "wrap", borderTop: "1px solid var(--line)" }}>
            {[["PTS", p.projected_peak_per36.pts], ["REB", p.projected_peak_per36.reb],
              ["AST", p.projected_peak_per36.ast], ["STL", p.projected_peak_per36.stl],
              ["BLK", p.projected_peak_per36.blk]].map(([lbl, v]) => (
              <div key={lbl as string} style={{ flex: 1, minWidth: 90, padding: "14px 18px 14px 0" }}>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 500, fontSize: 30,
                  letterSpacing: "-0.02em" }}>{(v as number).toFixed(1)}</div>
                <div style={{ fontFamily: "var(--font-display)", fontSize: 11, color: "var(--muted)",
                  textTransform: "uppercase", letterSpacing: "0.1em" }}>{lbl}</div>
              </div>
            ))}
          </div>
          <div className="note">
            Shooting {(p.projected_peak_per36.fg_pct * 100).toFixed(0)}% FG /{" "}
            {(p.projected_peak_per36.fg3_pct * 100).toFixed(0)}% 3P /{" "}
            {(p.projected_peak_per36.ts_pct * 100).toFixed(0)}% TS. Similarity-weighted from the
            peak seasons of his {p.projected_peak_per36.n} comparables — a fringe-All-Star scoring
            wing line.
          </div>
        </div>
      )}

      <div className="grid two">
        <div className="card">
          <h3>Outcome tier probability</h3>
          {TIER_ORDER.map((t) => (
            <Bar key={t} label={TIER_LABELS[t]} value={Math.round(tp[t] * 100)}
              suffix="%" color={TIER_COLOR[t]} />
          ))}
          <div className="badge-row">
            <span className="badge">P(starter+) {pct(p.p_starter_plus)}</span>
            <span className="badge">P(all-star+) {pct(p.p_star_plus)}</span>
            <span className="badge">P(bust) {pct(tp.bust)}</span>
          </div>
        </div>

        <div className="card">
          <h3>Career value — expected & 80% range</h3>
          <div className="kpi">{p.expected_career_vorp} <small>expected career VORP</small></div>
          <p className="sub" style={{ marginTop: 8 }}>
            80% interval <b>{band.p10}</b> to <b>{band.p90}</b> &nbsp;(median {band.p50})
          </p>
          {/* horizontal interval bar */}
          <IntervalBar lo={band.p10} q1={band.p25} med={band.p50} q3={band.p75} hi={band.p90} />
          <div className="note">
            The spread is the uncertainty: it comes directly from how differently the
            comparables' careers turned out.
          </div>
        </div>

        <div className="card">
          <h3>Projected VORP by season (Year 1–5)</h3>
          <SeasonCurve curve={p.season_curve} />
          <div className="note">Shaded band = 25th–75th percentile of comparables; line = median.</div>
        </div>

        <div className="card">
          <h3>Swing factors</h3>
          <p className="sub" style={{ marginTop: 0 }}>What most moves the projection (per +1 SD).</p>
          {p.swing_factors.slice(0, 6).map((s: any) => (
            <div className="bar-row" key={s.display}>
              <div className="lbl">{s.display}</div>
              <div className="bar-track">
                <div className="bar-fill"
                  style={{
                    width: `${Math.min(100, Math.abs(s.effect_vorp) * 28)}%`,
                    background: s.direction === "raises" ? "var(--ok)" : "var(--accent-2)",
                  }} />
              </div>
              <div className="val">{s.direction === "raises" ? "↑" : "↓"} {s.effect_vorp}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid two" style={{ marginTop: 18 }}>
        {p.skill_profile && (
          <div className="card">
            <h3>Skill profile <span style={{ color: "var(--muted)", fontWeight: 400,
              textTransform: "none", letterSpacing: 0 }}>— percentile vs prospects</span></h3>
            <Radar label="Skill profile radar"
              data={RADAR.map((r) => ({ label: r.label, value: p.skill_profile[r.key] }))} />
            <RadarLegend />
            <div className="note">
              Elite shot creation & playmaking, solid rebounding. "Rim Protect" is a center
              skill he isn't asked to provide — not a flaw for a scoring wing. He needs
              spacing & rim help around him.
            </div>
          </div>
        )}

        {p.combine && (
          <div className="card">
            <h3>Combine athleticism <span className="pill superstar" style={{ fontSize: 11 }}>
              {Math.round(p.combine.athleticism_pct)}th pct athlete</span></h3>
            <div className="grid three" style={{ gap: 10, marginBottom: 2 }}>
              <Measurable label="Height (no shoes)" value={inchesToFeet(p.combine.height_no_shoes_in)} />
              <Measurable label="Wingspan" value={inchesToFeet(p.combine.wingspan_in)} />
              <Measurable label="Max vertical" value={`${p.combine.max_vertical_in}"`}
                sub="combine-best" highlight />
            </div>
            <Radar label="Combine athleticism radar"
              data={COMBINE_RADAR.map((r) => ({ label: r.label, value: p.combine[r.key] }))} />
            <RadarLegend />
            <div className="note">
              Height-adjusted: explosiveness & length vs same-height players, size vs position.
              A 42" vertical on a 6'8" frame is 99th-pct explosive; his length is average for
              his height.
            </div>
          </div>
        )}
      </div>

      <div className="callout" style={{ marginTop: 18 }}>
        <b>Honest read:</b> {p.key_uncertainties?.[2] ?? p.key_uncertainties?.[0]}
      </div>

      {p.adjustments && (
        <div className="card" style={{ marginTop: 18 }}>
          <h3>How this projection is built</h3>
          {p.profile_only && (
            <p className="sub" style={{ marginTop: 0 }}>
              Profile-only baseline (stats alone, no draft/scouting signal): P(starter+){" "}
              <b>{pct(p.profile_only.p_starter_plus)}</b>, P(all-star+){" "}
              <b>{pct(p.profile_only.p_star_plus)}</b>, P(bust){" "}
              <b>{pct(p.profile_only.p_bust)}</b>. The signals below move it to the headline
              numbers — adding real information, not nudging the answer.
            </p>
          )}
          <ul>
            {p.adjustments.map((a: string, i: number) => <li key={i}>{a}</li>)}
          </ul>
        </div>
      )}
    </section>
  );
}

function Scenario({ label, vorp, accent }: { label: string; vorp: number; accent: string }) {
  return (
    <div className="card" style={{ borderTop: `3px solid ${accent}`, padding: "16px 18px" }}>
      <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.06em",
        color: "var(--muted-solid)", fontWeight: 700 }}>{label}</div>
      <div className="kpi" style={{ margin: "6px 0 2px" }}>{vorp}<small> career VORP</small></div>
      <div style={{ fontSize: 14, fontWeight: 600, color: accent }}>{vorpLabel(vorp)}</div>
    </div>
  );
}

function Measurable({ label, value, sub, highlight }:
  { label: string; value: string; sub?: string; highlight?: boolean }) {
  return (
    <div style={{ padding: "6px 0" }}>
      <div style={{ fontSize: 12, color: "var(--muted-solid)", fontWeight: 600 }}>{label}</div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: 24, fontWeight: 500,
        letterSpacing: "-0.02em", color: highlight ? "var(--pos)" : "var(--ink)" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 12, color: "var(--muted-solid)" }}>{sub}</div>}
    </div>
  );
}

function IntervalBar({ lo, q1, med, q3, hi }: any) {
  const min = Math.min(0, lo), max = hi;
  const span = max - min || 1;
  const X = (v: number) => ((v - min) / span) * 100;
  return (
    <svg viewBox="0 0 100 22" width="100%" height="44" preserveAspectRatio="none"
      style={{ marginTop: 6 }}>
      <line x1={X(lo)} x2={X(hi)} y1="11" y2="11" stroke="#c6d0db" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
      <rect x={X(q1)} width={X(q3) - X(q1)} y="5" height="12" rx="2" fill="var(--navy-2)" opacity="0.25" />
      <line x1={X(med)} x2={X(med)} y1="3" y2="19" stroke="var(--navy)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      {X(0) >= 0 && X(0) <= 100 && (
        <line x1={X(0)} x2={X(0)} y1="6" y2="16" stroke="#9c3a3a" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />
      )}
    </svg>
  );
}
