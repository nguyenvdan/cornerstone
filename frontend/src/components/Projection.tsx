import { pct, TIER_LABELS, TIER_ORDER } from "../data";
import { Bar, SeasonCurve } from "./Bits";

const TIER_COLOR: Record<string, string> = {
  superstar: "#7a2bb8", all_star: "#e8743b", starter: "#2e8b57",
  rotation: "#1f4e79", bust: "#9c3a3a",
};

export default function Projection({ p }: { p: any }) {
  const tp = p.tier_probabilities;
  const band = p.career_vorp_band;
  return (
    <section className="view">
      <h2 className="h2">Development projection</h2>
      <p className="sub">
        A probability distribution over outcomes — derived from how {p.n_comparables} historical
        comparables actually developed. No point estimates, no false precision.
      </p>

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

      <div className="callout" style={{ marginTop: 18 }}>
        <b>Honest read:</b> {p.key_uncertainties?.[2] ?? p.key_uncertainties?.[0]}
      </div>
    </section>
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
