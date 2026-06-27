export default function Methodology({ backtest }: { backtest: any }) {
  const s = backtest.summary;
  const rows: [string, any, any, any][] = [
    ["Within-1-tier accuracy", s.model.tier.within_one, s.baseline.tier.within_one, "—"],
    ["P(all-star+) AUC", s.model.star.auc, s.baseline.star.auc, s.combined.star.auc],
    ["P(all-star+) calibration (ECE ↓)", s.model.star.ece, s.baseline.star.ece, s.combined.star.ece],
    ["Career-VORP ranking (Spearman ↑)", s.model.vorp.spearman, s.baseline.vorp.spearman, s.combined.vorp.spearman],
  ];
  return (
    <section className="view">
      <h2 className="h2">Methodology & back-test</h2>
      <p className="sub">
        Leakage-aware expanding-window validation: each draft class is projected using only
        players drafted before it, then compared to what actually happened — and scored
        against a draft-position baseline.
      </p>

      <div className="grid two">
        <div className="card">
          <h3>Headline metrics — {s.n_prospects} prospects, {s.n_years} draft classes</h3>
          <table className="tbl">
            <thead>
              <tr><th>Metric</th><th className="num">Model</th><th className="num">Baseline</th><th className="num">Combined</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r[0]}>
                  <td>{r[0]}</td>
                  <td className="num">{r[1]}</td>
                  <td className="num">{r[2]}</td>
                  <td className="num">{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="note">
            <b>Honest read:</b> draft position is a strong baseline that profile stats alone
            don't beat — but the model is better <em>calibrated</em> (ECE {s.model.star.ece} vs{" "}
            {s.baseline.star.ece}), and <b>profile + draft slot beats draft slot alone</b>
            {" "}(AUC {s.baseline.star.auc} → {s.combined.star.auc}). The model adds independent signal.
          </div>
        </div>

        <div className="card">
          <h3>Calibration & accuracy</h3>
          <img src="/calibration.png" alt="Calibration reliability diagram and accuracy vs baseline"
            style={{ width: "100%", borderRadius: 8, border: "1px solid var(--line)" }} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <h3>Limitations (stated honestly)</h3>
        <ul>
          <li>College box-score profiles are genuinely weak NBA predictors — hence the wide,
            explicit uncertainty. No false precision.</li>
          <li>Small early-cohort samples; survivorship and era effects.</li>
          <li>Training uses earlier cohorts' eventually-realized careers (fair to both model and
            baseline). Roster-fit captures skills, not contracts, health, or scheme.</li>
        </ul>
      </div>
    </section>
  );
}
