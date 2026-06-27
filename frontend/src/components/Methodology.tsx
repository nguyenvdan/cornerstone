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
        players drafted before it, then compared to what actually happened — and scored against
        a draft-position baseline. The scouting model's settings were <b>tuned on 2010–2015
        prospects and validated on held-out 2016–2019 classes</b>, so they're proven to predict,
        not eyeballed.
      </p>

      <div className="grid two">
        <div className="card">
          <h3>Scouting model vs baseline — {s.n_prospects} prospects, {s.n_years} classes</h3>
          <table className="tbl">
            <thead>
              <tr><th>Metric</th><th className="num">Scouting</th><th className="num">Draft pos.</th><th className="num">Combined</th></tr>
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
            <b>The model beats the draft-position baseline</b> on star-detection (AUC{" "}
            {s.model.star.auc} vs {s.baseline.star.auc}), calibration (ECE {s.model.star.ece} vs{" "}
            {s.baseline.star.ece}), career-value ranking ({s.model.vorp.spearman} vs{" "}
            {s.baseline.vorp.spearman}) and within-1-tier accuracy ({s.model.tier.within_one} vs{" "}
            {s.baseline.tier.within_one}).{backtest.summary.holdout && (
              <> On the strict held-out cohorts it holds up (AUC {backtest.summary.holdout.scouting.auc}{" "}
              vs {backtest.summary.holdout.baseline.auc}; ranking{" "}
              {backtest.summary.holdout.scouting.spearman} vs {backtest.summary.holdout.baseline.spearman}).</>
            )}
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
          <li>Draft capital carries most of the signal. Recruiting rank, conference strength,
            age-relative-to-class and combine athleticism were all added and back-tested — none
            improved out-of-sample prediction (draft slot already encodes them), so they were
            left out rather than added as noise. An honest negative result.</li>
          <li>Profile signal alone is weak (the profile-only model loses to draft position); the
            value comes from combining profile with draft capital. Uncertainty stays wide and
            explicit — no false precision.</li>
          <li>Small early-cohort samples; survivorship and era effects. Training uses earlier
            cohorts' eventually-realized careers (fair to both model and baseline). Roster-fit
            captures skills, not contracts, health, or scheme.</li>
        </ul>
      </div>
    </section>
  );
}
