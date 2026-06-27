import { useState } from "react";
import { pct, useBundle } from "./data";
import Projection from "./components/Projection";
import Comparables from "./components/Comparables";
import RosterFit from "./components/RosterFit";
import Agent from "./components/Agent";
import Methodology from "./components/Methodology";

const TABS = ["Projection", "Comparables", "Wizards build", "Ask the agent", "Methodology"];

export default function App() {
  const { data, error } = useBundle();
  const [tab, setTab] = useState(0);

  if (error) return <div className="loading">Failed to load data: {error}</div>;
  if (!data) return <div className="loading">Loading Cornerstone…</div>;

  const p = data.projection;
  const fitScore = data.fit.report.fit_score;
  const acc = data.backtest.summary.model.tier.within_one;

  return (
    <>
      <header className="hero">
        <div className="wrap">
          <div className="eyebrow">Decision support under uncertainty</div>
          <h1>AJ Dybantsa as the Wizards' cornerstone</h1>
          <p>
            An agentic, uncertainty-aware player-development and roster-fit projection system.
            How is Dybantsa likely to develop — and how should Washington build around him?
          </p>
          <div className="stat-row">
            <div className="stat"><div className="v">{pct(p.p_starter_plus)}</div><div className="l">P(starter or better)</div></div>
            <div className="stat"><div className="v">{pct(p.p_star_plus)}</div><div className="l">P(all-star or better)</div></div>
            <div className="stat"><div className="v">{fitScore}/100</div><div className="l">Wizards roster fit</div></div>
            <div className="stat"><div className="v">{Math.round(acc * 100)}%</div><div className="l">within-1-tier back-test</div></div>
          </div>
        </div>
      </header>

      <nav className="nav">
        <div className="wrap">
          {TABS.map((t, i) => (
            <button key={t} className={i === tab ? "active" : ""} onClick={() => setTab(i)}>{t}</button>
          ))}
        </div>
      </nav>

      <main>
        <div className="wrap">
          {tab === 0 && <Projection p={p} />}
          {tab === 1 && <Comparables c={data.comparables} />}
          {tab === 2 && <RosterFit fit={data.fit} />}
          {tab === 3 && <Agent agent={data.agent} />}
          {tab === 4 && <Methodology backtest={data.backtest} />}
        </div>
      </main>

      <footer>
        <div className="wrap">
          Cornerstone — built around AJ Dybantsa &amp; the Washington Wizards rebuild.
          Probabilistic projections with stated uncertainty; back-tested leakage-free vs a
          draft-position baseline. Data: Basketball Reference / Sports Reference (public).
        </div>
      </footer>
    </>
  );
}
