import React from "react";

/** Minimal Markdown renderer for the agent's generated report. */
function inline(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong> : <span key={i}>{p}</span>
  );
}

function Markdown({ src }: { src: string }) {
  const lines = src.split("\n");
  const out: React.ReactNode[] = [];
  let list: React.ReactNode[] = [];
  const flush = () => {
    if (list.length) { out.push(<ul key={`ul${out.length}`}>{list}</ul>); list = []; }
  };
  lines.forEach((ln, i) => {
    if (ln.startsWith("## ")) { flush(); out.push(<h2 key={i}>{inline(ln.slice(3))}</h2>); }
    else if (ln.startsWith("# ")) { flush(); out.push(<h1 key={i}>{inline(ln.slice(2))}</h1>); }
    else if (ln.startsWith("- ")) { list.push(<li key={i}>{inline(ln.slice(2))}</li>); }
    else if (ln.startsWith("> ")) { flush(); out.push(<blockquote key={i}>{inline(ln.slice(2))}</blockquote>); }
    else if (/^_.+_$/.test(ln.trim())) { flush(); out.push(<p key={i}><em>{ln.trim().slice(1, -1)}</em></p>); }
    else if (ln.trim() === "") { flush(); }
    else { flush(); out.push(<p key={i}>{inline(ln)}</p>); }
  });
  flush();
  return <>{out}</>;
}

export default function Agent({ agent }: { agent: any }) {
  return (
    <section className="view">
      <h2 className="h2">Ask the agent</h2>
      <p className="sub">
        An autonomous agent plans a multi-step analysis, calls the modeling tools, and
        synthesizes a cited report. Every number traces to a tool output.
      </p>

      <div className="callout" style={{ marginBottom: 18 }}>
        <b>Question:</b> “{agent.query}” &nbsp;·&nbsp; mode: <b>{agent.mode}</b>.
        This deployed demo shows a precomputed run; the agent also runs live from the CLI
        (autonomous LLM tool-calling with an API key, or scripted without one).
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Reasoning trace</h3>
          <ol className="trace">
            {agent.steps.filter((s: any) => s.tool !== "—").map((s: any) => (
              <li key={s.n}>
                <span className="n">{s.n}</span>
                <div className="body">
                  <div className="thought">{s.thought}</div>
                  <div className="call">
                    {s.tool}({Object.entries(s.args).map(([k, v]) => `${k}=${v}`).join(", ")}) ⇒ {s.result_summary}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </div>

        <div className="report">
          <Markdown src={agent.report_markdown} />
        </div>
      </div>
    </section>
  );
}
