export function Bar({
  label, value, max = 100, suffix = "", color = "var(--navy-2)",
}: { label: string; value: number; max?: number; suffix?: string; color?: string }) {
  const w = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="bar-row">
      <div className="lbl">{label}</div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${w}%`, background: color }} />
      </div>
      <div className="val">{value}{suffix}</div>
    </div>
  );
}

/** Season development curve: p25–p75 band with a median line. */
export function SeasonCurve({ curve }: { curve: any[] }) {
  if (!curve || curve.length === 0) return <p className="sub">No curve data.</p>;
  const W = 460, H = 220, padL = 38, padB = 28, padT = 12, padR = 12;
  const xs = curve.map((d) => d.season);
  const lo = Math.min(...curve.map((d) => d.p10));
  const hi = Math.max(...curve.map((d) => d.p90));
  const yMin = Math.min(0, lo) - 0.3;
  const yMax = hi + 0.3;
  const x = (s: number) =>
    padL + ((s - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * (W - padL - padR);
  const y = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - padT - padB);

  const band =
    curve.map((d) => `${x(d.season)},${y(d.p75)}`).join(" ") + " " +
    curve.slice().reverse().map((d) => `${x(d.season)},${y(d.p25)}`).join(" ");
  const median = curve.map((d) => `${x(d.season)},${y(d.p50)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Projected VORP by season">
      {/* zero line */}
      <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)} stroke="#cdd5df" strokeDasharray="3 3" />
      <polygon points={band} fill="var(--navy-2)" opacity={0.16} />
      <polyline points={median} fill="none" stroke="var(--navy-2)" strokeWidth={2.5} />
      {curve.map((d) => (
        <circle key={d.season} cx={x(d.season)} cy={y(d.p50)} r={3.5} fill="var(--navy)" />
      ))}
      {curve.map((d) => (
        <text key={`t${d.season}`} x={x(d.season)} y={H - 8} textAnchor="middle"
          fontSize={11} fill="var(--muted-solid)">Y{d.season}</text>
      ))}
      <text x={6} y={y(yMax) + 8} fontSize={10} fill="var(--muted-solid)">{yMax.toFixed(0)}</text>
      <text x={6} y={y(yMin)} fontSize={10} fill="var(--muted-solid)">{yMin.toFixed(0)}</text>
      <text x={6} y={y(0) - 3} fontSize={10} fill="var(--muted-solid)">0</text>
    </svg>
  );
}
