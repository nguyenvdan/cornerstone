import { useEffect, useState } from "react";

export interface Bundle {
  projection: any;
  comparables: any;
  fit: any;
  agent: any;
  backtest: any;
}

const FILES: Record<keyof Bundle, string> = {
  projection: "/data/dybantsa_projection.json",
  comparables: "/data/dybantsa_comparables.json",
  fit: "/data/wizards_fit.json",
  agent: "/data/agent_report.json",
  backtest: "/data/backtest_results.json",
};

export function useBundle() {
  const [data, setData] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all(
      (Object.entries(FILES) as [keyof Bundle, string][]).map(([k, url]) =>
        fetch(url)
          .then((r) => {
            if (!r.ok) throw new Error(`${url}: ${r.status}`);
            return r.json();
          })
          .then((j) => [k, j] as const)
      )
    )
      .then((pairs) => setData(Object.fromEntries(pairs) as unknown as Bundle))
      .catch((e) => setError(String(e)));
  }, []);

  return { data, error };
}

export const TIER_LABELS: Record<string, string> = {
  superstar: "Superstar",
  all_star: "All-Star",
  starter: "Starter",
  rotation: "Rotation",
  bust: "Bust",
};

export const TIER_ORDER = ["superstar", "all_star", "starter", "rotation", "bust"];

export const pct = (x: number) => `${Math.round(x * 100)}%`;
export const titleCase = (s: string) =>
  s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** Career-VORP → plain-English tier (rule-of-thumb scale). */
export function vorpLabel(v: number): string {
  if (v >= 80) return "All-time great";
  if (v >= 40) return "Hall of Fame caliber";
  if (v >= 20) return "Multiple-time All-Star";
  if (v >= 10) return "Quality starter / good career";
  if (v >= 3) return "Rotation player / spot starter";
  if (v >= 0) return "Replacement level";
  return "Bust / out of the league";
}

/** Inches → e.g. 84.5 -> 7'0.5" */
export function inchesToFeet(inches: number): string {
  const ft = Math.floor(inches / 12);
  const rem = inches - ft * 12;
  const remStr = Number.isInteger(rem) ? `${rem}` : rem.toFixed(1);
  return `${ft}'${remStr}"`;
}
