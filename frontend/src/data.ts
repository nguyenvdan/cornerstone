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
