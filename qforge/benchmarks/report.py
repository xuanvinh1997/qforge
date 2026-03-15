# -*- coding: utf-8 -*-
"""HTML report generator for benchmark results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from qforge.benchmarks.results import BenchmarkResults
from qforge.benchmarks.charts import generate_suite_charts


def generate_report(results: BenchmarkResults, output_dir: Path) -> Path:
    """Generate a full benchmark report with charts and HTML.

    Creates:
        output_dir/charts/       — PNG chart images
        output_dir/results.json  — raw data
        output_dir/report.html   — standalone HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Generate charts per suite
    all_chart_paths: Dict[str, List[Path]] = {}
    for suite_name, suite_results in results.suites.items():
        paths = generate_suite_charts(suite_name, suite_results, charts_dir)
        all_chart_paths[suite_name] = paths

    # Save JSON
    results.save_json(output_dir / "results.json")

    # Generate HTML
    html_path = output_dir / "report.html"
    html = _build_html(results, all_chart_paths, charts_dir)
    html_path.write_text(html, encoding="utf-8")

    return html_path


# ---------------------------------------------------------------------------
# Suite display names
# ---------------------------------------------------------------------------

SUITE_TITLES = {
    "gates":       "1. Primitive Gate Operations",
    "circuits":    "2. Circuit Execution Patterns",
    "vqe":         "3. VQE Algorithm",
    "qaoa":        "4. QAOA Algorithm (Max-Cut)",
    "gradient":    "5. Gradient Computation",
    "measurement": "6. Measurement Operations",
    "scaling":     "7. Scalability",
    "accuracy":    "8. Accuracy & Correctness",
    "memory":      "9. Memory Usage",
    "mps":         "10. MPS Benchmarks",
    "dmrg":        "11. DMRG Benchmarks",
}


def _build_html(results: BenchmarkResults, chart_paths: Dict[str, List[Path]], charts_dir: Path) -> str:
    sections_html = []

    for suite_name in results.suites:
        title = SUITE_TITLES.get(suite_name, suite_name.upper())
        charts = chart_paths.get(suite_name, [])

        charts_html = ""
        for p in charts:
            rel = p.relative_to(charts_dir.parent)
            charts_html += f'      <img src="{rel}" alt="{p.stem}" class="chart">\n'

        # Build data table from results
        suite_data = results.suites[suite_name]
        table_html = _build_data_table(suite_data)

        sections_html.append(f"""
    <section>
      <h2>{title}</h2>
{charts_html}
      <details>
        <summary>Raw Data</summary>
        {table_html}
      </details>
    </section>""")

    sys_info = results.system_info
    sys_rows = "\n".join(f"        <tr><td>{k}</td><td>{v}</td></tr>" for k, v in sys_info.items())

    config = results.config
    cfg_rows = "\n".join(f"        <tr><td>{k}</td><td>{v}</td></tr>" for k, v in config.items())

    # Summary stats
    summary = _compute_summary(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Qforge Benchmark Report</title>
  <style>
    :root {{
      --bg: #fafafa; --fg: #222; --card: #fff; --border: #e0e0e0;
      --qforge: #2196F3; --pennylane: #4CAF50; --qiskit: #FF9800;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: var(--qforge); }}
    h2 {{ font-size: 1.4rem; margin: 1.5rem 0 1rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }}
    .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0 2rem; }}
    .meta table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .meta td {{ padding: 0.2rem 0.5rem; border-bottom: 1px solid var(--border); }}
    .meta td:first-child {{ font-weight: 600; width: 40%; }}
    section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
    .chart {{ max-width: 100%; height: auto; margin: 1rem 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    details {{ margin-top: 1rem; }}
    summary {{ cursor: pointer; font-weight: 600; color: var(--qforge); }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 0.5rem; }}
    .data-table th, .data-table td {{ padding: 0.3rem 0.6rem; border: 1px solid var(--border); text-align: right; }}
    .data-table th {{ background: #f5f5f5; text-align: left; }}
    .summary {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
    .summary-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
                     padding: 1rem; flex: 1; min-width: 200px; text-align: center; }}
    .summary-card .num {{ font-size: 1.8rem; font-weight: 700; color: var(--qforge); }}
    .summary-card .label {{ font-size: 0.85rem; color: #666; }}
    .footer {{ text-align: center; font-size: 0.8rem; color: #999; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
  </style>
</head>
<body>
  <h1>Qforge Benchmark Report</h1>
  <p>Generated: {results.timestamp}</p>

  <div class="summary">
    <div class="summary-card">
      <div class="num">{summary['n_suites']}</div>
      <div class="label">Suites Run</div>
    </div>
    <div class="summary-card">
      <div class="num">{summary['n_tests']}</div>
      <div class="label">Total Tests</div>
    </div>
    <div class="summary-card">
      <div class="num">{summary['qf_wins']}/{summary['n_comparable']}</div>
      <div class="label">Qforge Fastest</div>
    </div>
  </div>

  <div class="meta">
    <div>
      <h3>System Info</h3>
      <table>
{sys_rows}
      </table>
    </div>
    <div>
      <h3>Configuration</h3>
      <table>
{cfg_rows}
      </table>
    </div>
  </div>

{"".join(sections_html)}

  <div class="footer">
    Qforge Benchmark Suite &mdash; <a href="https://github.com/vinhpx/qforge">github.com/vinhpx/qforge</a>
  </div>
</body>
</html>"""


def _build_data_table(data: dict) -> str:
    if not data:
        return "<p>No data</p>"

    # Collect all keys from sub-dicts
    all_keys = set()
    for v in data.values():
        if isinstance(v, dict):
            all_keys.update(v.keys())
    all_keys = sorted(all_keys)

    if not all_keys:
        return "<p>No tabular data</p>"

    header = "<tr><th>Test</th>" + "".join(f"<th>{k}</th>" for k in all_keys) + "</tr>"
    rows = []
    for test_name, test_data in data.items():
        if not isinstance(test_data, dict):
            continue
        cells = []
        for k in all_keys:
            v = test_data.get(k)
            if v is None:
                cells.append("<td>N/A</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:.4g}</td>")
            elif isinstance(v, list):
                cells.append(f"<td>[{len(v)} items]</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append(f"<tr><td>{test_name}</td>{''.join(cells)}</tr>")

    return f'<table class="data-table">{header}{"".join(rows)}</table>'


def _compute_summary(results: BenchmarkResults) -> dict:
    n_suites = len(results.suites)
    n_tests = sum(len(v) for v in results.suites.values())
    qf_wins = 0
    n_comparable = 0

    for suite_data in results.suites.values():
        for test_data in suite_data.values():
            if not isinstance(test_data, dict):
                continue
            qf_t = test_data.get("qforge") or test_data.get("time")
            pl_t = test_data.get("pennylane")
            qk_t = test_data.get("qiskit")
            times = {k: v for k, v in [("qforge", qf_t), ("pennylane", pl_t), ("qiskit", qk_t)]
                     if v is not None and isinstance(v, (int, float))}
            if len(times) > 1:
                n_comparable += 1
                if min(times, key=times.get) == "qforge":
                    qf_wins += 1

    return {
        "n_suites": n_suites,
        "n_tests": n_tests,
        "qf_wins": qf_wins,
        "n_comparable": n_comparable,
    }
