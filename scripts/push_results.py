"""Push artifacts/reports/results.json (written by scripts.train) to the database.

usage: python -m scripts.push_results
"""
from __future__ import annotations

import json
import socket

from safenet import db
from safenet.config import REPORTS_DIR, database_url


def main() -> None:
    rep = json.loads((REPORTS_DIR / "results.json").read_text())
    url = database_url()
    db.init_schema(url)
    with db.connect(url) as conn:
        run_id = db.create_run(conn, rep["version"], socket.gethostname(), rep["notes"])
        for r in rep["results"]:
            db.add_eval(conn, run_id, r["experiment"], r["model"], r["train_ds"], r["test_ds"], r["metrics"], r["setting"])
        for imp in rep["importance"]:
            db.add_importance(conn, run_id, imp["model"], imp["dataset"], imp["items"])
        conn.commit()
    print(f"pushed {len(rep['results'])} results for model {rep['version']} (run {run_id})")


if __name__ == "__main__":
    main()
