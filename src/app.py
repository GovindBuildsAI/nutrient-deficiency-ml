"""FastAPI back-end: schema, prediction + SHAP report, model-comparison data, anonymised submission log.

Run:  PYTHONPATH=src .venv/bin/uvicorn app:app --port 8000      (serves the built React UI at /)
"""
import json, sqlite3, time
from contextlib import asynccontextmanager, closing
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from report import RiskReporter
from schema import ui_schema

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/submissions.sqlite"
UI = ROOT / "webapp/dist"
state = {}


def db():
    return closing(sqlite3.connect(DB))


@asynccontextmanager
async def lifespan(app):
    state["reporter"] = RiskReporter()
    with db() as c:   # anonymous: no name, no IP, no user id - only answers and the prediction
        c.execute("""CREATE TABLE IF NOT EXISTS submissions(id INTEGER PRIMARY KEY, ts REAL, inputs TEXT, predicted TEXT, scores TEXT)""")
        c.commit()
    yield


app = FastAPI(title="Nutrient Deficiency Risk Screener", lifespan=lifespan)


class Answers(BaseModel):
    answers: dict


@app.get("/api/schema")
def schema():
    return ui_schema()


@app.get("/api/samples")
def samples():
    return state["reporter"].samples()


@app.post("/api/predict")
def predict(body: Answers):
    try:
        rep = state["reporter"].report(body.answers)
    except ValueError as e:
        raise HTTPException(422, str(e))
    with db() as c:
        c.execute("INSERT INTO submissions(ts, inputs, predicted, scores) VALUES (?,?,?,?)",
                  (time.time(), json.dumps(body.answers), rep["predicted"], json.dumps(rep["scores"])))
        c.commit()
    return rep


@app.get("/api/model-info")
def model_info():
    r = state["reporter"]
    comp = pd.read_csv(ROOT / "results/model_comparison.csv", index_col=0).reset_index().to_dict("records")
    sel = r.sel
    return dict(best=r.best, comparison=comp, classes=r.classes, friedman_p=sel["friedman_p"], n_folds=sel["n_folds"],
                holm_p=sel["winner_vs_rival_holm_p"], holdout=sel["holdout"], baseline_accuracy=sel["baseline_accuracy"],
                baseline_f1=sel["baseline_f1"], prevalence=r.prevalence)


@app.get("/api/stats")
def stats():
    with db() as c:
        n = c.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        by = c.execute("SELECT predicted, COUNT(*) FROM submissions GROUP BY predicted").fetchall()
    return dict(total=n, by_class=dict(by))


if UI.exists():
    app.mount("/assets", StaticFiles(directory=UI / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = UI / path
        return FileResponse(f if f.is_file() else UI / "index.html")
