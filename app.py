from __future__ import annotations

import json, time, uuid, logging, threading, os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ortools.sat.python import cp_model

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_config(c):
    required = ["grades", "subjects", "teachers", "timeSlots", "workingDays"]
    for r in required:
        if r not in c:
            return False, f"Missing {r}", []
    return True, None, []

# ----------------------------------------------------------------------
# Data Index (required lessons per grade/subject, teacher availability)
# ----------------------------------------------------------------------
@dataclass
class ScheduleIndex:
    teacher_assignments: Dict = field(default_factory=dict)
    required_lessons: Dict = field(default_factory=dict)
    teacher_avail_days: Dict = field(default_factory=dict)
    var_index: Dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    teacher_var_index: Dict = field(default_factory=lambda: defaultdict(list))

    def build(self, config, working_days):
        teachers = config["teachers"]
        subjects = config["subjects"]

        for t, td in teachers.items():
            ua = set(td.get("unavailDays", []))
            self.teacher_avail_days[t] = {i for i, d in enumerate(working_days) if d not in ua}
            self.teacher_assignments[t] = td.get("assignments", [])

        for g in config["grades"]:
            for s in subjects:
                total = 0
                for t in teachers:
                    for a in self.teacher_assignments[t]:
                        if a["grade"] == g and a["subject"] == s:
                            total += a["lessons"]
                self.required_lessons[(g, s)] = total

# ----------------------------------------------------------------------
# Model Builder (strict teacher‑slot constraint = 1)
# ----------------------------------------------------------------------
class ModelBuilder:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()

        self.grades = config["grades"]
        self.subjects = config["subjects"]
        self.teachers = config["teachers"]
        self.days = config["workingDays"]
        self.slots = [s for s in config["timeSlots"] if s["type"] == "lesson"]

        self.idx = ScheduleIndex()
        self.idx.build(config, self.days)

        self.x = {}
        self.penalties = []

        self._create_vars()

    def _create_vars(self):
        for g in self.grades:
            self.x[g] = {}
            for d in range(len(self.days)):
                self.x[g][d] = {}
                for s in range(len(self.slots)):
                    self.x[g][d][s] = {}
                    for t in self.teachers:
                        if d not in self.idx.teacher_avail_days[t]:
                            continue
                        for a in self.idx.teacher_assignments[t]:
                            if a["grade"] != g:
                                continue
                            sub = a["subject"]
                            v = self.model.NewBoolVar(f"x_{g}_{d}_{s}_{t}_{sub}")
                            self.x[g][d][s].setdefault(t, {})[sub] = v
                            self.idx.var_index[(g, sub)].append(v)
                            self.idx.teacher_var_index[t].append(v)

    def add_hard(self):
        # One class per slot per grade
        for g in self.grades:
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    vars_ = [v for tv in self.x[g][d][s].values() for v in tv.values()]
                    if vars_:
                        self.model.Add(sum(vars_) <= 1)

        # Teacher can teach at most ONE class per slot (strict)
        for t in self.teachers:
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    vars_ = []
                    for g in self.grades:
                        vars_ += list(self.x[g][d][s].get(t, {}).values())
                    if vars_:
                        self.model.Add(sum(vars_) <= 1)

    def add_soft(self):
        for (g, sub), req in self.idx.required_lessons.items():
            vars_ = self.idx.var_index[(g, sub)]
            if not vars_:
                continue
            slack = self.model.NewIntVar(0, req, f"slack_{g}_{sub}")
            self.model.Add(sum(vars_) + slack == req)
            self.penalties.append(slack * 1000)

    def set_obj(self):
        if self.penalties:
            self.model.Minimize(sum(self.penalties))

# ----------------------------------------------------------------------
# Extractor (timetable from solved model)
# ----------------------------------------------------------------------
class Extractor:
    def __init__(self, builder, solver):
        self.b = builder
        self.s = solver

    def extract(self):
        out = {}
        for g in self.b.grades:
            out[g] = []
            for d in range(len(self.b.days)):
                row = []
                for s in range(len(self.b.slots)):
                    val = None
                    for t, sv in self.b.x[g][d][s].items():
                        for sub, v in sv.items():
                            if self.s.Value(v):
                                val = {"teacher": t, "subject": sub}
                    row.append(val)
                out[g].append(row)
        return out

# ----------------------------------------------------------------------
# Auto‑reduction: binary search per subject to find minimal lesson cuts
# ----------------------------------------------------------------------
def auto_reduce(config):
    """
    Returns (new_config, timetable, reduction_info) if a feasible strict schedule
    becomes possible after minimal lesson reductions; otherwise (None, None, suggestions).
    """
    builder_orig = ModelBuilder(config)
    total_before = sum(builder_orig.idx.required_lessons.values())

    # For each subject, binary search the minimal total reduction needed
    subject_reductions = {}
    for sub in config["subjects"]:
        # Total lessons of this subject in original config
        total_sub = 0
        for (g, s), val in builder_orig.idx.required_lessons.items():
            if s == sub:
                total_sub += val
        if total_sub == 0:
            continue

        lo, hi = 0, total_sub
        best = total_sub  # worst case: remove all of this subject
        while lo <= hi:
            mid = (lo + hi) // 2
            # Build a test config with `mid` lessons removed from this subject
            test_config = json.loads(json.dumps(config))
            remaining = mid
            # collect all assignments for this subject, sort descending
            assigns = []
            for t, tdata in test_config["teachers"].items():
                for a in tdata.get("assignments", []):
                    if a["subject"] == sub:
                        assigns.append(a)
            assigns.sort(key=lambda a: a["lessons"], reverse=True)
            for a in assigns:
                if remaining <= 0:
                    break
                cut = min(a["lessons"], remaining)
                a["lessons"] -= cut
                remaining -= cut

            # Test feasibility with strict constraints
            b_test = ModelBuilder(test_config)
            b_test.add_hard()
            b_test.add_soft()
            b_test.set_obj()
            s_test = cp_model.CpSolver()
            s_test.parameters.max_time_in_seconds = 10
            status = s_test.Solve(b_test.model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if best > 0:
            subject_reductions[sub] = best

    if not subject_reductions:
        # no subject reduction helped – must be another bottleneck (e.g., teacher unavailable)
        return None, None, [{"message": "No subject‑based reduction resolves infeasibility. Check teacher availability and daily max rules."}]

    # Apply the found reductions to the original config
    final_config = json.loads(json.dumps(config))
    for sub, reduce_by in subject_reductions.items():
        remaining = reduce_by
        assigns = []
        for t, tdata in final_config["teachers"].items():
            for a in tdata.get("assignments", []):
                if a["subject"] == sub:
                    assigns.append(a)
        assigns.sort(key=lambda a: a["lessons"], reverse=True)
        for a in assigns:
            if remaining <= 0:
                break
            cut = min(a["lessons"], remaining)
            a["lessons"] -= cut
            remaining -= cut

    total_after = 0
    for tdata in final_config["teachers"].values():
        for a in tdata.get("assignments", []):
            total_after += a["lessons"]

    # Final solve with reduced config
    builder_final = ModelBuilder(final_config)
    builder_final.add_hard()
    builder_final.add_soft()
    builder_final.set_obj()
    solver_final = cp_model.CpSolver()
    solver_final.parameters.max_time_in_seconds = 20
    status_final = solver_final.Solve(builder_final.model)
    if status_final in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tt = Extractor(builder_final, solver_final).extract()
        info = [{
            "message": f"Auto‑reduced lessons by {total_before - total_after} total",
            "details": subject_reductions
        }]
        return final_config, tt, info
    else:
        # Should not happen, but fallback
        return None, None, [{"message": "Reductions applied but still infeasible"}]

# ----------------------------------------------------------------------
# Solver entry point (strict only, auto‑reduce if needed)
# ----------------------------------------------------------------------
def run_solver(config):
    # First try original config
    builder = ModelBuilder(config)
    builder.add_hard()
    builder.add_soft()
    builder.set_obj()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    status = solver.Solve(builder.model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tt = Extractor(builder, solver).extract()
        return tt, []

    # Otherwise auto‑reduce
    _, tt, info = auto_reduce(config)
    if tt:
        return tt, info
    else:
        return None, info

# ----------------------------------------------------------------------
# API Routes
# ----------------------------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    config = request.json
    ok, msg, _ = validate_config(config)
    if not ok:
        return jsonify({"success": False, "message": msg})

    tt, info = run_solver(config)
    if tt:
        return jsonify({"success": True, "timetable": tt, "info": info})
    else:
        return jsonify({"success": False, "suggestions": info})

@app.route("/")
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>EduSchedule Pro – Auto‑Reduction</title>
        <style>
            body { font-family: Arial; margin: 2rem; }
            pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 5px; }
            button { padding: 0.5rem 1rem; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>📅 Timetable Generator (Strict Teacher‑Slot)</h1>
        <p>Upload your configuration (JSON) – the system automatically reduces the fewest lessons if needed.</p>
        <input type="file" id="configFile" accept=".json">
        <button onclick="generate()">Generate Timetable</button>
        <h3>Result</h3>
        <pre id="result">Awaiting input...</pre>
        <script>
            async function generate() {
                const file = document.getElementById('configFile').files[0];
                if (!file) return;
                const text = await file.text();
                const config = JSON.parse(text);
                document.getElementById('result').innerText = "⏳ Solving...";
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const data = await res.json();
                document.getElementById('result').innerText = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    ''')

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)