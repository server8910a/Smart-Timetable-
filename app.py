from __future__ import annotations

import json, os, logging, traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ortools.sat.python import cp_model

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow any frontend to call this API

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_config(c):
    required = ["grades", "subjects", "teachers", "timeSlots", "workingDays"]
    for r in required:
        if r not in c:
            return False, f"Missing '{r}'"
    if not isinstance(c.get("timeSlots"), list):
        return False, "'timeSlots' must be a list"
    for idx, slot in enumerate(c["timeSlots"]):
        if "type" not in slot:
            return False, f"timeSlots[{idx}] missing 'type' (maybe 'Docrype'?)"
    lesson_slots = [s for s in c["timeSlots"] if s.get("type") == "lesson"]
    if not lesson_slots:
        return False, "No timeSlot with type='lesson' found"
    return True, None

# ----------------------------------------------------------------------
# Data Index
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
                        if a.get("grade") == g and a.get("subject") == s:
                            total += a.get("lessons", 0)
                self.required_lessons[(g, s)] = total

# ----------------------------------------------------------------------
# Model Builder (strict teacher per slot)
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
                        if d not in self.idx.teacher_avail_days.get(t, set()):
                            continue
                        for a in self.idx.teacher_assignments.get(t, []):
                            if a.get("grade") != g:
                                continue
                            sub = a.get("subject")
                            if not sub:
                                continue
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
            vars_ = self.idx.var_index.get((g, sub), [])
            if not vars_:
                continue
            slack = self.model.NewIntVar(0, req, f"slack_{g}_{sub}")
            self.model.Add(sum(vars_) + slack == req)
            self.penalties.append(slack * 1000)

    def set_obj(self):
        if self.penalties:
            self.model.Minimize(sum(self.penalties))

# ----------------------------------------------------------------------
# Extractor
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
# Auto‑reduction (core logic)
# ----------------------------------------------------------------------
def auto_reduce(config):
    """Returns (new_config, timetable, info) after minimal reductions."""
    orig_builder = ModelBuilder(config)
    total_before = sum(orig_builder.idx.required_lessons.values())

    subject_reductions = {}
    for sub in config["subjects"]:
        total_sub = 0
        for (g, s), val in orig_builder.idx.required_lessons.items():
            if s == sub:
                total_sub += val
        if total_sub == 0:
            continue

        low, high = 0, total_sub
        best = total_sub
        while low <= high:
            mid = (low + high) // 2
            test_config = json.loads(json.dumps(config))
            remaining = mid
            assigns = []
            for t, tdata in test_config["teachers"].items():
                for a in tdata.get("assignments", []):
                    if a.get("subject") == sub:
                        assigns.append(a)
            assigns.sort(key=lambda a: a.get("lessons", 0), reverse=True)
            for a in assigns:
                if remaining <= 0:
                    break
                old = a.get("lessons", 0)
                cut = min(old, remaining)
                a["lessons"] = old - cut
                remaining -= cut

            b_test = ModelBuilder(test_config)
            b_test.add_hard()
            b_test.add_soft()
            b_test.set_obj()
            solver_test = cp_model.CpSolver()
            solver_test.parameters.max_time_in_seconds = 10
            status = solver_test.Solve(b_test.model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                best = mid
                high = mid - 1
            else:
                low = mid + 1

        if best > 0:
            subject_reductions[sub] = best

    if not subject_reductions:
        return None, None, [{"message": "No subject reduction resolves infeasibility. Check teacher availability."}]

    # Apply reductions
    final_config = json.loads(json.dumps(config))
    for sub, reduce_by in subject_reductions.items():
        remaining = reduce_by
        assigns = []
        for t, tdata in final_config["teachers"].items():
            for a in tdata.get("assignments", []):
                if a.get("subject") == sub:
                    assigns.append(a)
        assigns.sort(key=lambda a: a.get("lessons", 0), reverse=True)
        for a in assigns:
            if remaining <= 0:
                break
            old = a.get("lessons", 0)
            cut = min(old, remaining)
            a["lessons"] = old - cut
            remaining -= cut

    builder_final = ModelBuilder(final_config)
    builder_final.add_hard()
    builder_final.add_soft()
    builder_final.set_obj()
    solver_final = cp_model.CpSolver()
    solver_final.parameters.max_time_in_seconds = 20
    status_final = solver_final.Solve(builder_final.model)
    if status_final in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tt = Extractor(builder_final, solver_final).extract()
        total_after = sum(builder_final.idx.required_lessons.values())
        info = [{"message": f"Auto‑reduced lessons by {total_before - total_after}", "details": subject_reductions}]
        return final_config, tt, info
    else:
        return None, None, [{"message": "Reductions applied but still infeasible"}]

# ----------------------------------------------------------------------
# Solver entry point
# ----------------------------------------------------------------------
def run_solver(config):
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

    _, tt, info = auto_reduce(config)
    if tt:
        return tt, info
    else:
        return None, info

# ----------------------------------------------------------------------
# API routes – all return JSON (no HTML)
# ----------------------------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    # Always return JSON, even for errors
    try:
        data = request.get_data(as_text=True)
        if not data:
            return jsonify({"success": False, "message": "Empty request body"}), 400
        config = json.loads(data)
    except json.JSONDecodeError as e:
        return jsonify({"success": False, "message": f"Invalid JSON: {str(e)}. Check for typos like 'Docrype'."}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading request: {str(e)}"}), 400

    ok, msg = validate_config(config)
    if not ok:
        return jsonify({"success": False, "message": msg}), 400

    try:
        tt, info = run_solver(config)
        if tt:
            return jsonify({"success": True, "timetable": tt, "info": info})
        else:
            return jsonify({"success": False, "suggestions": info})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "message": f"Internal solver error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    # Return a simple JSON status (or a minimal HTML for humans)
    return jsonify({"status": "EduSchedule Pro backend is running", "endpoints": ["/generate (POST)"]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)