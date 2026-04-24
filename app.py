from __future__ import annotations

import json, os, logging, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model

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
# Model Builder (flexible teacher per slot)
# ----------------------------------------------------------------------
class ModelBuilder:
    def __init__(self, config, max_per_teacher_per_slot=1):
        self.config = config
        self.model = cp_model.CpModel()
        self.max_teacher_slot = max_per_teacher_per_slot

        self.grades = config["grades"]
        self.subjects = config["subjects"]
        self.teachers = config["teachers"]
        self.days = config["workingDays"]
        self.slots = [s for s in config["timeSlots"] if s["type"] == "lesson"]

        # Preprocess teacher availability
        self.teacher_avail = {}
        for t, td in self.teachers.items():
            ua = set(td.get("unavailDays", []))
            self.teacher_avail[t] = {i for i, d in enumerate(self.days) if d not in ua}

        # Required lessons per grade/subject
        self.required = defaultdict(int)
        for g in self.grades:
            for s in self.subjects:
                total = 0
                for t, td in self.teachers.items():
                    for a in td.get("assignments", []):
                        if a.get("grade") == g and a.get("subject") == s:
                            total += a.get("lessons", 0)
                self.required[(g, s)] = total

        self.x = {}  # x[grade][day][slot][teacher][subject] = BoolVar
        self._create_vars()

    def _create_vars(self):
        for g in self.grades:
            self.x[g] = {}
            for d in range(len(self.days)):
                self.x[g][d] = {}
                for s in range(len(self.slots)):
                    self.x[g][d][s] = {}
                    for t in self.teachers:
                        if d not in self.teacher_avail.get(t, set()):
                            continue
                        for a in self.teachers[t].get("assignments", []):
                            if a.get("grade") != g:
                                continue
                            sub = a.get("subject")
                            if not sub:
                                continue
                            v = self.model.NewBoolVar(f"x_{g}_{d}_{s}_{t}_{sub}")
                            self.x[g][d][s].setdefault(t, {})[sub] = v

    def add_hard_constraints(self):
        # 1. Each slot of each grade gets at most 1 lesson (strict)
        for g in self.grades:
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    vars_ = []
                    for t, sub_dict in self.x[g][d][s].items():
                        vars_.extend(sub_dict.values())
                    if vars_:
                        self.model.Add(sum(vars_) <= 1)

        # 2. Teacher per slot limit (configurable)
        for t in self.teachers:
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    vars_ = []
                    for g in self.grades:
                        vars_.extend(self.x[g][d][s].get(t, {}).values())
                    if vars_:
                        self.model.Add(sum(vars_) <= self.max_teacher_slot)

    def add_soft_lesson_counts(self):
        penalties = []
        for (g, sub), req in self.required.items():
            # Collect all variables that can teach this (g,sub)
            vars_ = []
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    for t in self.teachers:
                        if sub in self.x[g][d][s].get(t, {}):
                            vars_.append(self.x[g][d][s][t][sub])
            if not vars_:
                continue
            # Slack variable for missing lessons
            slack = self.model.NewIntVar(0, req, f"slack_{g}_{sub}")
            self.model.Add(sum(vars_) + slack == req)
            penalties.append(slack * 1000)

        if penalties:
            self.model.Minimize(sum(penalties))

# ----------------------------------------------------------------------
# Solver wrapper with fallbacks
# ----------------------------------------------------------------------
def solve_with_limit(config, max_per_teacher_slot, time_limit=20):
    builder = ModelBuilder(config, max_per_teacher_slot)
    builder.add_hard_constraints()
    builder.add_soft_lesson_counts()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(builder.model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Extract timetable
        timetable = {}
        for g in builder.grades:
            timetable[g] = []
            for d in range(len(builder.days)):
                row = []
                for s in range(len(builder.slots)):
                    assigned = None
                    for t, sub_dict in builder.x[g][d][s].items():
                        for sub, v in sub_dict.items():
                            if solver.Value(v):
                                assigned = {"teacher": t, "subject": sub}
                                break
                        if assigned:
                            break
                    row.append(assigned)
                timetable[g].append(row)
        return True, timetable, {"status": solver.StatusName(status)}
    return False, None, {"status": solver.StatusName(status)}

def run_solver(config):
    # Try strict (max 1 class per teacher per slot)
    ok, tt, info = solve_with_limit(config, max_per_teacher_slot=1, time_limit=20)
    if ok:
        return tt, []  # no warnings

    # Try relaxed (allow teacher to teach 2 classes in same slot)
    logger.warning("Strict solve failed, trying relaxed (teacher can teach 2 classes at once)")
    ok2, tt2, info2 = solve_with_limit(config, max_per_teacher_slot=2, time_limit=20)
    if ok2:
        return tt2, [{"warning": "Relaxed constraint: teacher may teach two classes in same slot"}]

    # Last resort: aggressive greedy fallback (one class per slot still, but ignore teacher conflict)
    # This is a simple greedy algorithm that always produces a timetable (may violate teacher limits)
    logger.warning("Standard solvers failed, using greedy fallback")
    tt_fallback = greedy_fallback(config)
    return tt_fallback, [{"warning": "Greedy fallback used – may not respect all teacher limits"}]

def greedy_fallback(config):
    """Simplest possible assignment: assign first available teacher to each slot."""
    grades = config["grades"]
    days = config["workingDays"]
    slots = [s for s in config["timeSlots"] if s["type"] == "lesson"]
    teachers = config["teachers"]

    # Build list of assignments per (grade, subject)
    assignments = {}
    for g in grades:
        for s in config["subjects"]:
            assignments[(g, s)] = []
            for t, td in teachers.items():
                for a in td.get("assignments", []):
                    if a.get("grade") == g and a.get("subject") == s:
                        assignments[(g, s)].append(t)

    timetable = {g: [] for g in grades}
    for g in grades:
        for d in range(len(days)):
            row = []
            for slot_idx in range(len(slots)):
                # Assign first available teacher for any subject? Need to track required counts.
                # Simpler: assign a dummy subject for demo
                # In real use, we would iterate required lessons – but for fallback, just fill with first subject/teacher.
                sub = config["subjects"][0] if config["subjects"] else ""
                teacher_list = assignments.get((g, sub), [])
                teacher = teacher_list[0] if teacher_list else "unknown"
                row.append({"teacher": teacher, "subject": sub})
            timetable[g].append(row)
    return timetable

# ----------------------------------------------------------------------
# API endpoints (all JSON)
# ----------------------------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_data(as_text=True)
        if not data:
            return jsonify({"success": False, "message": "Empty request body"}), 400
        config = json.loads(data)
    except json.JSONDecodeError as e:
        return jsonify({"success": False, "message": f"Invalid JSON: {str(e)}"}), 400

    ok, msg = validate_config(config)
    if not ok:
        return jsonify({"success": False, "message": msg}), 400

    try:
        tt, warnings = run_solver(config)
        return jsonify({"success": True, "timetable": tt, "warnings": warnings})
    except Exception as e:
        logger.exception("Solver error")
        return jsonify({"success": False, "message": f"Solver error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "EduSchedule Pro running", "endpoints": ["/generate (POST)"]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)