from __future__ import annotations

import json, os, logging
from collections import defaultdict
from typing import Dict, List

from flask import Flask, request, jsonify, render_template_string
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

        # Teacher availability
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
        # One class per slot per grade
        for g in self.grades:
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    vars_ = []
                    for t, sub_dict in self.x[g][d][s].items():
                        vars_.extend(sub_dict.values())
                    if vars_:
                        self.model.Add(sum(vars_) <= 1)

        # Teacher can teach at most ONE class per slot (strict)
        for t in self.teachers:
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    vars_ = []
                    for g in self.grades:
                        vars_.extend(self.x[g][d][s].get(t, {}).values())
                    if vars_:
                        self.model.Add(sum(vars_) <= 1)

    def add_soft_lesson_counts(self):
        penalties = []
        for (g, sub), req in self.required.items():
            vars_ = []
            for d in range(len(self.days)):
                for s in range(len(self.slots)):
                    for t in self.teachers:
                        if sub in self.x[g][d][s].get(t, {}):
                            vars_.append(self.x[g][d][s][t][sub])
            if not vars_:
                continue
            slack = self.model.NewIntVar(0, req, f"slack_{g}_{sub}")
            self.model.Add(sum(vars_) + slack == req)
            penalties.append(slack * 1000)
        if penalties:
            self.model.Minimize(sum(penalties))

# ----------------------------------------------------------------------
# Solver with auto‑reduction (respects max reduction per subject)
# ----------------------------------------------------------------------
def solve_with_limit(config, time_limit=20):
    """Solve with strict constraints, return (success, timetable)"""
    builder = ModelBuilder(config)
    builder.add_hard_constraints()
    builder.add_soft_lesson_counts()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(builder.model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
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
        return True, timetable
    return False, None

def auto_reduce(config, protected_subjects, max_reduction_per_subject=2):
    """
    Reduces unprotected subjects by the minimal amount (0 to max_reduction_per_subject)
    to achieve feasibility. Returns (new_config, timetable, info) or (None, None, error).
    """
    protected = set(protected_subjects or [])
    orig_builder = ModelBuilder(config)
    total_before = sum(orig_builder.required.values())

    reductions = {}
    for sub in config["subjects"]:
        if sub in protected:
            continue
        total_sub = sum(val for (g, s), val in orig_builder.required.items() if s == sub)
        if total_sub == 0:
            continue
        # Try reduction amounts 0, 1, 2, ... up to max
        best = 0
        for r in range(0, min(max_reduction_per_subject, total_sub) + 1):
            test_config = json.loads(json.dumps(config))
            remaining = r
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
            ok, _ = solve_with_limit(test_config, time_limit=10)
            if ok:
                best = r
                break
        if best > 0:
            reductions[sub] = best

    if not reductions:
        return None, None, [{"message": f"No reduction (up to {max_reduction_per_subject} per subject) made the schedule feasible."}]

    # Apply reductions to original config
    final_config = json.loads(json.dumps(config))
    for sub, reduce_by in reductions.items():
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

    ok_final, tt = solve_with_limit(final_config, time_limit=20)
    if ok_final:
        total_after = sum(ModelBuilder(final_config).required.values())
        info = [{
            "message": f"Auto‑reduced lessons by {total_before - total_after} total",
            "details": reductions
        }]
        return final_config, tt, info
    else:
        return None, None, [{"message": "Reductions applied but still infeasible (try increasing max reduction or check teacher availability)."}]

# ----------------------------------------------------------------------
# Main solver entry point
# ----------------------------------------------------------------------
def run_solver(config, protected_subjects, max_reduction_per_subject):
    # Try original config
    ok, tt = solve_with_limit(config)
    if ok:
        return tt, []

    # Auto‑reduce
    _, tt, info = auto_reduce(config, protected_subjects, max_reduction_per_subject)
    if tt:
        return tt, info

    # Ultimate fallback: use relaxed teacher limit (2 classes per slot)
    logger.warning("Auto‑reduce failed, falling back to relaxed teacher limit (2 per slot)")
    tt_relaxed = solve_relaxed_teacher(config)
    if tt_relaxed:
        return tt_relaxed, [{"warning": "Relaxed: teacher may teach up to 2 classes in same slot"}]
    else:
        return None, [{"error": "Could not generate timetable even with relaxed constraints. Check config."}]

def solve_relaxed_teacher(config):
    """Relax teacher constraint to max 2 classes per slot"""
    builder = ModelBuilder(config)
    builder.model = cp_model.CpModel()
    builder._create_vars()  # recreate vars with new model
    # One class per slot per grade (still strict)
    for g in builder.grades:
        for d in range(len(builder.days)):
            for s in range(len(builder.slots)):
                vars_ = []
                for t, sub_dict in builder.x[g][d][s].items():
                    vars_.extend(sub_dict.values())
                if vars_:
                    builder.model.Add(sum(vars_) <= 1)
    # Teacher limit = 2
    for t in builder.teachers:
        for d in range(len(builder.days)):
            for s in range(len(builder.slots)):
                vars_ = []
                for g in builder.grades:
                    vars_.extend(builder.x[g][d][s].get(t, {}).values())
                if vars_:
                    builder.model.Add(sum(vars_) <= 2)
    builder.add_soft_lesson_counts()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    status = solver.Solve(builder.model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
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
        return timetable
    return None

# ----------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>EduSchedule Pro – Protected Subjects & Auto‑Reduction</title>
        <style>
            body { font-family: Arial; margin: 2rem; max-width: 800px; }
            pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 5px; }
            button, input[type="file"] { margin: 0.5rem 0; padding: 0.5rem; }
            .subject-list { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
            .subject-item { background: #eef; padding: 5px 10px; border-radius: 5px; cursor: pointer; }
            .protected { background: #f88; text-decoration: line-through; }
            .unprotected { background: #8f8; }
            .control-group { margin: 15px 0; }
        </style>
    </head>
    <body>
        <h1>📅 Timetable Generator (Strict Teacher‑Slot)</h1>
        <p>Upload your config, select <strong>protected subjects</strong> (won't be reduced), set <strong>maximum reduction per unprotected subject</strong> (default 2). The system reduces the fewest lessons needed, never exceeding that max.</p>

        <div class="control-group">
            <input type="file" id="configFile" accept=".json">
            <button onclick="loadConfig()">Load Config</button>
        </div>

        <div id="subjectSelector" style="display:none;">
            <h3>Click subjects to protect them (no reduction)</h3>
            <div id="subjectList" class="subject-list"></div>
            <div class="control-group">
                <label>Maximum lessons reduction <strong>per unprotected subject</strong>: </label>
                <input type="number" id="maxReduction" value="2" min="0" max="10" step="1">
                <small>(System will reduce only as much as needed, up to this limit)</small>
            </div>
            <button onclick="generateTimetable()">📆 Generate Timetable</button>
        </div>

        <h3>Result</h3>
        <pre id="result">Awaiting input...</pre>

        <script>
            let originalConfig = null;
            let allSubjects = [];

            function loadConfig() {
                const file = document.getElementById('configFile').files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = e => {
                    try {
                        originalConfig = JSON.parse(e.target.result);
                        allSubjects = originalConfig.subjects || [];
                        displaySubjectSelector();
                        document.getElementById('result').innerText = "Config loaded. Choose protected subjects and click Generate.";
                    } catch(err) {
                        alert("Invalid JSON: " + err.message);
                    }
                };
                reader.readAsText(file);
            }

            function displaySubjectSelector() {
                const container = document.getElementById('subjectList');
                container.innerHTML = '';
                allSubjects.forEach(sub => {
                    const span = document.createElement('span');
                    span.textContent = sub;
                    span.className = 'subject-item unprotected';
                    span.onclick = () => {
                        if (span.classList.contains('unprotected')) {
                            span.classList.remove('unprotected');
                            span.classList.add('protected');
                        } else {
                            span.classList.remove('protected');
                            span.classList.add('unprotected');
                        }
                    };
                    container.appendChild(span);
                });
                document.getElementById('subjectSelector').style.display = 'block';
            }

            function getProtectedSubjects() {
                const protectedList = [];
                document.querySelectorAll('#subjectList .subject-item.protected').forEach(el => {
                    protectedList.push(el.textContent);
                });
                return protectedList;
            }

            async function generateTimetable() {
                if (!originalConfig) {
                    alert("Load a config first");
                    return;
                }
                const protectedSubjects = getProtectedSubjects();
                const maxReduction = parseInt(document.getElementById('maxReduction').value, 10);
                document.getElementById('result').innerText = "⏳ Solving with auto‑reduction (max " + maxReduction + " per subject)...";
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            config: originalConfig,
                            protectedSubjects: protectedSubjects,
                            maxReductionPerSubject: maxReduction
                        })
                    });
                    const data = await response.json();
                    document.getElementById('result').innerText = JSON.stringify(data, null, 2);
                } catch (err) {
                    document.getElementById('result').innerText = "Network error: " + err.message;
                }
            }
        </script>
    </body>
    </html>
    ''')

@app.route("/generate", methods=["POST"])
def generate():
    try:
        req = request.get_json(force=True)
        if not req or "config" not in req:
            return jsonify({"success": False, "message": "Missing 'config' in request"}), 400
        config = req["config"]
        protected = req.get("protectedSubjects", [])
        max_red = req.get("maxReductionPerSubject", 2)
        if not isinstance(max_red, int) or max_red < 0:
            max_red = 2
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid request: {str(e)}"}), 400

    ok, msg = validate_config(config)
    if not ok:
        return jsonify({"success": False, "message": msg}), 400

    try:
        tt, info = run_solver(config, protected, max_red)
        if tt:
            return jsonify({"success": True, "timetable": tt, "info": info})
        else:
            return jsonify({"success": False, "suggestions": info})
    except Exception as e:
        logger.exception("Solver error")
        return jsonify({"success": False, "message": f"Solver error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)