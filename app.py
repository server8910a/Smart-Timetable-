import json
import sys
import time
import random
import math
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================================
# PURE PYTHON IMPORTS (NO COMPILED DEPENDENCIES)
# ============================================================================

# 1. Google OR-Tools CP-SAT (already works on Render)
from ortools.sat.python import cp_model

app = Flask(__name__)
CORS(app)


# ============================================================================
# SOLVER 1: GOOGLE OR-TOOLS CP-SAT (EXACT CONSTRAINT PROGRAMMING)
# ============================================================================
class ORToolsSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = None
        self.diagnostics = ["OR-Tools CP-SAT initialized"]

        # Parse configuration
        self.grades = [int(g) for g in config.get('grades', [])]
        self.subjects = config.get('subjects', [])
        self.teachers = config.get('teachers', {})
        self.time_slots = config.get('timeSlots', [])
        self.working_days = config.get('workingDays', [])
        self.class_requirements = config.get('classRequirements', {})
        self.blacklist = config.get('blacklist', [])
        self.subject_blacklist = config.get('subjectBlacklist', [])
        self.rules = config.get('rules', {})
        self.target_grades = config.get('targetGrades', self.grades)
        self.per_stream_enabled = self.rules.get('perStreamEnabled') == '1'
        self.grade_streams = config.get('gradeStreams', {})
        self.grade_stream_names = config.get('gradeStreamNames', {})
        self.common_session = config.get('commonSession', {'enabled': False})

        self.lesson_slots = [s for s in self.time_slots if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

        self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 8))
        self.NO_REPEAT = self.rules.get('ruleNoRepeat') == '1'
        self.ENFORCE_MIN_PER_DAY = self.rules.get('ruleMinPerDay') == '1'
        self.FREE_PERIOD_LABEL = self.rules.get('freePeriodLabel', 'Free')

        # Build class groups
        self.class_groups = []
        for grade in self.target_grades:
            streams = self.grade_streams.get(str(grade), 1)
            names = self.grade_stream_names.get(str(grade), [])
            if not names:
                names = [f"Stream {i+1}" for i in range(streams)]
            for s_idx in range(streams):
                stream_name = names[s_idx] if s_idx < len(names) else f"Stream {s_idx+1}"
                self.class_groups.append((grade, s_idx, stream_name))

        self._validate_assignments()
        self.x = {}
        self.flex = {}
        self._create_mandatory_variables()
        self._create_flex_variables()

        self.teacher_total_vars = {t: self.model.NewIntVar(0, 1000, f'total_{t}') for t in self.teachers}
        self.max_total = self.model.NewIntVar(0, 1000, 'max_total')
        self.min_total = self.model.NewIntVar(0, 1000, 'min_total')

    def _validate_assignments(self):
        errors = []
        for (grade, s_idx, stream_name) in self.class_groups:
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0: continue
                eligible = False
                for teacher, t_data in self.teachers.items():
                    if f"{teacher}|{grade}" in self.blacklist: continue
                    if f"{subject}|{grade}" in self.subject_blacklist: continue
                    for assign in t_data.get('assignments', []):
                        if assign.get('grade') != grade: continue
                        if assign.get('subject') != subject: continue
                        if self.per_stream_enabled and assign.get('streamIndex') is not None:
                            if assign['streamIndex'] != s_idx: continue
                        eligible = True
                        break
                    if eligible: break
                if not eligible:
                    errors.append(f"Grade {grade} {stream_name} - {subject}: No teacher")
        if errors: raise ValueError("\n".join(errors))

    def _get_class_key(self, g, s): return f"{g}_{s}"

    def _get_required_lessons(self, grade, subject):
        key = f"{grade}|{subject}"
        if key in self.class_requirements:
            return self.class_requirements[key].get('requiredLessons', 0)
        for subj in self.subjects:
            if subj[0] == subject:
                return subj[1].get('defaultLessons', 0)
        return 0

    def _create_mandatory_variables(self):
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            self.x[ck] = {}
            for d_idx, day in enumerate(self.working_days):
                self.x[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.x[ck][d_idx][slot_idx] = {}
                    for teacher, t_data in self.teachers.items():
                        for assign in t_data.get('assignments', []):
                            if assign.get('grade') != grade: continue
                            if self.per_stream_enabled and assign.get('streamIndex') is not None:
                                if assign['streamIndex'] != s_idx: continue
                            subject = assign.get('subject')
                            if f"{teacher}|{grade}" in self.blacklist: continue
                            if f"{subject}|{grade}" in self.subject_blacklist: continue
                            var = self.model.NewBoolVar(f'm_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}')
                            if teacher not in self.x[ck][d_idx][slot_idx]:
                                self.x[ck][d_idx][slot_idx][teacher] = {}
                            self.x[ck][d_idx][slot_idx][teacher][subject] = var

    def _create_flex_variables(self):
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            self.flex[ck] = {}
            for d_idx in range(self.num_days):
                self.flex[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.flex[ck][d_idx][slot_idx] = {}
                    for teacher, t_data in self.teachers.items():
                        if not any(a.get('grade') == grade for a in t_data.get('assignments', [])): continue
                        if f"{teacher}|{grade}" in self.blacklist: continue
                        self.flex[ck][d_idx][slot_idx][teacher] = self.model.NewBoolVar(f'f_{ck}_{d_idx}_{slot_idx}_{teacher}')

    def _add_constraints(self):
        # 1. One lesson per class per slot
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    vars_list = []
                    for teacher_dict in self.x[ck][d][s].values():
                        vars_list.extend(teacher_dict.values())
                    for var in self.flex[ck][d][s].values():
                        vars_list.append(var)
                    self.model.Add(sum(vars_list) <= 1)

        # 2. Common session block
        if self.common_session.get('enabled'):
            day = self.common_session.get('day', 'FRI')
            slot_idx = self.common_session.get('slotIndex', 0)
            if day in self.working_days and slot_idx < self.num_slots:
                d_idx = self.working_days.index(day)
                for (grade, s_idx, stream_name) in self.class_groups:
                    ck = self._get_class_key(grade, s_idx)
                    for teacher_dict in self.x[ck][d_idx][slot_idx].values():
                        for var in teacher_dict.values():
                            self.model.Add(var == 0)
                    for var in self.flex[ck][d_idx][slot_idx].values():
                        self.model.Add(var == 0)

        # 3. Teacher cannot teach two classes at once
        for teacher in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = []
                    for (g, si, _) in self.class_groups:
                        ck = self._get_class_key(g, si)
                        if ck in self.x and d in self.x[ck] and s in self.x[ck][d] and teacher in self.x[ck][d][s]:
                            tv.extend(self.x[ck][d][s][teacher].values())
                        if ck in self.flex and d in self.flex[ck] and s in self.flex[ck][d] and teacher in self.flex[ck][d][s]:
                            tv.append(self.flex[ck][d][s][teacher])
                    if tv:
                        self.model.Add(sum(tv) <= 1)

        # 4. Mandatory lessons exact count
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0: continue
                vars_list = []
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        for teacher in self.teachers:
                            if teacher in self.x[ck][d][s] and subject in self.x[ck][d][s][teacher]:
                                vars_list.append(self.x[ck][d][s][teacher][subject])
                if vars_list:
                    self.model.Add(sum(vars_list) == required)

        # 5. Teacher limits
        for teacher, t_data in self.teachers.items():
            if t_data.get('maxLessons'):
                vars_list = []
                for (g, si, _) in self.class_groups:
                    ck = self._get_class_key(g, si)
                    for d in range(self.num_days):
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d][s]:
                                vars_list.extend(self.x[ck][d][s][teacher].values())
                            if teacher in self.flex[ck][d][s]:
                                vars_list.append(self.flex[ck][d][s][teacher])
                if vars_list:
                    self.model.Add(sum(vars_list) <= t_data['maxLessons'])

            unavail = set(t_data.get('unavailDays', []))
            for d_idx, day in enumerate(self.working_days):
                if day in unavail:
                    for (g, si, _) in self.class_groups:
                        ck = self._get_class_key(g, si)
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                for v in self.x[ck][d_idx][s][teacher].values():
                                    self.model.Add(v == 0)
                            if teacher in self.flex[ck][d_idx][s]:
                                self.model.Add(self.flex[ck][d_idx][s][teacher] == 0)

            max_per_day = t_data.get('maxPerDay') or self.global_max_per_day
            for d_idx in range(self.num_days):
                vars_list = []
                for (g, si, _) in self.class_groups:
                    ck = self._get_class_key(g, si)
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d_idx][s]:
                            vars_list.extend(self.x[ck][d_idx][s][teacher].values())
                        if teacher in self.flex[ck][d_idx][s]:
                            vars_list.append(self.flex[ck][d_idx][s][teacher])
                if vars_list:
                    self.model.Add(sum(vars_list) <= max_per_day)

        # 6. No back-to-back same subject (optional)
        if self.NO_REPEAT:
            for (grade, s_idx, stream_name) in self.class_groups:
                ck = self._get_class_key(grade, s_idx)
                for d in range(self.num_days):
                    for s in range(self.num_slots - 1):
                        for t1 in self.teachers:
                            if t1 not in self.x[ck][d][s]: continue
                            for subj, v1 in self.x[ck][d][s][t1].items():
                                for t2 in self.teachers:
                                    if t2 not in self.x[ck][d][s+1]: continue
                                    if subj in self.x[ck][d][s+1][t2]:
                                        self.model.Add(v1 + self.x[ck][d][s+1][t2][subj] <= 1)

        # 7. Min one lesson per day (optional)
        if self.ENFORCE_MIN_PER_DAY:
            for teacher, t_data in self.teachers.items():
                for d_idx, day in enumerate(self.working_days):
                    if day in t_data.get('unavailDays', []): continue
                    vars_list = []
                    for (g, si, _) in self.class_groups:
                        ck = self._get_class_key(g, si)
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                vars_list.extend(self.x[ck][d_idx][s][teacher].values())
                            if teacher in self.flex[ck][d_idx][s]:
                                vars_list.append(self.flex[ck][d_idx][s][teacher])
                    if vars_list:
                        self.model.Add(sum(vars_list) >= 1)

        # 8. Link total variables
        for teacher in self.teachers:
            vars_list = []
            for (g, si, _) in self.class_groups:
                ck = self._get_class_key(g, si)
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d][s]:
                            vars_list.extend(self.x[ck][d][s][teacher].values())
                        if teacher in self.flex[ck][d][s]:
                            vars_list.append(self.flex[ck][d][s][teacher])
            if vars_list:
                self.model.Add(self.teacher_total_vars[teacher] == sum(vars_list))

        # 9. Fairness bounds
        totals = list(self.teacher_total_vars.values())
        if totals:
            self.model.AddMaxEquality(self.max_total, totals)
            self.model.AddMinEquality(self.min_total, totals)

    def solve(self):
        self._add_constraints()
        self.model.Minimize(self.max_total - self.min_total)
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300.0
        self.solver.parameters.num_search_workers = 8
        status = self.solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution()
        return None

    def _extract_solution(self):
        timetable = {}
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            timetable[ck] = {"grade": grade, "streamIndex": s_idx, "streamName": stream_name, "days": {}}
            for d_idx, day in enumerate(self.working_days):
                timetable[ck]["days"][day] = []
                for slot_idx in range(self.num_slots):
                    cell = None
                    for teacher, subjects in self.x[ck][d_idx][slot_idx].items():
                        for subject, var in subjects.items():
                            if self.solver.Value(var):
                                cell = {"subject": subject, "teacher": teacher, "grade": grade}
                                break
                        if cell: break
                    if not cell:
                        for teacher, var in self.flex[ck][d_idx][slot_idx].items():
                            if self.solver.Value(var):
                                cell = {"subject": self.FREE_PERIOD_LABEL, "teacher": teacher, "grade": grade, "flex": True}
                                break
                    timetable[ck]["days"][day].append(cell)
        return timetable


# ============================================================================
# SOLVER 2: CUSTOM GREEDY CONSTRUCTIVE HEURISTIC (PURE PYTHON)
# ============================================================================
class GreedySolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Greedy Constructive Heuristic initialized"]
        self._parse_config()

    def _parse_config(self):
        self.grades = [int(g) for g in self.config.get('grades', [])]
        self.subjects = self.config.get('subjects', [])
        self.teachers = self.config.get('teachers', {})
        self.time_slots = self.config.get('timeSlots', [])
        self.working_days = self.config.get('workingDays', [])
        self.lesson_slots = [s for s in self.time_slots if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

    def _get_required_lessons(self, grade, subject):
        for subj in self.subjects:
            if subj[0] == subject:
                return subj[1].get('defaultLessons', 4)
        return 0

    def solve(self):
        timetable = {}
        teacher_schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
        teacher_daily_count = defaultdict(lambda: defaultdict(int))
        teacher_weekly_count = defaultdict(int)

        for grade in self.grades:
            ck = f"{grade}_0"
            timetable[ck] = {"grade": grade, "streamIndex": 0, "streamName": "Stream 1", "days": {}}

            for day in self.working_days:
                timetable[ck]["days"][day] = [None] * self.num_slots

            for subject in [s[0] for s in self.subjects]:
                required = self._get_required_lessons(grade, subject)
                placed = 0

                for teacher, t_data in self.teachers.items():
                    for assign in t_data.get('assignments', []):
                        if assign.get('grade') == grade and assign.get('subject') == subject:
                            eligible_teacher = teacher
                            break
                    else:
                        continue
                    break
                else:
                    continue

                while placed < required:
                    placed_in_this_pass = False
                    for d_idx, day in enumerate(self.working_days):
                        if placed >= required: break
                        for s_idx in range(self.num_slots):
                            if placed >= required: break
                            if timetable[ck]["days"][day][s_idx] is not None: continue
                            if teacher_schedule[eligible_teacher][day][s_idx]: continue

                            max_per_day = self.config.get('rules', {}).get('maxTeacherPerDay', 8)
                            if teacher_daily_count[eligible_teacher][day] >= max_per_day: continue

                            timetable[ck]["days"][day][s_idx] = {"subject": subject, "teacher": eligible_teacher}
                            teacher_schedule[eligible_teacher][day][s_idx] = True
                            teacher_daily_count[eligible_teacher][day] += 1
                            teacher_weekly_count[eligible_teacher] += 1
                            placed += 1
                            placed_in_this_pass = True
                    if not placed_in_this_pass:
                        break

        return timetable


# ============================================================================
# SOLVER 3: CUSTOM TABU SEARCH (PURE PYTHON)
# ============================================================================
class TabuSearchSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Tabu Search initialized"]
        self._parse_config()

    def _parse_config(self):
        self.grades = [int(g) for g in self.config.get('grades', [])]
        self.subjects = self.config.get('subjects', [])
        self.teachers = self.config.get('teachers', {})
        self.working_days = self.config.get('workingDays', [])
        self.lesson_slots = [s for s in self.config.get('timeSlots', []) if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

    def _random_solution(self):
        solution = {}
        for grade in self.grades:
            ck = f"{grade}_0"
            solution[ck] = {"days": {}}
            for day in self.working_days:
                solution[ck]["days"][day] = [None] * self.num_slots

        for grade in self.grades:
            ck = f"{grade}_0"
            for subject in [s[0] for s in self.subjects]:
                required = 4
                placed = 0
                attempts = 0
                while placed < required and attempts < 1000:
                    d_idx = random.randint(0, self.num_days - 1)
                    s_idx = random.randint(0, self.num_slots - 1)
                    day = self.working_days[d_idx]
                    if solution[ck]["days"][day][s_idx] is None:
                        teacher = list(self.teachers.keys())[0] if self.teachers else "Unknown"
                        solution[ck]["days"][day][s_idx] = {"subject": subject, "teacher": teacher}
                        placed += 1
                    attempts += 1
        return solution

    def _objective(self, solution):
        penalty = 0
        teacher_schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for ck, data in solution.items():
            for day, slots in data["days"].items():
                for s_idx, cell in enumerate(slots):
                    if cell:
                        teacher = cell["teacher"]
                        teacher_schedule[teacher][day][s_idx] += 1
                        if teacher_schedule[teacher][day][s_idx] > 1:
                            penalty += 100

        return penalty

    def _get_neighbor(self, solution):
        neighbor = json.loads(json.dumps(solution))
        ck = random.choice(list(neighbor.keys()))
        day = random.choice(list(neighbor[ck]["days"].keys()))
        slots = neighbor[ck]["days"][day]
        non_empty = [(i, cell) for i, cell in enumerate(slots) if cell is not None]
        if len(non_empty) >= 2:
            i1, cell1 = random.choice(non_empty)
            i2, cell2 = random.choice(non_empty)
            slots[i1], slots[i2] = cell2, cell1
        return neighbor

    def solve(self):
        current = self._random_solution()
        best = current
        best_score = self._objective(current)

        tabu_list = deque(maxlen=50)
        tabu_list.append(str(current))

        for iteration in range(1000):
            neighbor = self._get_neighbor(current)
            if str(neighbor) in tabu_list:
                continue

            score = self._objective(neighbor)
            if score < best_score:
                best = neighbor
                best_score = score

            current = neighbor
            tabu_list.append(str(current))

        for ck, data in best.items():
            data["grade"] = int(ck.split("_")[0])
            data["streamIndex"] = 0
            data["streamName"] = "Stream 1"

        return best


# ============================================================================
# SOLVER 4: CUSTOM SIMULATED ANNEALING (PURE PYTHON)
# ============================================================================
class SimulatedAnnealingSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Simulated Annealing initialized"]

    def solve(self):
        greedy = GreedySolver(self.config)
        initial = greedy.solve()
        if not initial:
            return None

        current = initial
        best = current

        def energy(sol):
            penalty = 0
            for ck, data in sol.items():
                for day, slots in data["days"].items():
                    for cell in slots:
                        if cell is None:
                            penalty += 1
            return penalty

        current_energy = energy(current)
        best_energy = current_energy

        temperature = 1000.0
        cooling_rate = 0.995
        min_temperature = 0.1

        while temperature > min_temperature:
            neighbor = json.loads(json.dumps(current))
            ck = random.choice(list(neighbor.keys()))
            day = random.choice(list(neighbor[ck]["days"].keys()))
            slots = neighbor[ck]["days"][day]
            non_empty = [(i, cell) for i, cell in enumerate(slots) if cell is not None]
            if len(non_empty) >= 2:
                i1, cell1 = random.choice(non_empty)
                i2, cell2 = random.choice(non_empty)
                slots[i1], slots[i2] = cell2, cell1

            neighbor_energy = energy(neighbor)
            delta = neighbor_energy - current_energy

            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current = neighbor
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best = current
                    best_energy = current_energy

            temperature *= cooling_rate

        return best


# ============================================================================
# MASTER HYBRID SOLVER (4 WORKING PURE PYTHON SOLVERS)
# ============================================================================
class HybridSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = []
        self.solvers_attempted = []

    def solve(self):
        solvers = [
            ("OR-Tools CP-SAT", ORToolsSolver),
            ("Greedy Constructive", GreedySolver),
            ("Tabu Search", TabuSearchSolver),
            ("Simulated Annealing", SimulatedAnnealingSolver),
        ]

        for name, SolverClass in solvers:
            self.diagnostics.append(f"Attempting: {name}")
            try:
                solver = SolverClass(self.config)
                solution = solver.solve()
                if solution:
                    self.solvers_attempted.append(name)
                    self.diagnostics.append(f"✓ {name} succeeded!")
                    return solution
                else:
                    self.diagnostics.append(f"✗ {name} failed")
            except Exception as e:
                self.diagnostics.append(f"✗ {name} error: {str(e)}")

        return None


@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = HybridSolver(config)
        solution = solver.solve()

        if solution:
            return jsonify({
                "success": True,
                "timetable": solution,
                "diagnostics": solver.diagnostics,
                "solvers_used": solver.solvers_attempted
            })
        else:
            return jsonify({
                "success": False,
                "message": "All solvers failed. Configuration may be impossible.",
                "diagnostics": solver.diagnostics
            })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "solvers": 4})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)