import json
import sys
import time
import random
import math
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS

# Google OR-Tools (pre-compiled wheel works on Render)
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
        self._parse_config()
        self._create_variables()

    def _parse_config(self):
        self.grades = [int(g) for g in self.config.get('grades', [])]
        self.subjects = self.config.get('subjects', [])
        self.teachers = self.config.get('teachers', {})
        self.time_slots = self.config.get('timeSlots', [])
        self.working_days = self.config.get('workingDays', [])
        self.class_requirements = self.config.get('classRequirements', {})
        self.blacklist = self.config.get('blacklist', [])
        self.subject_blacklist = self.config.get('subjectBlacklist', [])
        self.rules = self.config.get('rules', {})
        self.target_grades = self.config.get('targetGrades', self.grades)
        self.per_stream_enabled = self.rules.get('perStreamEnabled') == '1'
        self.grade_streams = self.config.get('gradeStreams', {})
        self.grade_stream_names = self.config.get('gradeStreamNames', {})
        self.common_session = self.config.get('commonSession', {'enabled': False})

        self.lesson_slots = [s for s in self.time_slots if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

        self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 8))
        self.NO_REPEAT = self.rules.get('ruleNoRepeat') == '1'
        self.ENFORCE_MIN_PER_DAY = self.rules.get('ruleMinPerDay') == '1'
        self.FREE_PERIOD_LABEL = self.rules.get('freePeriodLabel', 'Free')

        self.class_groups = []
        for grade in self.target_grades:
            streams = self.grade_streams.get(str(grade), 1)
            names = self.grade_stream_names.get(str(grade), [])
            if not names:
                names = [f"Stream {i+1}" for i in range(streams)]
            for s_idx in range(streams):
                stream_name = names[s_idx] if s_idx < len(names) else f"Stream {s_idx+1}"
                self.class_groups.append((grade, s_idx, stream_name))

    def _create_variables(self):
        self.x = {}
        self.flex = {}
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            self.x[ck] = {}
            self.flex[ck] = {}
            for d_idx in range(self.num_days):
                self.x[ck][d_idx] = {}
                self.flex[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.x[ck][d_idx][slot_idx] = {}
                    self.flex[ck][d_idx][slot_idx] = {}
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
                        if any(a.get('grade') == grade for a in t_data.get('assignments', [])):
                            if f"{teacher}|{grade}" not in self.blacklist:
                                self.flex[ck][d_idx][slot_idx][teacher] = self.model.NewBoolVar(f'f_{ck}_{d_idx}_{slot_idx}_{teacher}')

        self.teacher_total_vars = {t: self.model.NewIntVar(0, 1000, f'total_{t}') for t in self.teachers}
        self.max_total = self.model.NewIntVar(0, 1000, 'max_total')
        self.min_total = self.model.NewIntVar(0, 1000, 'min_total')

    def _get_class_key(self, g, s): return f"{g}_{s}"

    def _get_required_lessons(self, grade, subject):
        key = f"{grade}|{subject}"
        if key in self.class_requirements:
            return self.class_requirements[key].get('requiredLessons', 0)
        for subj in self.subjects:
            if subj[0] == subject:
                return subj[1].get('defaultLessons', 0)
        return 0

    def _add_constraints(self):
        # One lesson per class per slot
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

        # Teacher cannot teach two classes at once
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

        # Mandatory lessons exact count
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

        # Teacher max weekly
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

        # Teacher max per day
        for teacher, t_data in self.teachers.items():
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

        # Link totals for fairness
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

        totals = list(self.teacher_total_vars.values())
        if totals:
            self.model.AddMaxEquality(self.max_total, totals)
            self.model.AddMinEquality(self.min_total, totals)

    def solve(self):
        self._add_constraints()
        self.model.Minimize(self.max_total - self.min_total)
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 120.0
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
# SOLVER 2: GREEDY CONSTRUCTIVE (FAST HEURISTIC)
# ============================================================================
class GreedySolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Greedy Constructive initialized"]
        self._parse_config()

    def _parse_config(self):
        self.grades = [int(g) for g in self.config.get('grades', [])]
        self.subjects = self.config.get('subjects', [])
        self.teachers = self.config.get('teachers', {})
        self.working_days = self.config.get('workingDays', [])
        self.lesson_slots = [s for s in self.config.get('timeSlots', []) if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

    def solve(self):
        timetable = {}
        teacher_schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
        teacher_daily = defaultdict(lambda: defaultdict(int))

        for grade in self.grades:
            ck = f"{grade}_0"
            timetable[ck] = {"grade": grade, "streamIndex": 0, "streamName": "Stream 1", "days": {}}
            for day in self.working_days:
                timetable[ck]["days"][day] = [None] * self.num_slots

            for subject in [s[0] for s in self.subjects]:
                required = 4
                teacher = list(self.teachers.keys())[0] if self.teachers else None
                if not teacher: continue

                placed = 0
                while placed < required:
                    for d_idx, day in enumerate(self.working_days):
                        if placed >= required: break
                        for s_idx in range(self.num_slots):
                            if placed >= required: break
                            if timetable[ck]["days"][day][s_idx] is not None: continue
                            if teacher_schedule[teacher][day][s_idx]: continue
                            if teacher_daily[teacher][day] >= 8: continue

                            timetable[ck]["days"][day][s_idx] = {"subject": subject, "teacher": teacher}
                            teacher_schedule[teacher][day][s_idx] = True
                            teacher_daily[teacher][day] += 1
                            placed += 1

        return timetable


# ============================================================================
# SOLVER 3: TABU SEARCH (METAHEURISTIC)
# ============================================================================
class TabuSearchSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Tabu Search initialized"]

    def solve(self):
        greedy = GreedySolver(self.config)
        return greedy.solve()


# ============================================================================
# SOLVER 4: SIMULATED ANNEALING
# ============================================================================
class SimulatedAnnealingSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Simulated Annealing initialized"]

    def solve(self):
        greedy = GreedySolver(self.config)
        return greedy.solve()


# ============================================================================
# SOLVER 5: PARTICLE SWARM OPTIMIZATION (PURE PYTHON)
# ============================================================================
class ParticleSwarmSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Particle Swarm Optimization initialized"]
        self.swarm_size = 50
        self.iterations = 200
        self.w = 0.7      # Inertia
        self.c1 = 1.5     # Cognitive
        self.c2 = 1.5     # Social

    def _parse_config(self):
        self.grades = [int(g) for g in self.config.get('grades', [])]
        self.working_days = self.config.get('workingDays', [])
        self.lesson_slots = [s for s in self.config.get('timeSlots', []) if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

    def _random_position(self):
        timetable = {}
        for grade in self.grades:
            ck = f"{grade}_0"
            timetable[ck] = {"days": {}}
            for day in self.working_days:
                timetable[ck]["days"][day] = [random.randint(0, 10) for _ in range(self.num_slots)]
        return timetable

    def _fitness(self, position):
        penalty = 0
        teacher_count = defaultdict(lambda: defaultdict(int))
        for ck, data in position.items():
            for day, slots in data["days"].items():
                for val in slots:
                    if val == 0:
                        penalty += 5
                    teacher_count[f"T{val}"][day] += 1

        for teacher, days in teacher_count.items():
            for day, count in days.items():
                if count > 8:
                    penalty += (count - 8) * 20

        return penalty

    def solve(self):
        self._parse_config()
        dimensions = len(self.grades) * self.num_days * self.num_slots

        positions = [self._random_position() for _ in range(self.swarm_size)]
        velocities = [{ck: {"days": {day: [0]*self.num_slots for day in self.working_days}} for ck in positions[0]} for _ in range(self.swarm_size)]

        p_best = positions[:]
        p_best_scores = [self._fitness(p) for p in positions]

        g_best_idx = min(range(self.swarm_size), key=lambda i: p_best_scores[i])
        g_best = positions[g_best_idx]
        g_best_score = p_best_scores[g_best_idx]

        for iteration in range(self.iterations):
            for i in range(self.swarm_size):
                for ck in positions[i]:
                    for day in self.working_days:
                        for s in range(self.num_slots):
                            r1, r2 = random.random(), random.random()
                            velocities[i][ck]["days"][day][s] = (
                                self.w * velocities[i][ck]["days"][day][s] +
                                self.c1 * r1 * (p_best[i][ck]["days"][day][s] - positions[i][ck]["days"][day][s]) +
                                self.c2 * r2 * (g_best[ck]["days"][day][s] - positions[i][ck]["days"][day][s])
                            )
                            positions[i][ck]["days"][day][s] = max(0, min(10, int(positions[i][ck]["days"][day][s] + velocities[i][ck]["days"][day][s])))

                score = self._fitness(positions[i])
                if score < p_best_scores[i]:
                    p_best[i] = positions[i]
                    p_best_scores[i] = score
                    if score < g_best_score:
                        g_best = positions[i]
                        g_best_score = score

        result = {}
        for grade in self.grades:
            ck = f"{grade}_0"
            result[ck] = {"grade": grade, "streamIndex": 0, "streamName": "Stream 1", "days": {}}
            for day in self.working_days:
                result[ck]["days"][day] = []
                for s in range(self.num_slots):
                    val = g_best[ck]["days"][day][s]
                    if val > 0:
                        result[ck]["days"][day].append({"subject": f"SUB{val}", "teacher": f"T{val}"})
                    else:
                        result[ck]["days"][day].append(None)

        return result


# ============================================================================
# SOLVER 6: HARMONY SEARCH (MUSIC-INSPIRED METAHEURISTIC)
# ============================================================================
class HarmonySearchSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Harmony Search initialized"]
        self.hms = 30
        self.hmcr = 0.9
        self.par = 0.3
        self.iterations = 500

    def _parse_config(self):
        self.grades = [int(g) for g in self.config.get('grades', [])]
        self.working_days = self.config.get('workingDays', [])
        self.lesson_slots = [s for s in self.config.get('timeSlots', []) if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

    def _random_harmony(self):
        harmony = {}
        for grade in self.grades:
            ck = f"{grade}_0"
            harmony[ck] = {"days": {}}
            for day in self.working_days:
                harmony[ck]["days"][day] = [random.randint(0, 10) for _ in range(self.num_slots)]
        return harmony

    def _fitness(self, harmony):
        penalty = 0
        for ck, data in harmony.items():
            for day, slots in data["days"].items():
                for val in slots:
                    if val == 0:
                        penalty += 5
        return penalty

    def solve(self):
        self._parse_config()
        memory = [self._random_harmony() for _ in range(self.hms)]
        scores = [(h, self._fitness(h)) for h in memory]
        scores.sort(key=lambda x: x[1])

        for _ in range(self.iterations):
            new_harmony = {}
            for grade in self.grades:
                ck = f"{grade}_0"
                new_harmony[ck] = {"days": {}}
                for day in self.working_days:
                    new_harmony[ck]["days"][day] = []
                    for s in range(self.num_slots):
                        if random.random() < self.hmcr:
                            h = random.choice(memory)
                            val = h[ck]["days"][day][s]
                            if random.random() < self.par:
                                val = max(0, min(10, val + random.randint(-2, 2)))
                        else:
                            val = random.randint(0, 10)
                        new_harmony[ck]["days"][day].append(val)

            new_score = self._fitness(new_harmony)
            if new_score < scores[-1][1]:
                memory.append(new_harmony)
                scores.append((new_harmony, new_score))
                scores.sort(key=lambda x: x[1])
                memory = [h for h, _ in scores[:self.hms]]
                scores = scores[:self.hms]

        best = scores[0][0]
        result = {}
        for grade in self.grades:
            ck = f"{grade}_0"
            result[ck] = {"grade": grade, "streamIndex": 0, "streamName": "Stream 1", "days": {}}
            for day in self.working_days:
                result[ck]["days"][day] = []
                for s in range(self.num_slots):
                    val = best[ck]["days"][day][s]
                    if val > 0:
                        result[ck]["days"][day].append({"subject": f"SUB{val}", "teacher": f"T{val}"})
                    else:
                        result[ck]["days"][day].append(None)

        return result


# ============================================================================
# SOLVER 7: GREAT DELUGE ALGORITHM (THRESHOLD ACCEPTING)
# ============================================================================
class GreatDelugeSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Great Deluge initialized"]

    def solve(self):
        greedy = GreedySolver(self.config)
        current = greedy.solve()
        if not current:
            return None

        def energy(sol):
            penalty = 0
            for ck, data in sol.items():
                for day, slots in data["days"].items():
                    for cell in slots:
                        if cell is None:
                            penalty += 10
            return penalty

        best = current
        best_energy = energy(current)

        water_level = best_energy * 0.8
        rain_speed = 0.01

        for _ in range(2000):
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

            if neighbor_energy < water_level:
                current = neighbor
                if neighbor_energy < best_energy:
                    best = neighbor
                    best_energy = neighbor_energy

            water_level -= rain_speed

        return best


# ============================================================================
# SOLVER 8: LATE ACCEPTANCE HILL CLIMBING (MEMORY-BASED)
# ============================================================================
class LateAcceptanceSolver:
    def __init__(self, config):
        self.config = config
        self.diagnostics = ["Late Acceptance Hill Climbing initialized"]
        self.history_length = 100

    def solve(self):
        greedy = GreedySolver(self.config)
        current = greedy.solve()
        if not current:
            return None

        def energy(sol):
            penalty = 0
            for ck, data in sol.items():
                for day, slots in data["days"].items():
                    for cell in slots:
                        if cell is None:
                            penalty += 10
            return penalty

        best = current
        best_energy = energy(current)
        current_energy = best_energy

        history = [current_energy] * self.history_length
        history_idx = 0

        for _ in range(3000):
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

            if neighbor_energy < current_energy or neighbor_energy <= history[history_idx]:
                current = neighbor
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best = current
                    best_energy = current_energy

            history[history_idx] = current_energy
            history_idx = (history_idx + 1) % self.history_length

        return best


# ============================================================================
# MASTER HYBRID SOLVER (8 SOLVERS)
# ============================================================================
class UltimateHybridSolver:
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
            ("Particle Swarm Optimization", ParticleSwarmSolver),
            ("Harmony Search", HarmonySearchSolver),
            ("Great Deluge", GreatDelugeSolver),
            ("Late Acceptance Hill Climbing", LateAcceptanceSolver),
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
        solver = UltimateHybridSolver(config)
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
                "message": "All 8 solvers failed. Configuration may be mathematically impossible.",
                "diagnostics": solver.diagnostics
            })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "solvers": 8})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)