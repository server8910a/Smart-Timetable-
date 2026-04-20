import json
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model

# Optional AI integration (Google Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure Gemini if API key is set
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-2.0-flash')


class TimetableSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = None
        self.diagnostics = []
        self.violations = []

        # Parse configuration
        self.grades = [int(g) for g in config.get('grades', [])]
        self.subjects = [s[0] for s in config.get('subjects', [])]
        self.teachers = config.get('teachers', {})
        self.time_slots = config.get('timeSlots', [])
        self.working_days = config.get('workingDays', [])
        self.class_requirements = config.get('classRequirements', {})
        self.blacklist = set(config.get('blacklist', []))
        self.subject_blacklist = set(config.get('subjectBlacklist', []))
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
                self.class_groups.append({
                    'grade': grade, 'stream_index': s_idx, 'stream_name': stream_name,
                    'key': f"{grade}_{s_idx}"
                })

        self._validate_assignments()

        # Create variables
        self.x = {}
        self.free = {}
        self._create_variables()

        # Teacher total variables
        self.teacher_total = {t: self.model.NewIntVar(0, 1000, f'total_{t}') for t in self.teachers}

        # Penalties
        self.penalties = []

    def _validate_assignments(self):
        errors = []
        for cg in self.class_groups:
            grade = cg['grade']
            s_idx = cg['stream_index']
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0:
                    continue
                # Check blacklists
                if f"{subject}|{grade}" in self.subject_blacklist:
                    errors.append(f"Grade {grade} {cg['stream_name']} - {subject}: Subject is blacklisted for this grade.")
                    continue
                eligible = False
                for teacher, t_data in self.teachers.items():
                    if f"{teacher}|{grade}" in self.blacklist:
                        continue
                    for assign in t_data.get('assignments', []):
                        if int(assign.get('grade', 0)) != int(grade):
                            continue
                        if assign.get('subject') != subject:
                            continue
                        if self.per_stream_enabled and assign.get('streamIndex') is not None:
                            if int(assign.get('streamIndex', 0)) != s_idx:
                                continue
                        eligible = True
                        break
                    if eligible:
                        break
                if not eligible:
                    errors.append(f"Grade {grade} {cg['stream_name']} - {subject}: No eligible teacher assigned (or all are blacklisted).")
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))

    def _get_class_key(self, g, s):
        return f"{g}_{s}"

    def _get_required_lessons(self, grade, subject):
        total = 0
        for teacher, t_data in self.teachers.items():
            if f"{teacher}|{grade}" in self.blacklist:
                continue
            for assign in t_data.get('assignments', []):
                if int(assign.get('grade', 0)) != int(grade):
                    continue
                if assign.get('subject') != subject:
                    continue
                total += int(assign.get('lessons', 0))
        return total

    def _create_variables(self):
        for cg in self.class_groups:
            ck = cg['key']
            grade = cg['grade']
            s_idx = cg['stream_index']
            self.x[ck] = {}
            self.free[ck] = {}
            for d_idx in range(self.num_days):
                self.x[ck][d_idx] = {}
                self.free[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.x[ck][d_idx][slot_idx] = {}
                    self.free[ck][d_idx][slot_idx] = self.model.NewBoolVar(f'free_{ck}_{d_idx}_{slot_idx}')
                    for teacher, t_data in self.teachers.items():
                        if f"{teacher}|{grade}" in self.blacklist:
                            continue
                        self.x[ck][d_idx][slot_idx][teacher] = {}
                        for assign in t_data.get('assignments', []):
                            if int(assign.get('grade', 0)) != int(grade):
                                continue
                            if self.per_stream_enabled and assign.get('streamIndex') is not None:
                                if int(assign.get('streamIndex', 0)) != s_idx:
                                    continue
                            subject = assign.get('subject')
                            if subject not in self.subjects:
                                continue
                            if f"{subject}|{grade}" in self.subject_blacklist:
                                continue
                            var = self.model.NewBoolVar(f'x_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}')
                            self.x[ck][d_idx][slot_idx][teacher][subject] = var

    def _add_hard_constraints(self):
        # 1. EXACTLY ONE THING PER SLOT
        for cg in self.class_groups:
            ck = cg['key']
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    vars_in_slot = [self.free[ck][d][s]]
                    for teacher in self.x[ck][d][s]:
                        vars_in_slot.extend(self.x[ck][d][s][teacher].values())
                    self.model.Add(sum(vars_in_slot) == 1)

        # 2. Teacher cannot be in two places at once
        for teacher in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    teacher_vars = []
                    for cg in self.class_groups:
                        ck = cg['key']
                        if teacher in self.x[ck][d][s]:
                            teacher_vars.extend(self.x[ck][d][s][teacher].values())
                    if teacher_vars:
                        self.model.Add(sum(teacher_vars) <= 1)

        # 3. Unavailable days (HARD)
        for teacher, t_data in self.teachers.items():
            unavail = set(t_data.get('unavailDays', []))
            for d_idx, day in enumerate(self.working_days):
                if day in unavail:
                    for cg in self.class_groups:
                        ck = cg['key']
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                for var in self.x[ck][d_idx][s][teacher].values():
                                    self.model.Add(var == 0)

        # 4. Common session (HARD)
        if self.common_session.get('enabled'):
            day = self.common_session.get('day', 'FRI')
            slot_idx = self.common_session.get('slotIndex', 0)
            if day in self.working_days and slot_idx < self.num_slots:
                d_idx = self.working_days.index(day)
                for cg in self.class_groups:
                    ck = cg['key']
                    self.model.Add(self.free[ck][d_idx][slot_idx] == 1)

    def _add_soft_constraints(self):
        # SOFT: Mandatory lesson count
        for cg in self.class_groups:
            ck = cg['key']
            grade = cg['grade']
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0:
                    continue
                subject_vars = []
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        for teacher in self.x[ck][d][s]:
                            if subject in self.x[ck][d][s][teacher]:
                                subject_vars.append(self.x[ck][d][s][teacher][subject])
                if subject_vars:
                    shortage = self.model.NewIntVar(0, required, f'shortage_{ck}_{subject}')
                    self.model.Add(sum(subject_vars) + shortage == required)
                    self.penalties.append(shortage * 100000)
                    self.violations.append(f"Grade {grade} {subject}: missing lessons")

        # SOFT: Teacher max per day
        for teacher, t_data in self.teachers.items():
            max_per_day = t_data.get('maxPerDay') or self.global_max_per_day
            for d in range(self.num_days):
                day_vars = []
                for cg in self.class_groups:
                    ck = cg['key']
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d][s]:
                            day_vars.extend(self.x[ck][d][s][teacher].values())
                if day_vars:
                    overload = self.model.NewIntVar(0, len(day_vars), f'overload_{teacher}_{d}')
                    self.model.Add(sum(day_vars) <= max_per_day + overload)
                    self.penalties.append(overload * 10000)
                    self.violations.append(f"{teacher} daily overload")

        # SOFT: Teacher max weekly
        for teacher, t_data in self.teachers.items():
            if t_data.get('maxLessons'):
                week_vars = []
                for cg in self.class_groups:
                    ck = cg['key']
                    for d in range(self.num_days):
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d][s]:
                                week_vars.extend(self.x[ck][d][s][teacher].values())
                if week_vars:
                    overload = self.model.NewIntVar(0, len(week_vars), f'week_overload_{teacher}')
                    self.model.Add(sum(week_vars) <= t_data['maxLessons'] + overload)
                    self.penalties.append(overload * 20000)
                    self.violations.append(f"{teacher} weekly overload")

        # SOFT: No back-to-back same subject
        if self.NO_REPEAT:
            for cg in self.class_groups:
                ck = cg['key']
                for d in range(self.num_days):
                    for s in range(self.num_slots - 1):
                        for t1 in self.teachers:
                            if t1 not in self.x[ck][d][s]:
                                continue
                            for subj, v1 in self.x[ck][d][s][t1].items():
                                for t2 in self.teachers:
                                    if t2 not in self.x[ck][d][s+1]:
                                        continue
                                    if subj in self.x[ck][d][s+1][t2]:
                                        violation = self.model.NewBoolVar(f'btb_{ck}_{d}_{s}_{subj}')
                                        self.model.Add(v1 + self.x[ck][d][s+1][t2][subj] <= 1 + violation)
                                        self.penalties.append(violation * 2000)
                                        self.violations.append(f"Back-to-back {subj}")

        # Link totals
        for teacher in self.teachers:
            week_vars = []
            for cg in self.class_groups:
                ck = cg['key']
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d][s]:
                            week_vars.extend(self.x[ck][d][s][teacher].values())
            if week_vars:
                self.model.Add(self.teacher_total[teacher] == sum(week_vars))

    def solve(self):
        self._add_hard_constraints()
        self._add_soft_constraints()

        # Objective: minimize fairness gap + penalties
        if len(self.teacher_total) > 1:
            max_var = self.model.NewIntVar(0, 1000, 'max_total')
            min_var = self.model.NewIntVar(0, 1000, 'min_total')
            totals = list(self.teacher_total.values())
            self.model.AddMaxEquality(max_var, totals)
            self.model.AddMinEquality(min_var, totals)
            fairness_penalty = max_var - min_var
        else:
            fairness_penalty = 0

        total_penalty = sum(self.penalties) if self.penalties else 0
        self.model.Minimize(fairness_penalty + total_penalty)

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300.0
        self.solver.parameters.num_search_workers = 8

        status = self.solver.Solve(self.model)
        print(f"[INFO] Solver status: {self.solver.StatusName(status)}", file=sys.stderr)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution()
        return None

    def _extract_solution(self):
        timetable = {}
        for cg in self.class_groups:
            ck = cg['key']
            timetable[ck] = {
                "grade": cg['grade'],
                "streamIndex": cg['stream_index'],
                "streamName": cg['stream_name'],
                "days": {}
            }
            for d_idx, day in enumerate(self.working_days):
                timetable[ck]["days"][day] = []
                for slot_idx in range(self.num_slots):
                    if self.solver.Value(self.free[ck][d_idx][slot_idx]):
                        timetable[ck]["days"][day].append(None)
                    else:
                        cell = None
                        for teacher, subjects in self.x[ck][d_idx][slot_idx].items():
                            for subject, var in subjects.items():
                                if self.solver.Value(var):
                                    cell = {"subject": subject, "teacher": teacher, "grade": cg['grade']}
                                    break
                            if cell:
                                break
                        timetable[ck]["days"][day].append(cell)
        return timetable

    def get_ai_explanation(self):
        """Generate human-readable explanation using Gemini."""
        if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
            return "AI explanation unavailable (Gemini not configured)."

        summary = {
            "grades": len(self.grades),
            "subjects": len(self.subjects),
            "teachers": len(self.teachers),
            "total_required_lessons": sum(self._get_required_lessons(g, s) for g in self.grades for s in self.subjects),
            "total_available_slots": len(self.class_groups) * self.num_days * self.num_slots,
            "teacher_workloads": {t: self.teacher_required[t] for t in self.teachers},
            "blacklisted_combinations": len(self.blacklist) + len(self.subject_blacklist),
            "violations_considered": list(set(self.violations))[:5]
        }

        prompt = f"""
You are an expert school timetabling consultant. A user tried to generate a timetable with the following configuration, but it was infeasible (or had to break rules). Explain in simple terms why it's difficult and suggest specific actions to fix it.

Configuration:
- Grades: {summary['grades']}
- Subjects: {summary['subjects']}
- Teachers: {summary['teachers']}
- Total required lessons: {summary['total_required_lessons']}
- Total available slots (all classes): {summary['total_available_slots']}
- Teacher workloads (lessons/week): {summary['teacher_workloads']}
- Blacklisted combinations: {summary['blacklisted_combinations']}
- Potential violations that had to be penalized: {summary['violations_considered']}

Provide a concise, helpful response (max 200 words) that:
1. Identifies the main bottleneck (e.g., overloaded teachers, not enough slots).
2. Suggests 2-3 actionable fixes (e.g., reduce lessons for specific subjects, add another teacher, increase days).
3. Mentions if any blacklists are causing issues.
"""
        try:
            response = ai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI explanation failed: {str(e)}"


@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = TimetableSolver(config)
        solution = solver.solve()

        if solution:
            return jsonify({"success": True, "timetable": solution})
        else:
            ai_msg = solver.get_ai_explanation() if GEMINI_AVAILABLE else None
            return jsonify({
                "success": False,
                "message": "No feasible timetable found.",
                "ai_explanation": ai_msg,
                "violations": solver.violations
            })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)})
    except Exception as e:
        return jsonify({"success": False, "message": "Internal error: " + str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)