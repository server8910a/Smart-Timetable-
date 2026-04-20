import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model

app = Flask(__name__)
CORS(app)

class TimetableSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = None

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

        self.lesson_slots = [s for s in self.time_slots if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)

        self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 7))
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
                if not any(not (f"{t}|{grade}" in self.blacklist or f"{subject}|{grade}" in self.subject_blacklist) and any(a.get('grade')==grade and a.get('subject')==subject and (not self.per_stream_enabled or a.get('streamIndex') is None or a['streamIndex']==s_idx) for a in t_data.get('assignments',[])) for t, t_data in self.teachers.items()):
                    errors.append(f"Grade {grade} {stream_name} - {subject}: No eligible teacher")
        if errors: raise ValueError("\n".join(errors))

    def _get_class_key(self, g, s): return f"{g}_{s}"
    def _get_required_lessons(self, grade, subject):
        key = f"{grade}|{subject}"
        if key in self.class_requirements: return self.class_requirements[key].get('requiredLessons', 0)
        for subj in self.subjects:
            if subj[0] == subject: return subj[1].get('defaultLessons', 0)
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
                            if self.per_stream_enabled and assign.get('streamIndex') is not None and assign['streamIndex'] != s_idx: continue
                            subject = assign.get('subject')
                            if f"{teacher}|{grade}" in self.blacklist or f"{subject}|{grade}" in self.subject_blacklist: continue
                            var = self.model.NewBoolVar(f'm_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}')
                            self.x[ck][d_idx][slot_idx].setdefault(teacher, {})[subject] = var

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

    def add_constraints(self):
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    self.model.Add(sum(v for d in self.x[ck][d][s].values() for v in d.values()) + sum(self.flex[ck][d][s].values()) <= 1)
        for teacher in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    self.model.Add(sum(v for (g, si, _) in self.class_groups for v in (self.x.get(self._get_class_key(g,si),{}).get(d,{}).get(s,{}).get(teacher,{}).values() if teacher in self.x.get(self._get_class_key(g,si),{}).get(d,{}).get(s,{})) ) + sum(self.flex.get(self._get_class_key(g,si),{}).get(d,{}).get(s,{}).get(teacher,0) for (g,si,_) in self.class_groups) <= 1)
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0: continue
                vars = []
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        for teacher in self.teachers:
                            if teacher in self.x[ck][d][s] and subject in self.x[ck][d][s][teacher]:
                                vars.append(self.x[ck][d][s][teacher][subject])
                if vars: self.model.Add(sum(vars) == required)
        for teacher, t_data in self.teachers.items():
            if t_data.get('maxLessons'):
                vars = []
                for (g, si, _) in self.class_groups:
                    ck = self._get_class_key(g, si)
                    for d in range(self.num_days):
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d][s]: vars.extend(self.x[ck][d][s][teacher].values())
                            if teacher in self.flex[ck][d][s]: vars.append(self.flex[ck][d][s][teacher])
                if vars: self.model.Add(sum(vars) <= t_data['maxLessons'])
            unavail = set(t_data.get('unavailDays', []))
            for d_idx, day in enumerate(self.working_days):
                if day in unavail:
                    for (g, si, _) in self.class_groups:
                        ck = self._get_class_key(g, si)
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                for v in self.x[ck][d_idx][s][teacher].values(): self.model.Add(v == 0)
                            if teacher in self.flex[ck][d_idx][s]: self.model.Add(self.flex[ck][d_idx][s][teacher] == 0)
            max_per_day = t_data.get('maxPerDay') or self.global_max_per_day
            for d_idx in range(self.num_days):
                vars = []
                for (g, si, _) in self.class_groups:
                    ck = self._get_class_key(g, si)
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d_idx][s]: vars.extend(self.x[ck][d_idx][s][teacher].values())
                        if teacher in self.flex[ck][d_idx][s]: vars.append(self.flex[ck][d_idx][s][teacher])
                if vars: self.model.Add(sum(vars) <= max_per_day)
        # Fairness objective
        for teacher in self.teachers:
            vars = []
            for (g, si, _) in self.class_groups:
                ck = self._get_class_key(g, si)
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d][s]: vars.extend(self.x[ck][d][s][teacher].values())
                        if teacher in self.flex[ck][d][s]: vars.append(self.flex[ck][d][s][teacher])
            if vars: self.model.Add(self.teacher_total_vars[teacher] == sum(vars))
        totals = list(self.teacher_total_vars.values())
        if totals:
            self.model.AddMaxEquality(self.max_total, totals)
            self.model.AddMinEquality(self.min_total, totals)

    def solve(self):
        self.add_constraints()
        self.model.Minimize(self.max_total - self.min_total)
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 60.0
        if self.solver.Solve(self.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution()
        return None

    def _extract_solution(self):
        timetable = {}
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            timetable[ck] = {"grade": grade, "streamIndex": s_idx, "streamName": stream_name, "days": {}}
            for d_idx, day in enumerate(self.working_days):
                timetable[ck]["days"][day] = []
                for s_idx in range(self.num_slots):
                    cell = None
                    for teacher, subjects in self.x[ck][d_idx][s_idx].items():
                        for subject, var in subjects.items():
                            if self.solver.Value(var):
                                cell = {"subject": subject, "teacher": teacher, "grade": grade}
                                break
                        if cell: break
                    if not cell:
                        for teacher, var in self.flex[ck][d_idx][s_idx].items():
                            if self.solver.Value(var):
                                cell = {"subject": self.FREE_PERIOD_LABEL, "teacher": teacher, "grade": grade}
                                break
                    timetable[ck]["days"][day].append(cell)
        return timetable

@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = TimetableSolver(config)
        solution = solver.solve()
        if solution: return jsonify({"success": True, "timetable": solution})
        return jsonify({"success": False, "message": "No feasible timetable found."})
    except ValueError as e: return jsonify({"success": False, "message": str(e)})
    except Exception as e: return jsonify({"success": False, "message": "Internal error: " + str(e)})

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)