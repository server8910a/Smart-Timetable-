import json
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model

app = Flask(__name__)
CORS(app)


# ============================================================================
# KNEC CURRICULUM DATA (Grades 4-9)
# ============================================================================
KNEC_CURRICULUM = {
    "4": {"core": {"MATHEMATICS": 5, "ENGLISH": 5, "KISWAHILI": 4, "SCIENCE AND TECHNOLOGY": 3, "SOCIAL STUDIES": 3, "AGRICULTURE": 2}, "optional": {"CRE": 3, "IRE": 3, "CREATIVE ARTS": 4, "PHYSICAL EDUCATION": 2, "FRENCH": 2, "HOME SCIENCE": 3}},
    "5": {"core": {"MATHEMATICS": 5, "ENGLISH": 5, "KISWAHILI": 4, "SCIENCE AND TECHNOLOGY": 3, "SOCIAL STUDIES": 3, "AGRICULTURE": 2}, "optional": {"CRE": 3, "IRE": 3, "CREATIVE ARTS": 4, "PHYSICAL EDUCATION": 2, "FRENCH": 2, "HOME SCIENCE": 3}},
    "6": {"core": {"MATHEMATICS": 5, "ENGLISH": 5, "KISWAHILI": 4, "SCIENCE AND TECHNOLOGY": 3, "SOCIAL STUDIES": 3, "AGRICULTURE": 2}, "optional": {"CRE": 3, "IRE": 3, "CREATIVE ARTS": 4, "PHYSICAL EDUCATION": 2, "FRENCH": 2, "HOME SCIENCE": 3}},
    "7": {"core": {"MATHEMATICS": 5, "ENGLISH": 5, "KISWAHILI": 4, "INTEGRATED SCIENCE": 4, "SOCIAL STUDIES": 3, "HEALTH EDUCATION": 2, "LIFE SKILLS": 1}, "optional": {"CRE": 3, "IRE": 3, "BUSINESS STUDIES": 3, "AGRICULTURE": 3, "PRE-TECHNICAL STUDIES": 4, "CREATIVE ARTS": 3, "COMPUTER SCIENCE": 3}},
    "8": {"core": {"MATHEMATICS": 5, "ENGLISH": 5, "KISWAHILI": 4, "INTEGRATED SCIENCE": 4, "SOCIAL STUDIES": 3, "HEALTH EDUCATION": 2, "LIFE SKILLS": 1}, "optional": {"CRE": 3, "IRE": 3, "BUSINESS STUDIES": 3, "AGRICULTURE": 3, "PRE-TECHNICAL STUDIES": 4, "CREATIVE ARTS": 3, "COMPUTER SCIENCE": 3}},
    "9": {"core": {"MATHEMATICS": 5, "ENGLISH": 5, "KISWAHILI": 4, "INTEGRATED SCIENCE": 4, "SOCIAL STUDIES": 3, "HEALTH EDUCATION": 2, "LIFE SKILLS": 1}, "optional": {"CRE": 3, "IRE": 3, "BUSINESS STUDIES": 3, "AGRICULTURE": 3, "PRE-TECHNICAL STUDIES": 4, "CREATIVE ARTS": 3, "COMPUTER SCIENCE": 3}}
}


class TimetableSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = None
        self.violations = []
        self.penalty_vars = []

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
        self.FREE_PERIOD_LABEL = self.rules.get('freePeriodLabel', 'Free')

        self.class_groups = []
        for grade in self.target_grades:
            streams = self.grade_streams.get(str(grade), 1)
            names = self.grade_stream_names.get(str(grade), [])
            if not names: names = [f"Stream {i+1}" for i in range(streams)]
            for s_idx in range(streams):
                stream_name = names[s_idx] if s_idx < len(names) else f"Stream {s_idx+1}"
                self.class_groups.append({'grade': grade, 'stream_index': s_idx, 'stream_name': stream_name, 'key': f"{grade}_{s_idx}"})

        self._validate_assignments()
        self.x = {}
        self._create_variables()
        self.teacher_total = {t: self.model.NewIntVar(0, 1000, f'total_{t}') for t in self.teachers}

    def _validate_assignments(self):
        errors = []
        for cg in self.class_groups:
            grade, s_idx = cg['grade'], cg['stream_index']
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0: continue
                if f"{subject}|{grade}" in self.subject_blacklist:
                    errors.append(f"Grade {grade} {cg['stream_name']} - {subject}: Subject blacklisted")
                    continue
                eligible = False
                for teacher, t_data in self.teachers.items():
                    if f"{teacher}|{grade}" in self.blacklist: continue
                    for assign in t_data.get('assignments', []):
                        if int(assign.get('grade', 0)) != grade: continue
                        if assign.get('subject') != subject: continue
                        if self.per_stream_enabled and assign.get('streamIndex') is not None:
                            if int(assign.get('streamIndex', 0)) != s_idx: continue
                        eligible = True
                        break
                    if eligible: break
                if not eligible:
                    errors.append(f"Grade {grade} {cg['stream_name']} - {subject}: No teacher assigned")
        if errors: raise ValueError("\n".join(errors))

    def _get_class_key(self, g, s): return f"{g}_{s}"

    def _get_required_lessons(self, grade, subject):
        total = 0
        for t, t_data in self.teachers.items():
            if f"{t}|{grade}" in self.blacklist: continue
            for a in t_data.get('assignments', []):
                if int(a.get('grade', 0)) == grade and a.get('subject') == subject:
                    total += int(a.get('lessons', 0))
        return total

    def _create_variables(self):
        for cg in self.class_groups:
            ck, grade, s_idx = cg['key'], cg['grade'], cg['stream_index']
            self.x[ck] = {}
            for d_idx in range(self.num_days):
                self.x[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.x[ck][d_idx][slot_idx] = {}
                    for teacher, t_data in self.teachers.items():
                        if f"{teacher}|{grade}" in self.blacklist: continue
                        self.x[ck][d_idx][slot_idx][teacher] = {}
                        for a in t_data.get('assignments', []):
                            if int(a.get('grade', 0)) != grade: continue
                            if self.per_stream_enabled and a.get('streamIndex') is not None:
                                if int(a.get('streamIndex', 0)) != s_idx: continue
                            subject = a.get('subject')
                            if subject not in self.subjects: continue
                            if f"{subject}|{grade}" in self.subject_blacklist: continue
                            self.x[ck][d_idx][slot_idx][teacher][subject] = self.model.NewBoolVar(f'x_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}')

    def _add_hard_constraints(self):
        # 1. Exactly one lesson per slot for each class
        for cg in self.class_groups:
            ck = cg['key']
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    vars_in_slot = []
                    for t in self.x[ck][d][s]:
                        vars_in_slot.extend(self.x[ck][d][s][t].values())
                    if vars_in_slot:
                        self.model.Add(sum(vars_in_slot) == 1)

        # 2. Teacher cannot be in two places at once
        for teacher in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = []
                    for cg in self.class_groups:
                        ck = cg['key']
                        if teacher in self.x[ck][d][s]:
                            tv.extend(self.x[ck][d][s][teacher].values())
                    if tv:
                        self.model.Add(sum(tv) <= 1)

        # 3. Unavailable days
        for teacher, t_data in self.teachers.items():
            unavail = set(t_data.get('unavailDays', []))
            for d_idx, day in enumerate(self.working_days):
                if day in unavail:
                    for cg in self.class_groups:
                        ck = cg['key']
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                for v in self.x[ck][d_idx][s][teacher].values():
                                    self.model.Add(v == 0)

        # 4. Common session
        if self.common_session.get('enabled'):
            day = self.common_session.get('day', 'FRI')
            slot_idx = self.common_session.get('slotIndex', 0)
            if day in self.working_days and slot_idx < self.num_slots:
                d_idx = self.working_days.index(day)
                for cg in self.class_groups:
                    ck = cg['key']
                    common_var = self.model.NewBoolVar(f'common_{ck}_{d_idx}_{slot_idx}')
                    self.model.Add(common_var == 1)

    def _add_soft_constraints(self):
        # Mandatory lesson shortage
        for cg in self.class_groups:
            ck, grade = cg['key'], cg['grade']
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0: continue
                sv = []
                for d_idx in range(self.num_days):
                    for s in range(self.num_slots):
                        for t in self.teachers:
                            if t in self.x[ck][d_idx][s] and subject in self.x[ck][d_idx][s][t]:
                                sv.append(self.x[ck][d_idx][s][t][subject])
                if sv:
                    shortage = self.model.NewIntVar(0, required, f'shortage_{ck}_{subject}')
                    self.model.Add(sum(sv) + shortage == required)
                    self.penalty_vars.append((shortage, 100000, f"Grade {grade} {cg['stream_name']}: Missing {subject}"))

        # Teacher daily overload
        for teacher, t_data in self.teachers.items():
            mpd = t_data.get('maxPerDay') or self.global_max_per_day
            for d_idx in range(self.num_days):
                dv = []
                for cg in self.class_groups:
                    ck = cg['key']
                    if teacher in self.x[ck][d_idx]:
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                dv.extend(self.x[ck][d_idx][s][teacher].values())
                if dv:
                    overload = self.model.NewIntVar(0, len(dv), f'overload_{teacher}_{d_idx}')
                    self.model.Add(sum(dv) <= mpd + overload)
                    self.penalty_vars.append((overload, 10000, f"{teacher}: Exceeded daily max on {self.working_days[d_idx]}"))

        # Teacher weekly overload
        for teacher, t_data in self.teachers.items():
            if t_data.get('maxLessons'):
                wv = []
                for cg in self.class_groups:
                    ck = cg['key']
                    for d_idx in range(self.num_days):
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s]:
                                wv.extend(self.x[ck][d_idx][s][teacher].values())
                if wv:
                    overload = self.model.NewIntVar(0, len(wv), f'week_overload_{teacher}')
                    self.model.Add(sum(wv) <= t_data['maxLessons'] + overload)
                    self.penalty_vars.append((overload, 20000, f"{teacher}: Exceeded weekly max"))

        # Back-to-back same subject
        if self.NO_REPEAT:
            for cg in self.class_groups:
                ck, grade = cg['key'], cg['grade']
                for d_idx in range(self.num_days):
                    for s in range(self.num_slots - 1):
                        for t1 in self.teachers:
                            if t1 not in self.x[ck][d_idx][s]: continue
                            for subj, v1 in self.x[ck][d_idx][s][t1].items():
                                for t2 in self.teachers:
                                    if t2 not in self.x[ck][d_idx][s+1]: continue
                                    if subj in self.x[ck][d_idx][s+1][t2]:
                                        viol = self.model.NewBoolVar(f'btb_{ck}_{d_idx}_{s}_{subj}')
                                        self.model.Add(v1 + self.x[ck][d_idx][s+1][t2][subj] <= 1 + viol)
                                        self.penalty_vars.append((viol, 2000, f"Back-to-back {subj} in Grade {grade} on {self.working_days[d_idx]}"))

        # Link totals
        for teacher in self.teachers:
            wv = []
            for cg in self.class_groups:
                ck = cg['key']
                for d_idx in range(self.num_days):
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d_idx][s]:
                            wv.extend(self.x[ck][d_idx][s][teacher].values())
            if wv:
                self.model.Add(self.teacher_total[teacher] == sum(wv))

    def solve(self):
        self._add_hard_constraints()
        self._add_soft_constraints()

        if len(self.teacher_total) > 1:
            max_v = self.model.NewIntVar(0, 1000, 'max_total')
            min_v = self.model.NewIntVar(0, 1000, 'min_total')
            totals = list(self.teacher_total.values())
            self.model.AddMaxEquality(max_v, totals)
            self.model.AddMinEquality(min_v, totals)
            fairness = max_v - min_v
        else:
            fairness = 0

        total_penalty = sum(v[1] * v[0] for v in self.penalty_vars) if self.penalty_vars else 0
        self.model.Minimize(fairness + total_penalty)

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300.0
        self.solver.parameters.num_search_workers = 8

        status = self.solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution()
        return None

    def _extract_solution(self):
        timetable = {}
        for cg in self.class_groups:
            ck = cg['key']
            timetable[ck] = {"grade": cg['grade'], "streamIndex": cg['stream_index'], "streamName": cg['stream_name'], "days": {}}
            for d_idx, day in enumerate(self.working_days):
                timetable[ck]["days"][day] = []
                for slot_idx in range(self.num_slots):
                    cell = None
                    for teacher, subjects in self.x[ck][d_idx][slot_idx].items():
                        for subject, var in subjects.items():
                            if self.solver.Value(var):
                                cell = {"subject": subject, "teacher": teacher, "grade": cg['grade']}
                                break
                        if cell: break
                    timetable[ck]["days"][day].append(cell)
        return timetable

    def get_violations_summary(self):
        if not self.solver: return []
        return [{"description": d, "severity": "High" if w >= 100000 else "Medium" if w >= 10000 else "Low", "value": self.solver.Value(v)} for v, w, d in self.penalty_vars if self.solver.Value(v) > 0]


@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = TimetableSolver(config)
        sol = solver.solve()
        if sol:
            violations = solver.get_violations_summary()
            return jsonify({"success": True, "timetable": sol, "violations": violations})
        return jsonify({"success": False, "message": "No feasible timetable. Check teacher workloads or increase available slots."})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/ai/curriculum/<grade>', methods=['GET'])
def get_curriculum(grade):
    c = KNEC_CURRICULUM.get(str(grade), {"core": {}, "optional": {}})
    return jsonify({"success": True, "grade": grade, "core": c.get("core", {}), "optional": c.get("optional", {})})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)