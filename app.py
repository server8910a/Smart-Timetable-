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

        # Extract data
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

        # Global fallback values
        self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 7))
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

        # Validate mandatory assignments BEFORE creating variables
        self._validate_assignments()

        # Variables
        self.x = {}
        self.flex = {}
        self._create_mandatory_variables()
        self._create_flex_variables()

        self.teacher_total_vars = {}
        for teacher in self.teachers:
            self.teacher_total_vars[teacher] = self.model.NewIntVar(0, 1000, f'total_{teacher}')

        self.max_total = self.model.NewIntVar(0, 1000, 'max_total')
        self.min_total = self.model.NewIntVar(0, 1000, 'min_total')

    def _validate_assignments(self):
        """Check that every required grade/subject has at least one eligible teacher."""
        errors = []
        for (grade, s_idx, stream_name) in self.class_groups:
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject, s_idx)
                if required == 0:
                    continue
                # Find eligible teachers
                eligible = []
                for teacher, t_data in self.teachers.items():
                    if f"{teacher}|{grade}" in self.blacklist:
                        continue
                    if f"{subject}|{grade}" in self.subject_blacklist:
                        continue
                    # Check if teacher assigned to this grade/subject (and optionally stream)
                    for assign in t_data.get('assignments', []):
                        if assign.get('grade') != grade:
                            continue
                        if assign.get('subject') != subject:
                            continue
                        if self.per_stream_enabled and assign.get('streamIndex') is not None:
                            if assign['streamIndex'] != s_idx:
                                continue
                        eligible.append(teacher)
                        break
                if not eligible:
                    errors.append(f"Grade {grade} {stream_name} - {subject}: No eligible teacher assigned")
        if errors:
            raise ValueError("Infeasible configuration:\n" + "\n".join(errors))

    def _get_class_key(self, grade, stream_index):
        return f"{grade}_{stream_index}"

    def _create_mandatory_variables(self):
        for (grade, s_idx, stream_name) in self.class_groups:
            class_key = self._get_class_key(grade, s_idx)
            self.x[class_key] = {}
            for d_idx, day in enumerate(self.working_days):
                self.x[class_key][d_idx] = {}
                for slot_idx, slot in enumerate(self.lesson_slots):
                    self.x[class_key][d_idx][slot_idx] = {}
                    for teacher, t_data in self.teachers.items():
                        for assign in t_data.get('assignments', []):
                            if assign.get('grade') != grade:
                                continue
                            if self.per_stream_enabled and assign.get('streamIndex') is not None:
                                if assign['streamIndex'] != s_idx:
                                    continue
                            subject = assign.get('subject')
                            if f"{teacher}|{grade}" in self.blacklist:
                                continue
                            if f"{subject}|{grade}" in self.subject_blacklist:
                                continue
                            var = self.model.NewBoolVar(f'm_{class_key}_{d_idx}_{slot_idx}_{teacher}_{subject}')
                            if teacher not in self.x[class_key][d_idx][slot_idx]:
                                self.x[class_key][d_idx][slot_idx][teacher] = {}
                            self.x[class_key][d_idx][slot_idx][teacher][subject] = var

    def _create_flex_variables(self):
        for (grade, s_idx, stream_name) in self.class_groups:
            class_key = self._get_class_key(grade, s_idx)
            self.flex[class_key] = {}
            for d_idx, day in enumerate(self.working_days):
                self.flex[class_key][d_idx] = {}
                for slot_idx, slot in enumerate(self.lesson_slots):
                    self.flex[class_key][d_idx][slot_idx] = {}
                    for teacher, t_data in self.teachers.items():
                        if not any(a.get('grade') == grade for a in t_data.get('assignments', [])):
                            continue
                        if f"{teacher}|{grade}" in self.blacklist:
                            continue
                        var = self.model.NewBoolVar(f'f_{class_key}_{d_idx}_{slot_idx}_{teacher}')
                        self.flex[class_key][d_idx][slot_idx][teacher] = var

    def _get_required_lessons(self, grade, subject, stream_index=None):
        key = f"{grade}|{subject}"
        if key in self.class_requirements:
            return self.class_requirements[key].get('requiredLessons', 0)
        for subj_data in self.subjects:
            if subj_data[0] == subject:
                return subj_data[1].get('defaultLessons', 0)
        return 0

    def add_constraints(self):
        # 1. One lesson per class per slot
        for (grade, s_idx, stream_name) in self.class_groups:
            class_key = self._get_class_key(grade, s_idx)
            for d_idx in range(self.num_days):
                for slot_idx in range(self.num_slots):
                    all_vars = []
                    for teacher_dict in self.x[class_key][d_idx][slot_idx].values():
                        all_vars.extend(teacher_dict.values())
                    for teacher, var in self.flex[class_key][d_idx][slot_idx].items():
                        all_vars.append(var)
                    self.model.Add(sum(all_vars) <= 1)

        # 2. Teacher cannot teach two classes at once
        for teacher in self.teachers:
            for d_idx in range(self.num_days):
                for slot_idx in range(self.num_slots):
                    teacher_vars = []
                    for (grade, s_idx, stream_name) in self.class_groups:
                        class_key = self._get_class_key(grade, s_idx)
                        if teacher in self.x[class_key][d_idx][slot_idx]:
                            teacher_vars.extend(self.x[class_key][d_idx][slot_idx][teacher].values())
                        if teacher in self.flex[class_key][d_idx][slot_idx]:
                            teacher_vars.append(self.flex[class_key][d_idx][slot_idx][teacher])
                    self.model.Add(sum(teacher_vars) <= 1)

        # 3. Mandatory lessons exact count
        for (grade, s_idx, stream_name) in self.class_groups:
            class_key = self._get_class_key(grade, s_idx)
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject, s_idx)
                if required == 0:
                    continue
                subject_vars = []
                for d_idx in range(self.num_days):
                    for slot_idx in range(self.num_slots):
                        for teacher in self.teachers:
                            if (teacher in self.x[class_key][d_idx][slot_idx] and
                                subject in self.x[class_key][d_idx][slot_idx][teacher]):
                                subject_vars.append(self.x[class_key][d_idx][slot_idx][teacher][subject])
                if subject_vars:
                    self.model.Add(sum(subject_vars) == required)
                else:
                    raise ValueError(f"No variables for {grade} {stream_name} {subject}")

        # 4. Teacher max weekly lessons
        for teacher, t_data in self.teachers.items():
            max_lessons = t_data.get('maxLessons')
            if max_lessons is not None:
                all_teacher_vars = []
                for (grade, s_idx, stream_name) in self.class_groups:
                    class_key = self._get_class_key(grade, s_idx)
                    for d_idx in range(self.num_days):
                        for slot_idx in range(self.num_slots):
                            if teacher in self.x[class_key][d_idx][slot_idx]:
                                all_teacher_vars.extend(self.x[class_key][d_idx][slot_idx][teacher].values())
                            if teacher in self.flex[class_key][d_idx][slot_idx]:
                                all_teacher_vars.append(self.flex[class_key][d_idx][slot_idx][teacher])
                if all_teacher_vars:
                    self.model.Add(sum(all_teacher_vars) <= max_lessons)

        # 5. Unavailable days
        for teacher, t_data in self.teachers.items():
            unavail_days = set(t_data.get('unavailDays', []))
            for d_idx, day in enumerate(self.working_days):
                if day in unavail_days:
                    for (grade, s_idx, stream_name) in self.class_groups:
                        class_key = self._get_class_key(grade, s_idx)
                        for slot_idx in range(self.num_slots):
                            if teacher in self.x[class_key][d_idx][slot_idx]:
                                for var in self.x[class_key][d_idx][slot_idx][teacher].values():
                                    self.model.Add(var == 0)
                            if teacher in self.flex[class_key][d_idx][slot_idx]:
                                self.model.Add(self.flex[class_key][d_idx][slot_idx][teacher] == 0)

        # 6. Max per day (per-teacher or global)
        for teacher, t_data in self.teachers.items():
            teacher_max_per_day = t_data.get('maxPerDay')
            if teacher_max_per_day is None:
                teacher_max_per_day = self.global_max_per_day
            for d_idx in range(self.num_days):
                day_vars = []
                for (grade, s_idx, stream_name) in self.class_groups:
                    class_key = self._get_class_key(grade, s_idx)
                    for slot_idx in range(self.num_slots):
                        if teacher in self.x[class_key][d_idx][slot_idx]:
                            day_vars.extend(self.x[class_key][d_idx][slot_idx][teacher].values())
                        if teacher in self.flex[class_key][d_idx][slot_idx]:
                            day_vars.append(self.flex[class_key][d_idx][slot_idx][teacher])
                if day_vars:
                    self.model.Add(sum(day_vars) <= teacher_max_per_day)

        # 7. No back-to-back same subject
        if self.NO_REPEAT:
            for (grade, s_idx, stream_name) in self.class_groups:
                class_key = self._get_class_key(grade, s_idx)
                for d_idx in range(self.num_days):
                    for slot_idx in range(self.num_slots - 1):
                        for teacher1 in self.teachers:
                            if teacher1 not in self.x[class_key][d_idx][slot_idx]:
                                continue
                            for subject, var1 in self.x[class_key][d_idx][slot_idx][teacher1].items():
                                for teacher2 in self.teachers:
                                    if teacher2 not in self.x[class_key][d_idx][slot_idx+1]:
                                        continue
                                    if subject in self.x[class_key][d_idx][slot_idx+1][teacher2]:
                                        var2 = self.x[class_key][d_idx][slot_idx+1][teacher2][subject]
                                        self.model.Add(var1 + var2 <= 1)

        # 8. Max consecutive same subject (from overrides)
        for key, req in self.class_requirements.items():
            grade_str, subject = key.split('|')
            grade = int(grade_str)
            max_consec = req.get('maxConsecutive', 2)
            if max_consec < self.num_slots:
                for (g, s_idx, stream_name) in self.class_groups:
                    if g != grade:
                        continue
                    class_key = self._get_class_key(grade, s_idx)
                    for d_idx in range(self.num_days):
                        for start in range(self.num_slots - max_consec):
                            vars_in_window = []
                            for offset in range(max_consec + 1):
                                slot_idx = start + offset
                                for teacher in self.teachers:
                                    if (teacher in self.x[class_key][d_idx][slot_idx] and
                                        subject in self.x[class_key][d_idx][slot_idx][teacher]):
                                        vars_in_window.append(self.x[class_key][d_idx][slot_idx][teacher][subject])
                            if vars_in_window:
                                self.model.Add(sum(vars_in_window) <= max_consec)

        # 9. Min one lesson per day (if enabled)
        if self.ENFORCE_MIN_PER_DAY:
            for teacher, t_data in self.teachers.items():
                for d_idx, day in enumerate(self.working_days):
                    if day in t_data.get('unavailDays', []):
                        continue
                    day_vars = []
                    for (grade, s_idx, stream_name) in self.class_groups:
                        class_key = self._get_class_key(grade, s_idx)
                        for slot_idx in range(self.num_slots):
                            if teacher in self.x[class_key][d_idx][slot_idx]:
                                day_vars.extend(self.x[class_key][d_idx][slot_idx][teacher].values())
                            if teacher in self.flex[class_key][d_idx][slot_idx]:
                                day_vars.append(self.flex[class_key][d_idx][slot_idx][teacher])
                    if day_vars:
                        self.model.Add(sum(day_vars) >= 1)

        # 10. Link total variables
        for teacher in self.teachers:
            all_teacher_vars = []
            for (grade, s_idx, stream_name) in self.class_groups:
                class_key = self._get_class_key(grade, s_idx)
                for d_idx in range(self.num_days):
                    for slot_idx in range(self.num_slots):
                        if teacher in self.x[class_key][d_idx][slot_idx]:
                            all_teacher_vars.extend(self.x[class_key][d_idx][slot_idx][teacher].values())
                        if teacher in self.flex[class_key][d_idx][slot_idx]:
                            all_teacher_vars.append(self.flex[class_key][d_idx][slot_idx][teacher])
            if all_teacher_vars:
                self.model.Add(self.teacher_total_vars[teacher] == sum(all_teacher_vars))
            else:
                self.model.Add(self.teacher_total_vars[teacher] == 0)

        # 11. Fairness bounds
        teacher_totals = list(self.teacher_total_vars.values())
        if teacher_totals:
            self.model.AddMaxEquality(self.max_total, teacher_totals)
            self.model.AddMinEquality(self.min_total, teacher_totals)

    def solve(self):
        self.add_constraints()
        self.model.Minimize(self.max_total - self.min_total)
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 60.0
        status = self.solver.Solve(self.model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._extract_solution()
        else:
            return None

    def _extract_solution(self):
        timetable = {}
        for (grade, s_idx, stream_name) in self.class_groups:
            class_key = self._get_class_key(grade, s_idx)
            timetable[class_key] = {
                "grade": grade,
                "streamIndex": s_idx,
                "streamName": stream_name,
                "days": {}
            }
            for d_idx, day in enumerate(self.working_days):
                timetable[class_key]["days"][day] = []
                for slot_idx, slot in enumerate(self.lesson_slots):
                    cell = None
                    for teacher, subjects in self.x[class_key][d_idx][slot_idx].items():
                        for subject, var in subjects.items():
                            if self.solver.Value(var):
                                cell = {"subject": subject, "teacher": teacher, "grade": grade}
                                break
                        if cell:
                            break
                    if not cell:
                        for teacher, var in self.flex[class_key][d_idx][slot_idx].items():
                            if self.solver.Value(var):
                                cell = {"subject": self.FREE_PERIOD_LABEL, "teacher": teacher, "grade": grade, "flex": True}
                                break
                    timetable[class_key]["days"][day].append(cell)
        return timetable

@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = TimetableSolver(config)
        solution = solver.solve()
        if solution:
            return jsonify({"success": True, "timetable": solution})
        else:
            return jsonify({"success": False, "message": "No feasible timetable found with current constraints."})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)})
    except Exception as e:
        return jsonify({"success": False, "message": "Internal error: " + str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)