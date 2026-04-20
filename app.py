import json
import sys
import time
import random
import math
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS

# CORRECTED: lowercase 'import'
from ortools.sat.python import cp_model

app = Flask(__name__)
CORS(app)


class TimetableSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = None
        self.diagnostics = []

        # Parse configuration with type safety
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
        self._create_variables()

        self.teacher_total_vars = {t: self.model.NewIntVar(0, 1000, f'total_{t}') for t in self.teachers}
        self.max_total = self.model.NewIntVar(0, 1000, 'max_total')
        self.min_total = self.model.NewIntVar(0, 1000, 'min_total')

    def _validate_assignments(self):
        errors = []
        for (grade, s_idx, stream_name) in self.class_groups:
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0:
                    continue
                eligible = False
                for teacher, t_data in self.teachers.items():
                    if f"{teacher}|{grade}" in self.blacklist:
                        continue
                    if f"{subject}|{grade}" in self.subject_blacklist:
                        continue
                    for assign in t_data.get('assignments', []):
                        # CRITICAL FIX: Type-safe grade comparison
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
                    errors.append(f"Grade {grade} {stream_name} - {subject}: No eligible teacher")
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))

    def _get_class_key(self, g, s):
        return f"{g}_{s}"

    def _get_required_lessons(self, grade, subject):
        key = f"{grade}|{subject}"
        if key in self.class_requirements:
            return self.class_requirements[key].get('requiredLessons', 0)
        for subj in self.subjects:
            if subj[0] == subject:
                return subj[1].get('defaultLessons', 0)
        return 0

    def _create_variables(self):
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
                    
                    # Create flex variable for EVERY slot (forces exact one lesson/free per slot)
                    self.flex[ck][d_idx][slot_idx] = self.model.NewBoolVar(f'flex_{ck}_{d_idx}_{slot_idx}')
                    
                    for teacher, t_data in self.teachers.items():
                        for assign in t_data.get('assignments', []):
                            # CRITICAL FIX: Type-safe grade comparison
                            if int(assign.get('grade', 0)) != int(grade):
                                continue
                            # Handle stream index correctly
                            if self.per_stream_enabled and assign.get('streamIndex') is not None:
                                if int(assign.get('streamIndex', 0)) != s_idx:
                                    continue
                            subject = assign.get('subject')
                            if f"{teacher}|{grade}" in self.blacklist:
                                continue
                            if f"{subject}|{grade}" in self.subject_blacklist:
                                continue
                            var = self.model.NewBoolVar(f'm_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}')
                            if teacher not in self.x[ck][d_idx][slot_idx]:
                                self.x[ck][d_idx][slot_idx][teacher] = {}
                            self.x[ck][d_idx][slot_idx][teacher][subject] = var

    def _add_constraints(self):
        # CRITICAL FIX: EXACT EQUALITY for each slot
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    vars_list = []
                    # Add all teacher-subject variables
                    for teacher_dict in self.x[ck][d][s].values():
                        vars_list.extend(teacher_dict.values())
                    # Add the flex variable
                    vars_list.append(self.flex[ck][d][s])
                    # EXACT EQUALITY: must have exactly one (lesson or free)
                    self.model.Add(sum(vars_list) == 1)

        # Common session block
        if self.common_session.get('enabled'):
            day = self.common_session.get('day', 'FRI')
            slot_idx = self.common_session.get('slotIndex', 0)
            if day in self.working_days and slot_idx < self.num_slots:
                d_idx = self.working_days.index(day)
                for (grade, s_idx, stream_name) in self.class_groups:
                    ck = self._get_class_key(grade, s_idx)
                    # Force flex variable to be 1 (free period blocked for common session)
                    self.model.Add(self.flex[ck][d_idx][slot_idx] == 1)

        # Teacher cannot teach two classes at once
        for teacher in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = []
                    for (g, si, _) in self.class_groups:
                        ck = self._get_class_key(g, si)
                        if ck in self.x and d in self.x[ck] and s in self.x[ck][d] and teacher in self.x[ck][d][s]:
                            tv.extend(self.x[ck][d][s][teacher].values())
                    if tv:
                        self.model.Add(sum(tv) <= 1)

        # CRITICAL FIX: MANDATORY LESSONS EXACT COUNT (== not <=)
        for (grade, s_idx, stream_name) in self.class_groups:
            ck = self._get_class_key(grade, s_idx)
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0:
                    continue
                vars_list = []
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        for teacher in self.teachers:
                            if teacher in self.x[ck][d][s] and subject in self.x[ck][d][s][teacher]:
                                vars_list.append(self.x[ck][d][s][teacher][subject])
                if vars_list:
                    # EXACT EQUALITY: must place exactly the required number of lessons
                    self.model.Add(sum(vars_list) == required)

        # Teacher max weekly lessons
        for teacher, t_data in self.teachers.items():
            if t_data.get('maxLessons'):
                vars_list = []
                for (g, si, _) in self.class_groups:
                    ck = self._get_class_key(g, si)
                    for d in range(self.num_days):
                        for s in range(self.num_slots):
                            if teacher in self.x[ck][d][s]:
                                vars_list.extend(self.x[ck][d][s][teacher].values())
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
                if vars_list:
                    self.model.Add(sum(vars_list) <= max_per_day)

        # No back-to-back same subject (optional)
        if self.NO_REPEAT:
            for (grade, s_idx, stream_name) in self.class_groups:
                ck = self._get_class_key(grade, s_idx)
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
                                        self.model.Add(v1 + self.x[ck][d][s+1][t2][subj] <= 1)

        # Link total variables for fairness
        for teacher in self.teachers:
            vars_list = []
            for (g, si, _) in self.class_groups:
                ck = self._get_class_key(g, si)
                for d in range(self.num_days):
                    for s in range(self.num_slots):
                        if teacher in self.x[ck][d][s]:
                            vars_list.extend(self.x[ck][d][s][teacher].values())
            if vars_list:
                self.model.Add(self.teacher_total_vars[teacher] == sum(vars_list))

        # Fairness bounds
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
                    # Check if flex variable is true (free period)
                    if self.solver.Value(self.flex[ck][d_idx][slot_idx]):
                        timetable[ck]["days"][day].append(None)
                    else:
                        cell = None
                        for teacher, subjects in self.x[ck][d_idx][slot_idx].items():
                            for subject, var in subjects.items():
                                if self.solver.Value(var):
                                    cell = {"subject": subject, "teacher": teacher, "grade": grade}
                                    break
                            if cell:
                                break
                        timetable[ck]["days"][day].append(cell)
        return timetable


@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = TimetableSolver(config)
        solution = solver.solve()

        if solution:
            return jsonify({
                "success": True,
                "timetable": solution,
                "diagnostics": solver.diagnostics
            })
        else:
            return jsonify({
                "success": False,
                "message": "No feasible timetable found. Check constraints or increase available slots.",
                "diagnostics": solver.diagnostics
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