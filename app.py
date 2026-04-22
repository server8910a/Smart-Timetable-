"""
EduSchedule Pro Backend
Advanced Timetable Generator with OR-Tools CP-SAT Solver
Production-ready with intelligent, specific, and actionable suggestions
"""

import json
import sys
import time
from collections import defaultdict
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
        self.violations = []
        self.penalty_vars = []
        
        # Parse configuration with safe defaults
        self.grades = [int(g) for g in config.get('grades', [])]
        self.subjects = [s[0] for s in config.get('subjects', [])]
        self.teachers = config.get('teachers', {})
        self.time_slots = config.get('timeSlots', [])
        self.working_days = config.get('workingDays', ['MON', 'TUE', 'WED', 'THU', 'FRI'])
        self.class_requirements = config.get('classRequirements', {})
        self.blacklist = set(config.get('blacklist', []))
        self.subject_blacklist = set(config.get('subjectBlacklist', []))
        self.rules = config.get('rules', {})
        self.target_grades = config.get('targetGrades', self.grades)
        self.grade_streams = config.get('gradeStreams', {})
        self.grade_stream_names = config.get('gradeStreamNames', {})
        self.common_session = config.get('commonSession', {'enabled': False})
        
        # Filter lesson slots only
        self.lesson_slots = [s for s in self.time_slots if s.get('type') == 'lesson']
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)
        
        # Safety check
        if self.num_slots == 0:
            raise ValueError("No lesson slots defined. Please configure time slots first.")
        
        # Rules with defaults
        self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 8))
        self.NO_REPEAT = self.rules.get('ruleNoRepeat') == '1'
        
        # Build class groups
        self.class_groups = []
        for grade in self.target_grades:
            streams = int(self.grade_streams.get(str(grade), 1))
            names = self.grade_stream_names.get(str(grade), [])
            if not names:
                names = [f"Stream {chr(65+i)}" for i in range(streams)]
            for s_idx in range(streams):
                stream_name = names[s_idx] if s_idx < len(names) else f"Stream {s_idx+1}"
                self.class_groups.append({
                    'grade': grade,
                    'stream_index': s_idx,
                    'stream_name': stream_name,
                    'key': f"{grade}_{s_idx}"
                })
        
        # Pre-validate
        self._validate_basic()
        self._validate_teacher_assignments()  # NEW: Explicit validation
        
        # Create variables
        self.x = {}
        self._create_variables()
        
        # Teacher totals for fairness
        self.teacher_total = {}
        for t in self.teachers:
            self.teacher_total[t] = self.model.NewIntVar(0, 1000, f'total_{t}')

    def _validate_basic(self):
        """Basic validation to catch obvious errors early"""
        if not self.class_groups:
            raise ValueError("No classes defined. Please add grades first.")
        if not self.teachers:
            raise ValueError("No teachers defined. Please add teachers first.")
        if not self.subjects:
            raise ValueError("No subjects defined. Please add learning areas first.")

    def _validate_teacher_assignments(self):
        """Ensure teachers are only assigned to subjects they actually teach"""
        for teacher, t_data in self.teachers.items():
            teacher_subjects = set()
            for assign in t_data.get('assignments', []):
                teacher_subjects.add(assign.get('subject'))
            
            if not teacher_subjects:
                print(f"[WARNING] Teacher '{teacher}' has no subject assignments", file=sys.stderr)

    def _get_required_lessons(self, grade, subject):
        """Calculate required lessons per week for a grade-subject pair"""
        total = 0
        for teacher, t_data in self.teachers.items():
            if f"{teacher}|{grade}" in self.blacklist:
                continue
            for assign in t_data.get('assignments', []):
                if int(assign.get('grade', 0)) == grade and assign.get('subject') == subject:
                    total += int(assign.get('lessons', 0))
        return total

    def _create_variables(self):
        """Create decision variables ONLY for teacher's assigned subjects"""
        for cg in self.class_groups:
            ck = cg['key']
            grade = cg['grade']
            s_idx = cg['stream_index']
            
            self.x[ck] = {}
            for d_idx in range(self.num_days):
                self.x[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.x[ck][d_idx][slot_idx] = {}
                    
                    for teacher, t_data in self.teachers.items():
                        # Skip blacklisted teachers
                        if f"{teacher}|{grade}" in self.blacklist:
                            continue
                        
                        # Check teacher availability
                        unavail = set(t_data.get('unavailDays', []))
                        if self.working_days[d_idx] in unavail:
                            continue
                        
                        self.x[ck][d_idx][slot_idx][teacher] = {}
                        
                        # ONLY create variables for subjects this teacher is assigned to
                        for assign in t_data.get('assignments', []):
                            if int(assign.get('grade', 0)) != grade:
                                continue
                            
                            # Check stream-specific assignment
                            assign_stream = assign.get('streamIndex')
                            if assign_stream is not None and int(assign_stream) != s_idx:
                                continue
                            
                            subject = assign.get('subject')
                            if not subject:
                                continue
                                
                            # Skip blacklisted subjects
                            if f"{subject}|{grade}" in self.subject_blacklist:
                                continue
                            
                            # Create boolean variable
                            var_name = f'x_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}'
                            self.x[ck][d_idx][slot_idx][teacher][subject] = self.model.NewBoolVar(var_name)

    def _add_hard_constraints(self):
        """Add hard constraints that must be satisfied"""
        
        # 1. Each class has exactly one lesson per slot
        for cg in self.class_groups:
            ck = cg['key']
            for d_idx in range(self.num_days):
                for s_idx in range(self.num_slots):
                    vars_in_slot = []
                    for teacher in self.x[ck][d_idx][s_idx]:
                        vars_in_slot.extend(self.x[ck][d_idx][s_idx][teacher].values())
                    if vars_in_slot:
                        self.model.Add(sum(vars_in_slot) == 1)
        
        # 2. Teacher can't be in two places at once
        for teacher in self.teachers:
            for d_idx in range(self.num_days):
                for s_idx in range(self.num_slots):
                    teacher_vars = []
                    for cg in self.class_groups:
                        ck = cg['key']
                        if teacher in self.x[ck][d_idx][s_idx]:
                            teacher_vars.extend(self.x[ck][d_idx][s_idx][teacher].values())
                    if teacher_vars:
                        self.model.Add(sum(teacher_vars) <= 1)

    def _add_soft_constraints(self):
        """Add soft constraints with penalty weights"""
        
        # 1. Required lessons per subject (CRITICAL - penalty 100000)
        for cg in self.class_groups:
            ck = cg['key']
            grade = cg['grade']
            
            for subject in self.subjects:
                required = self._get_required_lessons(grade, subject)
                if required == 0:
                    continue
                
                # Collect all variables for this subject
                subject_vars = []
                for d_idx in range(self.num_days):
                    for s_idx in range(self.num_slots):
                        for teacher in self.x[ck][d_idx][s_idx]:
                            if subject in self.x[ck][d_idx][s_idx][teacher]:
                                subject_vars.append(self.x[ck][d_idx][s_idx][teacher][subject])
                
                if subject_vars:
                    # Allow shortage with penalty
                    shortage = self.model.NewIntVar(0, required, f'shortage_{ck}_{subject}')
                    self.model.Add(sum(subject_vars) + shortage == required)
                    self.penalty_vars.append((shortage, 100000, 
                        f"Grade {grade} {cg['stream_name']}: Missing {subject} lessons"))
        
        # 2. Teacher daily limit (MEDIUM - penalty 10000)
        for teacher, t_data in self.teachers.items():
            max_per_day = t_data.get('maxPerDay', self.global_max_per_day)
            
            for d_idx in range(self.num_days):
                daily_vars = []
                for cg in self.class_groups:
                    ck = cg['key']
                    if teacher in self.x[ck][d_idx]:
                        for s_idx in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s_idx]:
                                daily_vars.extend(self.x[ck][d_idx][s_idx][teacher].values())
                
                if daily_vars:
                    overload = self.model.NewIntVar(0, len(daily_vars), f'overload_{teacher}_{d_idx}')
                    self.model.Add(sum(daily_vars) <= max_per_day + overload)
                    self.penalty_vars.append((overload, 10000, 
                        f"{teacher}: Exceeded daily max on {self.working_days[d_idx]}"))
        
        # 3. Teacher weekly limit (HIGH - penalty 50000)
        for teacher, t_data in self.teachers.items():
            max_weekly = t_data.get('maxLessons')
            if max_weekly:
                weekly_vars = []
                for cg in self.class_groups:
                    ck = cg['key']
                    for d_idx in range(self.num_days):
                        for s_idx in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s_idx]:
                                weekly_vars.extend(self.x[ck][d_idx][s_idx][teacher].values())
                
                if weekly_vars:
                    overload = self.model.NewIntVar(0, len(weekly_vars), f'week_overload_{teacher}')
                    self.model.Add(sum(weekly_vars) <= max_weekly + overload)
                    self.penalty_vars.append((overload, 50000, 
                        f"{teacher}: Exceeded weekly max"))
        
        # 4. Avoid back-to-back same subject (LOW - penalty 2000)
        if self.NO_REPEAT:
            for cg in self.class_groups:
                ck = cg['key']
                grade = cg['grade']
                
                for d_idx in range(self.num_days):
                    for s_idx in range(self.num_slots - 1):
                        for t1 in self.teachers:
                            if t1 not in self.x[ck][d_idx][s_idx]:
                                continue
                            for subject, v1 in self.x[ck][d_idx][s_idx][t1].items():
                                for t2 in self.teachers:
                                    if t2 not in self.x[ck][d_idx][s_idx + 1]:
                                        continue
                                    if subject in self.x[ck][d_idx][s_idx + 1][t2]:
                                        v2 = self.x[ck][d_idx][s_idx + 1][t2][subject]
                                        viol = self.model.NewBoolVar(f'btb_{ck}_{d_idx}_{s_idx}_{subject}')
                                        self.model.Add(v1 + v2 <= 1 + viol)
                                        self.penalty_vars.append((viol, 2000,
                                            f"Back-to-back {subject} in Grade {grade}"))
        
        # Calculate teacher totals for fairness
        for teacher in self.teachers:
            weekly_vars = []
            for cg in self.class_groups:
                ck = cg['key']
                for d_idx in range(self.num_days):
                    for s_idx in range(self.num_slots):
                        if teacher in self.x[ck][d_idx][s_idx]:
                            weekly_vars.extend(self.x[ck][d_idx][s_idx][teacher].values())
            if weekly_vars:
                self.model.Add(self.teacher_total[teacher] == sum(weekly_vars))

    def _analyze_infeasibility(self):
        """Analyze why the timetable is infeasible and suggest SPECIFIC, ACTIONABLE fixes."""
        suggestions = []
        
        # Track detailed statistics
        teacher_subject_totals = defaultdict(lambda: defaultdict(int))
        teacher_grade_totals = defaultdict(lambda: defaultdict(int))
        subject_grade_totals = defaultdict(lambda: defaultdict(int))
        teacher_assigned_subjects = defaultdict(set)
        
        # Build teacher assigned subjects map
        for teacher, t_data in self.teachers.items():
            for assign in t_data.get('assignments', []):
                teacher_assigned_subjects[teacher].add(assign.get('subject'))
        
        # Calculate total required vs available
        total_required = 0
        total_available = len(self.class_groups) * self.num_days * self.num_slots
        
        for cg in self.class_groups:
            grade = cg['grade']
            for subject in self.subjects:
                req = self._get_required_lessons(grade, subject)
                total_required += req
                subject_grade_totals[subject][grade] = req
                
                # Track which teachers teach what
                for teacher, t_data in self.teachers.items():
                    if f"{teacher}|{grade}" in self.blacklist:
                        continue
                    for assign in t_data.get('assignments', []):
                        if int(assign.get('grade', 0)) == grade and assign.get('subject') == subject:
                            lessons = int(assign.get('lessons', 0))
                            teacher_subject_totals[teacher][subject] += lessons
                            teacher_grade_totals[teacher][grade] += lessons
        
        # ========== PRIORITY 1: REDUCE LESSONS (Most Practical) ==========
        
        # 1. CAPACITY CHECK - Suggest reducing lessons first
        if total_required > total_available:
            shortage = total_required - total_available
            slots_per_day = self.num_slots
            days = self.num_days
            classes_count = len(self.class_groups)
            
            # Find subjects with highest lesson counts to suggest reductions
            subject_totals = {}
            for subject in self.subjects:
                subject_totals[subject] = sum(subject_grade_totals[subject].values())
            
            high_volume_subjects = sorted(subject_totals.items(), key=lambda x: x[1], reverse=True)[:3]
            subject_names = [s[0] for s in high_volume_subjects]
            
            suggestions.append({
                "type": "capacity",
                "message": f"Total required lessons ({total_required}) exceed available slots ({total_available}) by {shortage}.",
                "fixes": [
                    f"PRIMARY FIX: Reduce lessons by {shortage} total. Suggested subjects to reduce: {', '.join(subject_names)}",
                    f"Reduce each subject by 1 lesson where possible (would save up to {len(self.subjects) * classes_count} lessons)",
                    f"Remove or combine stream groups if applicable",
                    f"LAST RESORT: Add an extra working day (would add {classes_count * slots_per_day} slots)"
                ]
            })
        
        # 2. TEACHER OVERLOAD - Suggest redistribution first
        for teacher, t_data in self.teachers.items():
            teacher_required = sum(teacher_subject_totals[teacher].values())
            max_weekly = t_data.get('maxLessons')
            
            if max_weekly and teacher_required > max_weekly:
                overload = teacher_required - max_weekly
                
                # Find which subjects this teacher could offload
                teacher_subjects = list(teacher_subject_totals[teacher].keys())
                
                # Find other teachers who teach the SAME subjects and have capacity
                available_teachers = []
                for other_teacher, other_data in self.teachers.items():
                    if other_teacher == teacher:
                        continue
                    # Check if other teacher teaches any of the same subjects
                    other_subjects = teacher_assigned_subjects[other_teacher]
                    if not other_subjects.intersection(teacher_subjects):
                        continue
                    other_required = sum(teacher_subject_totals[other_teacher].values())
                    other_max = other_data.get('maxLessons')
                    if other_max and other_required < other_max:
                        available_teachers.append(other_teacher)
                
                fixes = [
                    f"PRIMARY FIX: Reduce {teacher}'s lessons by {overload} (from {teacher_required} to {max_weekly})",
                ]
                
                if available_teachers:
                    fixes.append(f"Reassign {overload} lessons to teachers who teach the same subjects: {', '.join(available_teachers[:3])}")
                else:
                    fixes.append(f"No other teachers available for {teacher}'s subjects. Consider training another teacher.")
                
                fixes.append(f"LAST RESORT: Increase {teacher}'s weekly maximum to {teacher_required}")
                
                suggestions.append({
                    "type": "teacher_overload",
                    "teacher": teacher,
                    "message": f"{teacher} is assigned {teacher_required} lessons but maximum is {max_weekly}.",
                    "fixes": fixes
                })
        
        # 3. DAILY LIMIT CHECK - Suggest reducing lessons or adjusting availability
        for teacher, t_data in self.teachers.items():
            teacher_required = sum(teacher_subject_totals[teacher].values())
            max_per_day = t_data.get('maxPerDay', self.global_max_per_day)
            unavail_days = len(set(t_data.get('unavailDays', [])))
            available_days = self.num_days - unavail_days
            
            if available_days > 0:
                daily_needed = teacher_required / available_days
                if daily_needed > max_per_day:
                    suggested_max = int(daily_needed) + 1
                    
                    fixes = [
                        f"PRIMARY FIX: Increase {teacher}'s daily maximum from {max_per_day} to at least {suggested_max}",
                        f"Reduce {teacher}'s total lessons from {teacher_required} to {max_per_day * available_days}"
                    ]
                    
                    if unavail_days > 0:
                        fixes.append(f"Reduce {teacher}'s unavailable days (currently {unavail_days} days off)")
                    
                    fixes.append(f"LAST RESORT: Spread {teacher}'s lessons across more days")
                    
                    suggestions.append({
                        "type": "daily_limit",
                        "teacher": teacher,
                        "message": f"{teacher} needs {daily_needed:.1f} lessons per available day but daily maximum is {max_per_day}.",
                        "fixes": fixes
                    })
        
        # 4. UNASSIGNED SUBJECTS - Suggest assigning to teachers who CAN teach it
        for subject in self.subjects:
            teachers_for_subject = [t for t, subs in teacher_subject_totals.items() if subject in subs]
            if len(teachers_for_subject) == 0:
                total_needed = sum(subject_grade_totals[subject].values())
                if total_needed > 0:
                    # Find teachers with capacity who could potentially teach this
                    available_teachers = []
                    for other_teacher, other_data in self.teachers.items():
                        other_required = sum(teacher_subject_totals[other_teacher].values())
                        other_max = other_data.get('maxLessons')
                        if other_max and other_required < other_max:
                            available_teachers.append(other_teacher)
                    
                    fixes = [f"Assign a teacher to teach {subject}"]
                    if available_teachers:
                        fixes.append(f"Suggested teachers with capacity: {', '.join(available_teachers[:3])}")
                    fixes.append(f"LAST RESORT: Add a new teacher qualified to teach {subject}")
                    
                    suggestions.append({
                        "type": "unassigned_subject",
                        "subject": subject,
                        "message": f"{subject} requires {total_needed} total lessons but no teacher is assigned.",
                        "fixes": fixes
                    })
        
        # 5. SUBJECT-TEACHER MISMATCH - Teacher assigned to subject they don't teach
        for teacher, t_data in self.teachers.items():
            assigned_subjects = teacher_assigned_subjects[teacher]
            for grade in self.grades:
                for subject in self.subjects:
                    # Check if there's a requirement for this subject in this grade
                    if subject_grade_totals[subject].get(grade, 0) > 0:
                        # Check if ANY teacher is assigned to this subject for this grade
                        teachers_for_this = []
                        for other_teacher in self.teachers:
                            if subject in teacher_assigned_subjects[other_teacher]:
                                # Check if assigned to this grade
                                for assign in self.teachers[other_teacher].get('assignments', []):
                                    if assign.get('grade') == grade and assign.get('subject') == subject:
                                        teachers_for_this.append(other_teacher)
                                        break
                        
                        if not teachers_for_this:
                            suggestions.append({
                                "type": "unassigned_grade_subject",
                                "subject": subject,
                                "message": f"Grade {grade} needs {subject} but no teacher is assigned to teach it for this grade.",
                                "fixes": [
                                    f"Assign a teacher to teach {subject} for Grade {grade}",
                                    f"Remove {subject} requirement for Grade {grade}",
                                    f"LAST RESORT: Add a new teacher qualified to teach {subject}"
                                ]
                            })
        
        # 6. BLACKLIST CONFLICTS
        for teacher, t_data in self.teachers.items():
            for grade in self.grades:
                if f"{teacher}|{grade}" in self.blacklist:
                    # Check if teacher has assignments for this grade
                    if teacher_grade_totals[teacher].get(grade, 0) > 0:
                        lessons = teacher_grade_totals[teacher][grade]
                        suggestions.append({
                            "type": "blacklist_conflict",
                            "teacher": teacher,
                            "message": f"{teacher} is blacklisted from Grade {grade} but has {lessons} lessons assigned there.",
                            "fixes": [
                                f"PRIMARY FIX: Remove the blacklist for {teacher} on Grade {grade}",
                                f"Reassign {teacher}'s {lessons} lessons for Grade {grade} to another teacher",
                                f"LAST RESORT: Remove {teacher}'s assignments for Grade {grade}"
                            ]
                        })
        
        return suggestions

    def solve(self):
        """Solve the timetabling problem"""
        print(f"[INFO] Building model with {len(self.class_groups)} classes, "
              f"{len(self.teachers)} teachers, {self.num_days} days, {self.num_slots} slots/day",
              file=sys.stderr)
        
        self._add_hard_constraints()
        self._add_soft_constraints()
        
        # Fairness objective: minimize teacher workload variance
        if len(self.teacher_total) > 1:
            max_lessons = self.model.NewIntVar(0, 1000, 'max_lessons')
            min_lessons = self.model.NewIntVar(0, 1000, 'min_lessons')
            totals = list(self.teacher_total.values())
            self.model.AddMaxEquality(max_lessons, totals)
            self.model.AddMinEquality(min_lessons, totals)
            fairness_penalty = max_lessons - min_lessons
        else:
            fairness_penalty = 0
        
        # Total penalty from soft constraints
        total_penalty = sum(weight * var for var, weight, _ in self.penalty_vars) if self.penalty_vars else 0
        
        # Objective: minimize combined penalties
        self.model.Minimize(fairness_penalty * 1000 + total_penalty)
        
        # Configure solver
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300.0
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.log_search_progress = True
        
        print(f"[INFO] Starting solver...", file=sys.stderr)
        start_time = time.time()
        status = self.solver.Solve(self.model)
        solve_time = time.time() - start_time
        
        print(f"[INFO] Solver finished in {solve_time:.2f}s with status: {self.solver.StatusName(status)}",
              file=sys.stderr)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution()
        
        return None

    def _extract_solution(self):
        """Extract the timetable from solver solution"""
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
                    cell = None
                    
                    # Check if this is a common session slot
                    if (self.common_session.get('enabled') and 
                        day == self.common_session.get('day', 'FRI') and 
                        slot_idx == int(self.common_session.get('slotIndex', 0))):
                        timetable[ck]["days"][day].append(None)
                        continue
                    
                    for teacher in self.x[ck][d_idx][slot_idx]:
                        for subject, var in self.x[ck][d_idx][slot_idx][teacher].items():
                            if self.solver.Value(var):
                                cell = {
                                    "subject": subject,
                                    "teacher": teacher,
                                    "grade": cg['grade']
                                }
                                break
                        if cell:
                            break
                    
                    timetable[ck]["days"][day].append(cell)
        
        return timetable

    def get_violations(self):
        """Get list of constraint violations"""
        if not self.solver:
            return []
        
        violations = []
        for var, weight, description in self.penalty_vars:
            value = self.solver.Value(var)
            if value > 0:
                severity = "High" if weight >= 50000 else "Medium" if weight >= 10000 else "Low"
                violations.append({
                    "description": description,
                    "severity": severity,
                    "value": value
                })
        
        return violations


@app.route('/generate', methods=['POST'])
def generate():
    """Generate timetable endpoint"""
    try:
        config = request.json
        
        # Create and solve
        solver = TimetableSolver(config)
        solution = solver.solve()
        
        if solution:
            violations = solver.get_violations()
            return jsonify({
                "success": True,
                "timetable": solution,
                "violations": violations
            })
        else:
            # Analyze why it failed and suggest fixes
            suggestions = solver._analyze_infeasibility()
            return jsonify({
                "success": False,
                "message": "Could not generate timetable. Review suggestions below.",
                "suggestions": suggestions
            })
            
    except ValueError as e:
        return jsonify({
            "success": False,
            "message": str(e),
            "suggestions": []
        })
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "suggestions": []
        })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "timestamp": time.time(),
        "service": "EduSchedule Pro Backend"
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)