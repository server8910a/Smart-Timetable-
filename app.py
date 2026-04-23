"""
EduSchedule Pro Backend
Advanced Timetable Generator with OR-Tools CP-SAT Solver
Production-ready with intelligent, specific, and actionable suggestions
"""

import json
import sys
import time
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Violation:
    """Constraint violation representation"""
    description: str
    severity: ConstraintSeverity
    value: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "severity": self.severity.value,
            "value": self.value
        }


@dataclass
class Suggestion:
    """Infeasibility suggestion representation - ANY ONE can solve the problem"""
    type: str
    message: str
    fixes: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "message": self.message,
            "fixes": self.fixes
        }
        if self.metadata:
            data.update(self.metadata)
        return data


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

class ConfigValidator:
    """Validates input configuration"""
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        required_fields = ['grades', 'subjects', 'teachers', 'timeSlots', 'workingDays']
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        if not isinstance(config['grades'], list) or len(config['grades']) == 0:
            return False, "grades must be a non-empty list"
        
        if not isinstance(config['subjects'], list) or len(config['subjects']) == 0:
            return False, "subjects must be a non-empty list"
        
        if not isinstance(config['teachers'], dict) or len(config['teachers']) == 0:
            return False, "teachers must be a non-empty dictionary"
        
        if not isinstance(config['timeSlots'], list) or len(config['timeSlots']) == 0:
            return False, "timeSlots must be a non-empty list"
        
        if not isinstance(config['workingDays'], list) or len(config['workingDays']) == 0:
            return False, "workingDays must be a non-empty list"
        
        try:
            [int(g) for g in config['grades']]
        except (ValueError, TypeError):
            return False, "All grades must be integers"
        
        for teacher_name, teacher_data in config['teachers'].items():
            if not isinstance(teacher_data, dict):
                return False, f"Teacher '{teacher_name}' data must be a dictionary"
            if 'assignments' not in teacher_data:
                return False, f"Teacher '{teacher_name}' missing assignments field"
            if not isinstance(teacher_data['assignments'], list):
                return False, f"Teacher '{teacher_name}' assignments must be a list"
        
        for i, slot in enumerate(config['timeSlots']):
            if 'type' not in slot:
                return False, f"TimeSlot {i} missing 'type' field"
        
        return True, None


# ============================================================================
# TIMETABLE SOLVER
# ============================================================================

class TimetableSolver:
    """Advanced timetable solver using OR-Tools CP-SAT"""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            is_valid, error_msg = ConfigValidator.validate(config)
            if not is_valid:
                raise ValueError(error_msg)
            
            logger.info(f"Initializing TimetableSolver with {len(config.get('grades', []))} grades, "
                       f"{len(config.get('teachers', {}))} teachers")
            
            self.config = config
            self.model = cp_model.CpModel()
            self.solver = None
            self.violations: List[Tuple[Any, int, str]] = []
            self.penalty_vars: List[Tuple[Any, int, str]] = []
            
            # Parse configuration
            self.grades = [int(g) for g in config.get('grades', [])]
            self.subjects = [s[0] if isinstance(s, (list, tuple)) else s 
                           for s in config.get('subjects', [])]
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
            
            self.lesson_slots = [s for s in self.time_slots if s.get('type') == 'lesson']
            self.num_slots = len(self.lesson_slots)
            self.num_days = len(self.working_days)
            
            if self.num_slots == 0:
                raise ValueError("No lesson slots defined. Please configure time slots first.")
            
            if self.num_days == 0:
                raise ValueError("No working days defined. Please configure working days.")
            
            self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 8))
            self.NO_REPEAT = self.rules.get('ruleNoRepeat') == '1'
            
            self.class_groups = self._build_class_groups()
            
            self._validate_basic()
            self._validate_teacher_assignments()
            
            self.x: Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, Any]]]]] = {}
            self._create_variables()
            
            self.teacher_total: Dict[str, Any] = {}
            for t in self.teachers:
                self.teacher_total[t] = self.model.NewIntVar(0, 1000, f'total_{t}')
            
            logger.info("TimetableSolver initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing TimetableSolver: {str(e)}", exc_info=True)
            raise

    def _build_class_groups(self) -> List[Dict[str, Any]]:
        class_groups = []
        for grade in self.target_grades:
            try:
                streams = int(self.grade_streams.get(str(grade), 1))
                names = self.grade_stream_names.get(str(grade), [])
                if not names:
                    names = [f"Stream {chr(65 + i)}" for i in range(streams)]
                
                for s_idx in range(streams):
                    stream_name = names[s_idx] if s_idx < len(names) else f"Stream {s_idx + 1}"
                    class_groups.append({
                        'grade': grade,
                        'stream_index': s_idx,
                        'stream_name': stream_name,
                        'key': f"{grade}_{s_idx}"
                    })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error building class groups for grade {grade}: {str(e)}")
                continue
        
        return class_groups

    def _validate_basic(self) -> None:
        if not self.class_groups:
            raise ValueError("No classes defined. Please add grades first.")
        if not self.teachers:
            raise ValueError("No teachers defined. Please add teachers first.")
        if not self.subjects:
            raise ValueError("No subjects defined. Please add learning areas first.")

    def _validate_teacher_assignments(self) -> None:
        for teacher, t_data in self.teachers.items():
            try:
                teacher_subjects = set()
                for assign in t_data.get('assignments', []):
                    subject = assign.get('subject')
                    if subject:
                        teacher_subjects.add(subject)
                
                if not teacher_subjects:
                    logger.warning(f"Teacher '{teacher}' has no subject assignments")
            except Exception as e:
                logger.warning(f"Error validating assignments for {teacher}: {str(e)}")

    def _get_required_lessons(self, grade: int, subject: str) -> int:
        total = 0
        try:
            for teacher, t_data in self.teachers.items():
                if f"{teacher}|{grade}" in self.blacklist:
                    continue
                
                for assign in t_data.get('assignments', []):
                    if int(assign.get('grade', 0)) == grade and assign.get('subject') == subject:
                        total += int(assign.get('lessons', 0))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error calculating required lessons for {grade}, {subject}: {str(e)}")
        
        return total

    def _create_variables(self) -> None:
        try:
            var_count = 0
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
                            if f"{teacher}|{grade}" in self.blacklist:
                                continue
                            
                            unavail = set(t_data.get('unavailDays', []))
                            if self.working_days[d_idx] in unavail:
                                continue
                            
                            self.x[ck][d_idx][slot_idx][teacher] = {}
                            
                            for assign in t_data.get('assignments', []):
                                try:
                                    if int(assign.get('grade', 0)) != grade:
                                        continue
                                    
                                    assign_stream = assign.get('streamIndex')
                                    if assign_stream is not None and int(assign_stream) != s_idx:
                                        continue
                                    
                                    subject = assign.get('subject')
                                    if not subject:
                                        continue
                                    
                                    if f"{subject}|{grade}" in self.subject_blacklist:
                                        continue
                                    
                                    var_name = f'x_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}'
                                    self.x[ck][d_idx][slot_idx][teacher][subject] = \
                                        self.model.NewBoolVar(var_name)
                                    var_count += 1
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error creating variable: {str(e)}")
            
            logger.info(f"Created {var_count} decision variables")
        except Exception as e:
            logger.error(f"Error creating variables: {str(e)}", exc_info=True)
            raise

    def _add_hard_constraints(self) -> None:
        try:
            for cg in self.class_groups:
                ck = cg['key']
                for d_idx in range(self.num_days):
                    for s_idx in range(self.num_slots):
                        vars_in_slot = []
                        for teacher in self.x[ck][d_idx][s_idx]:
                            vars_in_slot.extend(self.x[ck][d_idx][s_idx][teacher].values())
                        if vars_in_slot:
                            self.model.Add(sum(vars_in_slot) == 1)
            
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
            
            logger.info("Hard constraints added successfully")
        except Exception as e:
            logger.error(f"Error adding hard constraints: {str(e)}", exc_info=True)
            raise

    def _add_soft_constraints(self) -> None:
        try:
            for cg in self.class_groups:
                ck = cg['key']
                grade = cg['grade']
                
                for subject in self.subjects:
                    required = self._get_required_lessons(grade, subject)
                    if required == 0:
                        continue
                    
                    subject_vars = []
                    for d_idx in range(self.num_days):
                        for s_idx in range(self.num_slots):
                            for teacher in self.x[ck][d_idx][s_idx]:
                                if subject in self.x[ck][d_idx][s_idx][teacher]:
                                    subject_vars.append(
                                        self.x[ck][d_idx][s_idx][teacher][subject]
                                    )
                    
                    if subject_vars:
                        shortage = self.model.NewIntVar(
                            0, required, f'shortage_{ck}_{subject}'
                        )
                        self.model.Add(sum(subject_vars) + shortage == required)
                        self.penalty_vars.append(
                            (shortage, 100000, 
                             f"Grade {grade} {cg['stream_name']}: Missing {subject} lessons")
                        )
            
            for teacher, t_data in self.teachers.items():
                max_per_day = t_data.get('maxPerDay', self.global_max_per_day)
                
                for d_idx in range(self.num_days):
                    daily_vars = []
                    for cg in self.class_groups:
                        ck = cg['key']
                        if teacher in self.x[ck][d_idx]:
                            for s_idx in range(self.num_slots):
                                if teacher in self.x[ck][d_idx][s_idx]:
                                    daily_vars.extend(
                                        self.x[ck][d_idx][s_idx][teacher].values()
                                    )
                    
                    if daily_vars:
                        overload = self.model.NewIntVar(
                            0, len(daily_vars), f'overload_{teacher}_{d_idx}'
                        )
                        self.model.Add(sum(daily_vars) <= max_per_day + overload)
                        self.penalty_vars.append(
                            (overload, 10000, 
                             f"{teacher}: Exceeded daily max on {self.working_days[d_idx]}")
                        )
            
            for teacher, t_data in self.teachers.items():
                max_weekly = t_data.get('maxLessons')
                if max_weekly:
                    weekly_vars = []
                    for cg in self.class_groups:
                        ck = cg['key']
                        for d_idx in range(self.num_days):
                            for s_idx in range(self.num_slots):
                                if teacher in self.x[ck][d_idx][s_idx]:
                                    weekly_vars.extend(
                                        self.x[ck][d_idx][s_idx][teacher].values()
                                    )
                    
                    if weekly_vars:
                        overload = self.model.NewIntVar(
                            0, len(weekly_vars), f'week_overload_{teacher}'
                        )
                        self.model.Add(sum(weekly_vars) <= max_weekly + overload)
                        self.penalty_vars.append(
                            (overload, 50000, f"{teacher}: Exceeded weekly max")
                        )
            
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
                                            viol = self.model.NewBoolVar(
                                                f'btb_{ck}_{d_idx}_{s_idx}_{subject}'
                                            )
                                            self.model.Add(v1 + v2 <= 1 + viol)
                                            self.penalty_vars.append(
                                                (viol, 2000,
                                                 f"Back-to-back {subject} in Grade {grade}")
                                            )
            
            for teacher in self.teachers:
                weekly_vars = []
                for cg in self.class_groups:
                    ck = cg['key']
                    for d_idx in range(self.num_days):
                        for s_idx in range(self.num_slots):
                            if teacher in self.x[ck][d_idx][s_idx]:
                                weekly_vars.extend(
                                    self.x[ck][d_idx][s_idx][teacher].values()
                                )
                if weekly_vars:
                    self.model.Add(self.teacher_total[teacher] == sum(weekly_vars))
            
            logger.info(f"Soft constraints added successfully ({len(self.penalty_vars)} penalty vars)")
        except Exception as e:
            logger.error(f"Error adding soft constraints: {str(e)}", exc_info=True)
            raise

    def _analyze_infeasibility(self) -> List[Suggestion]:
        """
        Exhaustive analysis of why the timetable is infeasible.
        Checks hard constraints first, then soft constraints.
        Each suggestion is a complete, independent solution.
        """
        suggestions: List[Suggestion] = []
        
        try:
            logger.info(f"Starting infeasibility analysis: {len(self.class_groups)} classes, "
                       f"{len(self.teachers)} teachers, {len(self.subjects)} subjects")
            
            # Pre-compute teacher availability
            teacher_availability = {}
            for teacher, t_data in self.teachers.items():
                unavail = set(t_data.get('unavailDays', []))
                teacher_availability[teacher] = [d for d in self.working_days if d not in unavail]
            
            # =========================================================================
            # PHASE 1: HARD CONSTRAINT CONFLICTS (These make the problem impossible)
            # =========================================================================
            
            for cg in self.class_groups:
                grade = cg['grade']
                stream_idx = cg['stream_index']
                stream_name = cg['stream_name']
                
                for subject in self.subjects:
                    required = self._get_required_lessons(grade, subject)
                    if required == 0:
                        continue
                    
                    # Find eligible teachers for this grade/stream/subject
                    eligible_teachers = []
                    teacher_available_slots = {}
                    
                    for teacher, t_data in self.teachers.items():
                        # Check blacklist
                        if f"{teacher}|{grade}" in self.blacklist:
                            continue
                        # Check subject blacklist
                        if f"{subject}|{grade}" in self.subject_blacklist:
                            continue
                        
                        # Check if teacher is assigned to this subject for this grade/stream
                        teaches_this = False
                        for assign in t_data.get('assignments', []):
                            if (assign.get('grade') == grade and 
                                assign.get('subject') == subject):
                                # Stream check
                                assign_stream = assign.get('streamIndex')
                                if assign_stream is not None and assign_stream != stream_idx:
                                    continue
                                teaches_this = True
                                break
                        
                        if not teaches_this:
                            continue
                        
                        eligible_teachers.append(teacher)
                        avail_days = teacher_availability.get(teacher, [])
                        teacher_available_slots[teacher] = len(avail_days) * self.num_slots
                    
                    # Conflict 1: No eligible teacher at all
                    if not eligible_teachers:
                        suggestions.append(Suggestion(
                            type="hard_no_teacher",
                            message=f"Grade {grade} {stream_name} needs {required} lessons of {subject}, "
                                    f"but NO teacher is assigned to teach it.",
                            fixes=[
                                f"Assign a teacher to teach {subject} for Grade {grade} {stream_name}",
                                f"Remove {subject} requirement for this class",
                                f"Check if {subject} is blacklisted for Grade {grade}"
                            ],
                            metadata={"grade": grade, "subject": subject, "stream": stream_name}
                        ))
                        continue
                    
                    # Conflict 2: Total available slots from all eligible teachers < required
                    total_available = sum(teacher_available_slots.values())
                    if total_available < required:
                        suggestions.append(Suggestion(
                            type="hard_insufficient_slots",
                            message=f"Grade {grade} {stream_name} needs {required} lessons of {subject}, "
                                    f"but eligible teachers have only {total_available} total available slots.",
                            fixes=[
                                f"PRIMARY: Reduce required lessons for {subject} to {total_available}",
                                f"Increase teacher availability (reduce unavailable days)",
                                f"LAST RESORT: Add another teacher for {subject}"
                            ],
                            metadata={"grade": grade, "subject": subject, "available": total_available, "required": required}
                        ))
                    
                    # Conflict 3: Daily slot capacity per subject per class
                    max_slots_per_day = self.num_slots
                    if required > self.num_days * max_slots_per_day:
                        suggestions.append(Suggestion(
                            type="hard_daily_capacity",
                            message=f"Grade {grade} {stream_name} needs {required} lessons of {subject}, "
                                    f"but maximum possible per week is {self.num_days * max_slots_per_day}.",
                            fixes=[
                                f"PRIMARY: Reduce {subject} lessons to at most {self.num_days * max_slots_per_day}",
                                f"LAST RESORT: Add more lesson slots per day or extra working day"
                            ],
                            metadata={"grade": grade, "subject": subject}
                        ))
            
            # Conflict 4: Teacher completely unavailable but assigned lessons
            for teacher, t_data in self.teachers.items():
                avail_days = teacher_availability.get(teacher, [])
                if not avail_days:
                    total_assigned = 0
                    for grade in self.grades:
                        for subject in self.subjects:
                            for assign in t_data.get('assignments', []):
                                if assign.get('grade') == grade and assign.get('subject') == subject:
                                    total_assigned += assign.get('lessons', 0)
                    
                    if total_assigned > 0:
                        suggestions.append(Suggestion(
                            type="teacher_never_available",
                            message=f"{teacher} is assigned {total_assigned} lessons but is unavailable on all days.",
                            fixes=[
                                f"PRIMARY: Remove unavailable days for {teacher}",
                                f"Reassign all {teacher}'s lessons to other teachers",
                                f"LAST RESORT: Remove {teacher}'s assignments"
                            ],
                            metadata={"teacher": teacher, "lessons": total_assigned}
                        ))
            
            # Conflict 5: Common session reduces available slots below requirement
            if self.common_session.get('enabled'):
                session_day = self.common_session.get('day', 'FRI')
                session_slot = int(self.common_session.get('slotIndex', 0))
                if session_day in self.working_days and session_slot < self.num_slots:
                    effective_slots_per_week = self.num_days * self.num_slots - 1
                    for cg in self.class_groups:
                        grade = cg['grade']
                        for subject in self.subjects:
                            required = self._get_required_lessons(grade, subject)
                            if required > effective_slots_per_week:
                                suggestions.append(Suggestion(
                                    type="common_session_conflict",
                                    message=f"Grade {grade} {cg['stream_name']} needs {required} lessons of {subject}, "
                                            f"but only {effective_slots_per_week} slots available (common session takes one).",
                                    fixes=[
                                        f"PRIMARY: Reduce {subject} lessons to {effective_slots_per_week}",
                                        f"Disable common session or move to a break slot",
                                        f"LAST RESORT: Add an extra working day"
                                    ],
                                    metadata={"grade": grade, "subject": subject}
                                ))
            
            # Conflict 6: Blacklist contradictions (teacher assigned but blacklisted)
            for teacher, t_data in self.teachers.items():
                for grade in self.grades:
                    if f"{teacher}|{grade}" in self.blacklist:
                        assigned_lessons = 0
                        for assign in t_data.get('assignments', []):
                            if assign.get('grade') == grade:
                                assigned_lessons += assign.get('lessons', 0)
                        
                        if assigned_lessons > 0:
                            suggestions.append(Suggestion(
                                type="blacklist_contradiction",
                                message=f"{teacher} is blacklisted from Grade {grade} but has {assigned_lessons} lessons assigned.",
                                fixes=[
                                    f"PRIMARY: Remove the blacklist for {teacher} on Grade {grade}",
                                    f"Reassign {teacher}'s Grade {grade} lessons to another teacher",
                                    f"LAST RESORT: Remove {teacher}'s assignments for Grade {grade}"
                                ],
                                metadata={"teacher": teacher, "grade": grade}
                            ))
            
            # =========================================================================
            # PHASE 2: SOFT CONSTRAINT VIOLATIONS (Only if no hard conflicts found)
            # =========================================================================
            if not suggestions:
                logger.info("No hard conflicts found; analyzing soft constraints")
                
                teacher_subject_totals = defaultdict(lambda: defaultdict(int))
                subject_grade_totals = defaultdict(lambda: defaultdict(int))
                
                total_required = 0
                total_available = len(self.class_groups) * self.num_days * self.num_slots
                
                for cg in self.class_groups:
                    grade = cg['grade']
                    for subject in self.subjects:
                        req = self._get_required_lessons(grade, subject)
                        total_required += req
                        subject_grade_totals[subject][grade] = req
                        
                        for teacher, t_data in self.teachers.items():
                            for assign in t_data.get('assignments', []):
                                if assign.get('grade') == grade and assign.get('subject') == subject:
                                    teacher_subject_totals[teacher][subject] += assign.get('lessons', 0)
                
                # Capacity overload
                if total_required > total_available:
                    shortage = total_required - total_available
                    subject_totals = {s: sum(subject_grade_totals[s].values()) for s in self.subjects}
                    high_volume = sorted(subject_totals.items(), key=lambda x: x[1], reverse=True)[:3]
                    subject_names = [s[0] for s in high_volume]
                    
                    suggestions.append(Suggestion(
                        type="capacity_overload",
                        message=f"Total required lessons ({total_required}) exceed available slots ({total_available}) by {shortage}.",
                        fixes=[
                            f"PRIMARY: Reduce lessons by {shortage} (suggested: {', '.join(subject_names)})",
                            f"Remove a stream to free {self.num_days * self.num_slots} slots",
                            f"LAST RESORT: Add an extra working day to gain {len(self.class_groups) * self.num_slots} slots"
                        ]
                    ))
                
                # Teacher overload
                for teacher, t_data in self.teachers.items():
                    teacher_req = sum(teacher_subject_totals[teacher].values())
                    max_weekly = t_data.get('maxLessons')
                    if max_weekly and teacher_req > max_weekly:
                        overload = teacher_req - max_weekly
                        
                        # Find other teachers who teach same subjects and have capacity
                        teacher_subjects = set(teacher_subject_totals[teacher].keys())
                        available_teachers = []
                        for other, other_data in self.teachers.items():
                            if other == teacher:
                                continue
                            other_subjects = set()
                            for assign in other_data.get('assignments', []):
                                other_subjects.add(assign.get('subject'))
                            if not teacher_subjects.intersection(other_subjects):
                                continue
                            other_req = sum(teacher_subject_totals[other].values())
                            other_max = other_data.get('maxLessons')
                            if other_max and other_req < other_max:
                                available_teachers.append(other)
                        
                        fixes = [
                            f"PRIMARY: Reduce {teacher}'s lessons by {overload} (from {teacher_req} to {max_weekly})"
                        ]
                        if available_teachers:
                            fixes.append(f"Reassign {overload} lessons to {available_teachers[0]} (teaches same subjects, has capacity)")
                        fixes.append(f"LAST RESORT: Increase {teacher}'s weekly maximum to {teacher_req}")
                        
                        suggestions.append(Suggestion(
                            type="teacher_overload",
                            message=f"{teacher} is assigned {teacher_req} lessons but max is {max_weekly}.",
                            fixes=fixes,
                            metadata={"teacher": teacher}
                        ))
                
                # Daily limit
                for teacher, t_data in self.teachers.items():
                    teacher_req = sum(teacher_subject_totals[teacher].values())
                    max_per_day = t_data.get('maxPerDay', self.global_max_per_day)
                    avail_days = len(teacher_availability.get(teacher, []))
                    if avail_days > 0 and teacher_req / avail_days > max_per_day:
                        daily_needed = teacher_req / avail_days
                        reduction_needed = teacher_req - (max_per_day * avail_days)
                        
                        fixes = [
                            f"PRIMARY: Increase {teacher}'s daily max from {max_per_day} to {int(daily_needed) + 1}"
                        ]
                        if reduction_needed > 0:
                            fixes.append(f"Reduce {teacher}'s total lessons by {reduction_needed} (from {teacher_req} to {max_per_day * avail_days})")
                        fixes.append(f"LAST RESORT: Reduce {teacher}'s unavailable days")
                        
                        suggestions.append(Suggestion(
                            type="daily_limit",
                            message=f"{teacher} needs {daily_needed:.1f} lessons/day but max is {max_per_day}.",
                            fixes=fixes,
                            metadata={"teacher": teacher}
                        ))
            
            # If still no suggestions, provide a diagnostic fallback
            if not suggestions:
                logger.warning("No specific issue identified; likely a complex constraint interaction")
                suggestions.append(Suggestion(
                    type="complex_conflict",
                    message="The solver could not find a solution due to a combination of constraints.",
                    fixes=[
                        "Try reducing the number of lessons gradually to isolate the conflict",
                        "Check for back-to-back subject restrictions combined with limited slots",
                        "Verify stream-specific assignments match exactly",
                        "Consider increasing the solver time limit"
                    ]
                ))
            
            logger.info(f"Analysis complete: {len(suggestions)} independent solutions generated")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error in infeasibility analysis: {str(e)}", exc_info=True)
            return [Suggestion(
                type="analysis_error",
                message=f"Error during analysis: {str(e)}",
                fixes=["Check server logs for details", "Verify configuration format"]
            )]

    def solve(self) -> Optional[Dict[str, Any]]:
        try:
            logger.info(
                f"Building model with {len(self.class_groups)} classes, "
                f"{len(self.teachers)} teachers, {self.num_days} days, "
                f"{self.num_slots} slots/day"
            )
            
            self._add_hard_constraints()
            self._add_soft_constraints()
            
            if len(self.teacher_total) > 1:
                max_lessons = self.model.NewIntVar(0, 1000, 'max_lessons')
                min_lessons = self.model.NewIntVar(0, 1000, 'min_lessons')
                totals = list(self.teacher_total.values())
                self.model.AddMaxEquality(max_lessons, totals)
                self.model.AddMinEquality(min_lessons, totals)
                fairness_penalty = max_lessons - min_lessons
            else:
                fairness_penalty = 0
            
            total_penalty = (
                sum(weight * var for var, weight, _ in self.penalty_vars) 
                if self.penalty_vars else 0
            )
            
            self.model.Minimize(fairness_penalty * 1000 + total_penalty)
            
            self.solver = cp_model.CpSolver()
            self.solver.parameters.max_time_in_seconds = 300.0
            self.solver.parameters.num_search_workers = 8
            self.solver.parameters.log_search_progress = True
            
            logger.info("Starting solver...")
            start_time = time.time()
            status = self.solver.Solve(self.model)
            solve_time = time.time() - start_time
            
            logger.info(
                f"Solver finished in {solve_time:.2f}s with status: "
                f"{self.solver.StatusName(status)}"
            )
            
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return self._extract_solution()
            
            return None
        except Exception as e:
            logger.error(f"Error during solve: {str(e)}", exc_info=True)
            raise

    def _extract_solution(self) -> Dict[str, Any]:
        timetable: Dict[str, Any] = {}
        
        try:
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
                        
                        if (self.common_session.get('enabled') and 
                            day == self.common_session.get('day', 'FRI') and 
                            slot_idx == int(self.common_session.get('slotIndex', 0))):
                            timetable[ck]["days"][day].append(None)
                            continue
                        
                        for teacher in self.x[ck][d_idx][slot_idx]:
                            for subject, var in self.x[ck][d_idx][slot_idx][teacher].items():
                                if self.solver and self.solver.Value(var):
                                    cell = {
                                        "subject": subject,
                                        "teacher": teacher,
                                        "grade": cg['grade']
                                    }
                                    break
                            if cell:
                                break
                        
                        timetable[ck]["days"][day].append(cell)
            
            logger.info(f"Solution extracted successfully for {len(timetable)} classes")
            return timetable
        except Exception as e:
            logger.error(f"Error extracting solution: {str(e)}", exc_info=True)
            raise

    def get_violations(self) -> List[Dict[str, Any]]:
        if not self.solver:
            return []
        
        violations: List[Violation] = []
        try:
            for var, weight, description in self.penalty_vars:
                value = self.solver.Value(var)
                if value > 0:
                    severity = (ConstraintSeverity.HIGH 
                               if weight >= 50000 
                               else ConstraintSeverity.MEDIUM 
                               if weight >= 10000 
                               else ConstraintSeverity.LOW)
                    violations.append(Violation(
                        description=description,
                        severity=severity,
                        value=value
                    ))
        except Exception as e:
            logger.error(f"Error getting violations: {str(e)}", exc_info=True)
        
        return [v.to_dict() for v in violations]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        if not config:
            return jsonify({
                "success": False,
                "message": "Request body must contain valid JSON configuration",
                "suggestions": []
            }), 400        
        logger.info(f"Received timetable generation request")
        
        solver = TimetableSolver(config)
        solution = solver.solve()
        
        if solution:
            violations = solver.get_violations()
            logger.info(f"Solution found with {len(violations)} violations")
            return jsonify({
                "success": True,
                "timetable": solution,
                "violations": violations
            })
        else:
            suggestions = solver._analyze_infeasibility()
            logger.info(f"No solution found. Generated {len(suggestions)} independent solutions")
            return jsonify({
                "success": False,
                "message": "Could not generate timetable. Each suggestion below is a COMPLETE solution.",
                "suggestions": [s.to_dict() for s in suggestions]
            })
            
    except ValueError as e:
        logger.warning(f"Validation error in /generate: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e),
            "suggestions": []
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in /generate: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "suggestions": []
        }), 500


@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify({
            "status": "ok",
            "timestamp": time.time(),
            "service": "EduSchedule Pro Backend",
            "version": "3.0.0"
        })
    except Exception as e:
        logger.error(f"Error in /health: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {str(error)}", exc_info=True)
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500


if __name__ == '__main__':
    logger.info("Starting EduSchedule Pro Backend v3.0.0")
    app.run(debug=False, host='0.0.0.0', port=5000)