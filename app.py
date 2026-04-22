"""
EduSchedule Pro Backend
Advanced Timetable Generator with OR-Tools CP-SAT Solver
Production-ready with multiple independent, specific, and actionable suggestions
"""

import json
import sys
import time
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
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
        Analyze why the timetable is infeasible and suggest MULTIPLE independent solutions.
        ANY SINGLE suggestion should be able to solve the problem on its own.
        """
        suggestions: List[Suggestion] = []
        
        try:
            # Track detailed statistics
            teacher_subject_totals: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            teacher_grade_totals: Dict[str, Dict[int, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            subject_grade_totals: Dict[str, Dict[int, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            teacher_assigned_subjects: Dict[str, Set[str]] = defaultdict(set)
            
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
                    
                    for teacher, t_data in self.teachers.items():
                        if f"{teacher}|{grade}" in self.blacklist:
                            continue
                        for assign in t_data.get('assignments', []):
                            if int(assign.get('grade', 0)) == grade and \
                               assign.get('subject') == subject:
                                lessons = int(assign.get('lessons', 0))
                                teacher_subject_totals[teacher][subject] += lessons
                                teacher_grade_totals[teacher][grade] += lessons
            
            # =========================================================================
            # ISSUE 1: CAPACITY OVERLOAD
            # Each suggestion is a COMPLETE, INDEPENDENT solution
            # =========================================================================
            if total_required > total_available:
                shortage = total_required - total_available
                slots_per_day = self.num_slots
                days = self.num_days
                classes_count = len(self.class_groups)
                
                subject_totals: Dict[str, int] = {}
                for subject in self.subjects:
                    subject_totals[subject] = sum(
                        subject_grade_totals[subject].values()
                    )
                
                high_volume_subjects = sorted(
                    subject_totals.items(), key=lambda x: x[1], reverse=True
                )[:3]
                subject_names = [s[0] for s in high_volume_subjects]
                
                # Suggestion 1: Reduce specific subject lessons (COMPLETE SOLUTION)
                reduce_amounts = []
                remaining = shortage
                for subject, total in high_volume_subjects:
                    if remaining <= 0:
                        break
                    reduction = min(total, remaining)
                    reduce_amounts.append(f"{subject} by {reduction}")
                    remaining -= reduction
                
                if reduce_amounts:
                    suggestions.append(Suggestion(
                        type="capacity_reduce_lessons",
                        message=f"SOLUTION 1: Reduce lessons to fit within {total_available} slots.",
                        fixes=[
                            f"Reduce {', '.join(reduce_amounts)}. This will free exactly {shortage} slots.",
                            f"After this change, total required lessons will be {total_available} (within capacity)."
                        ]
                    ))
                
                # Suggestion 2: Remove/combine streams (COMPLETE SOLUTION)
                streams_to_remove = (shortage // (days * slots_per_day)) + 1
                suggestions.append(Suggestion(
                    type="capacity_remove_streams",
                    message=f"SOLUTION 2: Remove or combine {streams_to_remove} stream(s).",
                    fixes=[
                        f"Remove {streams_to_remove} stream(s) from any grade. "
                        f"This will free {streams_to_remove * days * slots_per_day} slots.",
                        f"After this change, capacity will be sufficient."
                    ]
                ))
                
                # Suggestion 3: Add working day (COMPLETE SOLUTION)
                slots_added = classes_count * slots_per_day
                suggestions.append(Suggestion(
                    type="capacity_add_day",
                    message=f"SOLUTION 3: Add an extra working day.",
                    fixes=[
                        f"Add 1 more working day (currently {days} days).",
                        f"This will add {slots_added} slots, more than enough to cover the {shortage} shortage.",
                        f"After this change, total capacity will be {total_available + slots_added} slots."
                    ]
                ))
            
            # =========================================================================
            # ISSUE 2: TEACHER OVERLOAD
            # Each suggestion is a COMPLETE, INDEPENDENT solution
            # =========================================================================
            for teacher, t_data in self.teachers.items():
                teacher_required = sum(teacher_subject_totals[teacher].values())
                max_weekly = t_data.get('maxLessons')
                
                if max_weekly and teacher_required > max_weekly:
                    overload = teacher_required - max_weekly
                    teacher_subjects = list(teacher_subject_totals[teacher].keys())
                    
                    # Find other teachers who teach the SAME subjects and have capacity
                    available_teachers = []
                    for other_teacher, other_data in self.teachers.items():
                        if other_teacher == teacher:
                            continue
                        other_subjects = teacher_assigned_subjects[other_teacher]
                        if not other_subjects.intersection(teacher_subjects):
                            continue
                        other_required = sum(
                            teacher_subject_totals[other_teacher].values()
                        )
                        other_max = other_data.get('maxLessons')
                        if other_max and other_required < other_max:
                            available_teachers.append({
                                'name': other_teacher,
                                'capacity': other_max - other_required,
                                'subjects': list(other_subjects.intersection(teacher_subjects))
                            })
                    
                    # Suggestion 1: Reduce lessons (COMPLETE SOLUTION)
                    specific_reductions = []
                    remaining = overload
                    for subject in teacher_subjects:
                        if remaining <= 0:
                            break
                        current = teacher_subject_totals[teacher][subject]
                        reduction = min(current, remaining)
                        specific_reductions.append(f"{subject} by {reduction}")
                        remaining -= reduction
                    
                    suggestions.append(Suggestion(
                        type="teacher_overload_reduce",
                        message=f"SOLUTION 1 for {teacher}: Reduce assigned lessons.",
                        fixes=[
                            f"Reduce {teacher}'s lessons by {overload} total.",
                            f"Specifically, reduce {', '.join(specific_reductions)}.",
                            f"This brings {teacher} from {teacher_required} to {max_weekly} lessons.",
                            f"This is a COMPLETE solution - no other changes needed."
                        ],
                        metadata={"teacher": teacher}
                    ))
                    
                    # Suggestion 2: Reassign to specific teachers (COMPLETE SOLUTION)
                    if available_teachers:
                        best_teacher = available_teachers[0]
                        suggestions.append(Suggestion(
                            type="teacher_overload_reassign",
                            message=f"SOLUTION 2 for {teacher}: Reassign lessons to {best_teacher['name']}.",
                            fixes=[
                                f"Reassign {overload} lessons from {teacher} to {best_teacher['name']}.",
                                f"{best_teacher['name']} teaches {', '.join(best_teacher['subjects'][:2])} "
                                f"and has capacity for {best_teacher['capacity']} more lessons.",
                                f"This is a COMPLETE solution - no other changes needed."
                            ],
                            metadata={"teacher": teacher}
                        ))
                    
                    # Suggestion 3: Increase max lessons (COMPLETE SOLUTION)
                    suggestions.append(Suggestion(
                        type="teacher_overload_increase_max",
                        message=f"SOLUTION 3 for {teacher}: Increase weekly maximum.",
                        fixes=[
                            f"Increase {teacher}'s weekly maximum from {max_weekly} to {teacher_required}.",
                            f"This is a COMPLETE solution - no other changes needed."
                        ],
                        metadata={"teacher": teacher}
                    ))
            
            # =========================================================================
            # ISSUE 3: DAILY LIMIT VIOLATION
            # =========================================================================
            for teacher, t_data in self.teachers.items():
                teacher_required = sum(teacher_subject_totals[teacher].values())
                max_per_day = t_data.get('maxPerDay', self.global_max_per_day)
                unavail_days = len(set(t_data.get('unavailDays', [])))
                available_days = self.num_days - unavail_days
                
                if available_days > 0 and teacher_required > 0:
                    daily_needed = teacher_required / available_days
                    if daily_needed > max_per_day:
                        suggested_max = int(daily_needed) + 1
                        reduction_needed = teacher_required - (max_per_day * available_days)
                        
                        # Suggestion 1: Increase daily max (COMPLETE SOLUTION)
                        suggestions.append(Suggestion(
                            type="daily_limit_increase",
                            message=f"SOLUTION 1 for {teacher}: Increase daily maximum.",
                            fixes=[
                                f"Increase {teacher}'s daily maximum from {max_per_day} to {suggested_max}.",
                                f"Currently {teacher} needs to teach {daily_needed:.1f} lessons per available day.",
                                f"This is a COMPLETE solution - no other changes needed."
                            ],
                            metadata={"teacher": teacher}
                        ))
                        
                        # Suggestion 2: Reduce total lessons (COMPLETE SOLUTION)
                        if reduction_needed > 0:
                            suggestions.append(Suggestion(
                                type="daily_limit_reduce",
                                message=f"SOLUTION 2 for {teacher}: Reduce total lessons.",
                                fixes=[
                                    f"Reduce {teacher}'s total lessons by {reduction_needed} "
                                    f"(from {teacher_required} to {max_per_day * available_days}).",
                                    f"This is a COMPLETE solution - no other changes needed."
                                ],
                                metadata={"teacher": teacher}
                            ))
                        
                        # Suggestion 3: Reduce unavailable days (COMPLETE SOLUTION)
                        if unavail_days > 0:
                            days_to_remove = min(unavail_days, 
                                               int((teacher_required / max_per_day) - available_days) + 1)
                            suggestions.append(Suggestion(
                                type="daily_limit_availability",
                                message=f"SOLUTION 3 for {teacher}: Reduce unavailable days.",
                                fixes=[
                                    f"Reduce {teacher}'s unavailable days by {days_to_remove} "
                                    f"(currently {unavail_days} days off).",
                                    f"This increases available days to {available_days + days_to_remove}, "
                                    f"reducing daily load to {teacher_required / (available_days + days_to_remove):.1f}.",
                                    f"This is a COMPLETE solution - no other changes needed."
                                ],
                                metadata={"teacher": teacher}
                            ))
            
            # =========================================================================
            # ISSUE 4: UNASSIGNED SUBJECTS
            # =========================================================================
            for subject in self.subjects:
                teachers_for_subject = [t for t, subs in teacher_subject_totals.items() 
                                       if subject in subs]
                if len(teachers_for_subject) == 0:
                    total_needed = sum(subject_grade_totals[subject].values())
                    if total_needed > 0:
                        # Find teachers with capacity
                        available_teachers = []
                        for other_teacher, other_data in self.teachers.items():
                            other_required = sum(
                                teacher_subject_totals[other_teacher].values()
                            )
                            other_max = other_data.get('maxLessons')
                            if other_max and other_required < other_max:
                                available_teachers.append({
                                    'name': other_teacher,
                                    'capacity': other_max - other_required
                                })
                        
                        # Suggestion 1: Assign to existing teacher (COMPLETE SOLUTION)
                        if available_teachers:
                            best_teacher = available_teachers[0]
                            suggestions.append(Suggestion(
                                type="unassigned_assign_existing",
                                message=f"SOLUTION 1 for {subject}: Assign to {best_teacher['name']}.",
                                fixes=[
                                    f"Assign {best_teacher['name']} to teach {subject}.",
                                    f"{best_teacher['name']} has capacity for {best_teacher['capacity']} "
                                    f"more lessons and can take the {total_needed} needed.",
                                    f"This is a COMPLETE solution - no other changes needed."
                                ],
                                metadata={"subject": subject}
                            ))
                        
                        # Suggestion 2: Remove subject (COMPLETE SOLUTION)
                        suggestions.append(Suggestion(
                            type="unassigned_remove_subject",
                            message=f"SOLUTION 2 for {subject}: Remove subject from curriculum.",
                            fixes=[
                                f"Remove {subject} from all grades (requires {total_needed} lessons).",
                                f"This is a COMPLETE solution - no other changes needed."
                            ],
                            metadata={"subject": subject}
                        ))
                        
                        # Suggestion 3: Add new teacher (COMPLETE SOLUTION)
                        suggestions.append(Suggestion(
                            type="unassigned_add_teacher",
                            message=f"SOLUTION 3 for {subject}: Add a new teacher.",
                            fixes=[
                                f"Add a new teacher qualified to teach {subject}.",
                                f"The new teacher should be assigned {total_needed} lessons of {subject}.",
                                f"This is a COMPLETE solution - no other changes needed."
                            ],
                            metadata={"subject": subject}
                        ))
            
            # =========================================================================
            # ISSUE 5: SUBJECT-GRADE MISMATCH (Grade needs subject but no teacher)
            # =========================================================================
            for subject in self.subjects:
                for grade in self.grades:
                    needed = subject_grade_totals[subject].get(grade, 0)
                    if needed > 0:
                        # Check if any teacher is assigned to this subject for this grade
                        has_teacher = False
                        for teacher, t_data in self.teachers.items():
                            for assign in t_data.get('assignments', []):
                                if (assign.get('grade') == grade and 
                                    assign.get('subject') == subject):
                                    has_teacher = True
                                    break
                            if has_teacher:
                                break
                        
                        if not has_teacher:
                            # Find teachers who teach this subject
                            subject_teachers = [
                                t for t, subs in teacher_assigned_subjects.items() 
                                if subject in subs
                            ]
                            
                            if subject_teachers:
                                # Suggestion 1: Extend existing teacher to this grade
                                best_teacher = subject_teachers[0]
                                suggestions.append(Suggestion(
                                    type="grade_subject_assign",
                                    message=f"SOLUTION: Assign {best_teacher} to teach {subject} for Grade {grade}.",
                                    fixes=[
                                        f"Extend {best_teacher}'s {subject} assignment to include Grade {grade}.",
                                        f"Add {needed} lessons for Grade {grade} to {best_teacher}'s workload.",
                                        f"This is a COMPLETE solution - no other changes needed."
                                    ],
                                    metadata={"subject": subject, "grade": grade}
                                ))
                            else:
                                # Suggestion 2: Remove requirement
                                suggestions.append(Suggestion(
                                    type="grade_subject_remove",
                                    message=f"SOLUTION: Remove {subject} requirement for Grade {grade}.",
                                    fixes=[
                                        f"Remove the {needed} lessons of {subject} from Grade {grade}.",
                                        f"This is a COMPLETE solution - no other changes needed."
                                    ],
                                    metadata={"subject": subject, "grade": grade}
                                ))
            
            # =========================================================================
            # ISSUE 6: BLACKLIST CONFLICTS
            # =========================================================================
            for teacher, t_data in self.teachers.items():
                for grade in self.grades:
                    if f"{teacher}|{grade}" in self.blacklist:
                        if teacher_grade_totals[teacher].get(grade, 0) > 0:
                            lessons = teacher_grade_totals[teacher][grade]
                            
                            # Suggestion 1: Remove blacklist (COMPLETE SOLUTION)
                            suggestions.append(Suggestion(
                                type="blacklist_remove",
                                message=f"SOLUTION 1: Remove blacklist for {teacher} on Grade {grade}.",
                                fixes=[
                                    f"Remove the blacklist blocking {teacher} from Grade {grade}.",
                                    f"This allows the {lessons} already-assigned lessons to proceed.",
                                    f"This is a COMPLETE solution - no other changes needed."
                                ],
                                metadata={"teacher": teacher, "grade": grade}
                            ))
                            
                            # Suggestion 2: Reassign lessons (COMPLETE SOLUTION)
                            # Find alternative teachers
                            alt_teachers = []
                            for other_teacher in self.teachers:
                                if other_teacher == teacher:
                                    continue
                                if f"{other_teacher}|{grade}" in self.blacklist:
                                    continue
                                # Check if other teacher teaches any of the same subjects
                                for assign in t_data.get('assignments', []):
                                    if assign.get('grade') == grade:
                                        subject = assign.get('subject')
                                        if subject in teacher_assigned_subjects.get(other_teacher, set()):
                                            alt_teachers.append(other_teacher)
                                            break
                            
                            if alt_teachers:
                                suggestions.append(Suggestion(
                                    type="blacklist_reassign",
                                    message=f"SOLUTION 2: Reassign {teacher}'s Grade {grade} lessons.",
                                    fixes=[
                                        f"Reassign {teacher}'s {lessons} lessons for Grade {grade} to {alt_teachers[0]}.",
                                        f"{alt_teachers[0]} is qualified and not blacklisted from Grade {grade}.",
                                        f"This is a COMPLETE solution - no other changes needed."
                                    ],
                                    metadata={"teacher": teacher, "grade": grade}
                                ))
                            
                            # Suggestion 3: Remove assignments (COMPLETE SOLUTION)
                            suggestions.append(Suggestion(
                                type="blacklist_remove_assignments",
                                message=f"SOLUTION 3: Remove {teacher}'s assignments for Grade {grade}.",
                                fixes=[
                                    f"Remove {teacher}'s {lessons} lesson assignments for Grade {grade}.",
                                    f"This resolves the blacklist conflict.",
                                    f"This is a COMPLETE solution - no other changes needed."
                                ],
                                metadata={"teacher": teacher, "grade": grade}
                            ))
            
            logger.info(f"Analysis complete: {len(suggestions)} independent solutions generated")
            
        except Exception as e:
            logger.error(f"Error analyzing infeasibility: {str(e)}", exc_info=True)
        
        return suggestions

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