"""
EduSchedule Pro Backend - Version 4.0.0
Advanced Timetable Generator with Performance Optimizations
+ Exhaustive Diagnostic Logging + Specific Actionable Suggestions
"""

import json
import sys
import time
import logging
import hashlib
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
# CACHING SYSTEM
# ============================================================================

class CachedSolver:
    """Caches solver results to avoid redundant computations"""
    
    def __init__(self, max_cache_size: int = 500):
        self.cache = {}
        self.max_cache_size = max_cache_size
    
    def get_hash(self, config: Dict) -> str:
        """Generate hash of configuration"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get(self, config: Dict) -> Optional[Dict]:
        """Get cached result"""
        key = self.get_hash(config)
        if key in self.cache:
            logger.info(f"Cache HIT for configuration {key[:8]}...")
        return self.cache.get(key)
    
    def set(self, config: Dict, result: Dict) -> None:
        """Cache result"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest (first key)
            self.cache.pop(next(iter(self.cache)))
        key = self.get_hash(config)
        self.cache[key] = result
        logger.info(f"Cached result for configuration {key[:8]}...")


cache_manager = CachedSolver()


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class ConstraintSeverity(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Violation:
    description: str
    severity: ConstraintSeverity
    value: int
    penalty: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "severity": self.severity.value,
            "value": self.value,
            "penalty": self.penalty
        }


@dataclass
class Suggestion:
    type: str
    message: str
    fixes: List[str]
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=highest, 3=lowest
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "message": self.message,
            "fixes": self.fixes,
            "priority": self.priority
        }
        if self.metadata:
            data.update(self.metadata)
        return data


@dataclass
class SolverStats:
    total_variables: int = 0
    total_constraints: int = 0
    solve_time: float = 0.0
    status: str = "UNKNOWN"
    optimal: bool = False
    wall_time: float = 0.0
    branches: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "totalVariables": self.total_variables,
            "totalConstraints": self.total_constraints,
            "solveTime": round(self.solve_time, 3),
            "status": self.status,
            "optimal": self.optimal,
            "wallTime": round(self.wall_time, 3),
            "branches": self.branches
        }


# ============================================================================
# CONFIGURATION VALIDATION & PREPROCESSING
# ============================================================================

class ConfigValidator:
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        required = ['grades', 'subjects', 'teachers', 'timeSlots', 'workingDays']
        for field in required:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        if not config['grades'] or not config['subjects'] or not config['teachers']:
            return False, "Grades, subjects, and teachers cannot be empty"
        
        return True, None
    
    @staticmethod
    def preprocess(config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove infeasible assignments early"""
        preprocessed = json.loads(json.dumps(config))
        
        blacklist = set(preprocessed.get('blacklist', []))
        subject_blacklist = set(preprocessed.get('subjectBlacklist', []))
        
        for teacher, t_data in preprocessed['teachers'].items():
            filtered = []
            for assign in t_data.get('assignments', []):
                grade = assign.get('grade')
                subject = assign.get('subject')
                
                if f"{teacher}|{grade}" in blacklist:
                    continue
                if f"{subject}|{grade}" in subject_blacklist:
                    continue
                
                filtered.append(assign)
            
            t_data['assignments'] = filtered
        
        return preprocessed


# ============================================================================
# ADVANCED TIMETABLE SOLVER
# ============================================================================

class TimetableSolver:
    
    def __init__(self, config: Dict[str, Any], use_cache: bool = True):
        try:
            # Check cache
            if use_cache:
                cached = cache_manager.get(config)
                if cached:
                    logger.info("Using cached result - skipping computation")
                    self._load_from_cache(cached)
                    return
            
            is_valid, error_msg = ConfigValidator.validate(config)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Preprocess
            config = ConfigValidator.preprocess(config)
            
            logger.info(f"Initializing solver with {len(config.get('grades', []))} grades, "
                       f"{len(config.get('teachers', {}))} teachers")
            
            self.config = config
            self.model = cp_model.CpModel()
            self.solver = None
            self.stats = SolverStats()
            
            # Parse
            self.grades = sorted([int(g) for g in config.get('grades', [])])
            self.subjects = [s[0] if isinstance(s, (list, tuple)) else s 
                           for s in config.get('subjects', [])]
            self.teachers = config.get('teachers', {})
            self.time_slots = config.get('timeSlots', [])
            self.working_days = config.get('workingDays', ['MON', 'TUE', 'WED', 'THU', 'FRI'])
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
                raise ValueError("No lesson slots defined")
            if self.num_days == 0:
                raise ValueError("No working days defined")
            
            self.global_max_per_day = int(self.rules.get('maxTeacherPerDay', 8))
            self.NO_REPEAT = self.rules.get('ruleNoRepeat') == '1'
            
            self.class_groups = self._build_class_groups()
            self._validate_basic()
            
            self.x: Dict = {}
            self.teacher_total: Dict[str, Any] = {}
            self.penalty_vars: List[Tuple[Any, int, str]] = []
            self.diagnostic_counts = defaultdict(lambda: defaultdict(int))
            
            self._create_variables()
            
            logger.info("Solver initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}", exc_info=True)
            raise
    
    def _load_from_cache(self, cached: Dict) -> None:
        """Load state from cache"""
        self.config = cached.get('config', {})
        self.stats = cached.get('stats', SolverStats())
        self.grades = cached.get('grades', [])
        self.subjects = cached.get('subjects', [])
        self.teachers = cached.get('teachers', {})
        self.class_groups = cached.get('class_groups', [])
        self.num_days = cached.get('num_days', 0)
        self.num_slots = cached.get('num_slots', 0)
        self.working_days = cached.get('working_days', [])
        self.time_slots = cached.get('time_slots', [])
        self.common_session = cached.get('common_session', {})
        logger.info("Loaded from cache")
    
    def _build_class_groups(self) -> List[Dict[str, Any]]:
        groups = []
        for grade in self.target_grades:
            streams = int(self.grade_streams.get(str(grade), 1))
            names = self.grade_stream_names.get(str(grade), [])
            if not names:
                names = [f"Stream {chr(65 + i)}" for i in range(streams)]
            
            for s_idx in range(streams):
                groups.append({
                    'grade': grade,
                    'stream_index': s_idx,
                    'stream_name': names[s_idx] if s_idx < len(names) else f"Stream {s_idx+1}",
                    'key': f"{grade}_{s_idx}"
                })
        return groups
    
    def _validate_basic(self) -> None:
        if not self.class_groups:
            raise ValueError("No classes defined")
        if not self.teachers:
            raise ValueError("No teachers defined")
        if not self.subjects:
            raise ValueError("No subjects defined")
    
    def _get_required_lessons(self, grade: int, subject: str) -> int:
        total = 0
        for teacher, t_data in self.teachers.items():
            if f"{teacher}|{grade}" in self.blacklist:
                continue
            for assign in t_data.get('assignments', []):
                if int(assign.get('grade', 0)) == grade and assign.get('subject') == subject:
                    total += int(assign.get('lessons', 0))
        return total
    
    def _create_variables(self) -> None:
        """Create decision variables with diagnostic tracking"""
        try:
            var_count = 0
            teacher_availability = {}
            
            for teacher, t_data in self.teachers.items():
                unavail = set(t_data.get('unavailDays', []))
                teacher_availability[teacher] = [
                    i for i, d in enumerate(self.working_days) if d not in unavail
                ]
                logger.info(f"Teacher {teacher}: available on {len(teacher_availability[teacher])} days")
            
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
                            if d_idx not in teacher_availability[teacher]:
                                continue
                            
                            self.x[ck][d_idx][slot_idx][teacher] = {}
                            
                            for assign in t_data.get('assignments', []):
                                if int(assign.get('grade', 0)) != grade:
                                    continue
                                
                                assign_stream = assign.get('streamIndex')
                                if assign_stream is not None and int(assign_stream) != s_idx:
                                    continue
                                
                                subject = assign.get('subject')
                                if not subject or f"{subject}|{grade}" in self.subject_blacklist:
                                    continue
                                
                                var_name = f'x_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}'
                                self.x[ck][d_idx][slot_idx][teacher][subject] = \
                                    self.model.NewBoolVar(var_name)
                                var_count += 1
                                self.diagnostic_counts[ck][subject] += 1
            
            # 🔍 DIAGNOSTIC: Check for zero-variable subjects
            for cg in self.class_groups:
                ck = cg['key']
                grade = cg['grade']
                for subject in self.subjects:
                    required = self._get_required_lessons(grade, subject)
                    created = self.diagnostic_counts[ck][subject]
                    if required > 0 and created == 0:
                        logger.error(f"🔴 CRITICAL: Grade {grade} {cg['stream_name']} needs {required} "
                                    f"lessons of {subject}, but ZERO variables created!")
                    elif required > 0 and created < required:
                        logger.warning(f"🟡 WARNING: Grade {grade} {cg['stream_name']} needs {required} "
                                      f"lessons of {subject}, but only {created} variables created")
            
            logger.info(f"Created {var_count} decision variables")
            self.stats.total_variables = var_count
            
        except Exception as e:
            logger.error(f"Error creating variables: {str(e)}", exc_info=True)
            raise
    
    def _add_hard_constraints(self) -> None:
        constraint_count = 0
        
        for cg in self.class_groups:
            ck = cg['key']
            for d_idx in range(self.num_days):
                for s_idx in range(self.num_slots):
                    vars_in_slot = []
                    for teacher in self.x[ck][d_idx][s_idx]:
                        vars_in_slot.extend(self.x[ck][d_idx][s_idx][teacher].values())
                    if vars_in_slot:
                        self.model.Add(sum(vars_in_slot) == 1)
                        constraint_count += 1
        
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
                        constraint_count += 1
        
        logger.info(f"Added {constraint_count} hard constraints")
        self.stats.total_constraints = constraint_count
    
    def _add_soft_constraints(self) -> None:
        for t in self.teachers:
            self.teacher_total[t] = self.model.NewIntVar(0, 10000, f'total_{t}')
        
        # Subject requirements
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
                    shortage = self.model.NewIntVar(0, required, f'shortage_{ck}_{subject}')
                    self.model.Add(sum(subject_vars) + shortage == required)
                    self.penalty_vars.append((shortage, 100000, 
                        f"Grade {grade} {cg['stream_name']}: Missing {subject}"))
        
        # Teacher daily limits
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
                        f"{teacher}: Daily overload"))
        
        # Teacher weekly limits
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
                    self.penalty_vars.append((overload, 50000, f"{teacher}: Weekly overload"))
        
        # Track totals for fairness
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
        
        logger.info(f"Added soft constraints ({len(self.penalty_vars)} penalty vars)")
    
    def _add_decision_strategy(self) -> None:
        try:
            critical_vars = []
            for cg in self.class_groups:
                ck = cg['key']
                for d_idx in range(self.num_days):
                    for s_idx in range(self.num_slots):
                        for teacher in self.x[ck][d_idx][s_idx]:
                            critical_vars.extend(self.x[ck][d_idx][s_idx][teacher].values())
            
            if critical_vars:
                self.model.AddDecisionStrategy(
                    critical_vars,
                    cp_model.CHOOSE_MIN_DOMAIN_SIZE,
                    cp_model.SELECT_MIN_VALUE
                )
                logger.info(f"Added decision strategy for {len(critical_vars)} variables")
        except Exception as e:
            logger.warning(f"Could not add decision strategy: {str(e)}")
    
    def _analyze_infeasibility(self) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        
        try:
            # Pre-compute teacher availability
            teacher_availability = {}
            for teacher, t_data in self.teachers.items():
                unavail = set(t_data.get('unavailDays', []))
                teacher_availability[teacher] = [d for d in self.working_days if d not in unavail]
            
            # Check for zero-variable subjects (from diagnostic)
            for cg in self.class_groups:
                ck = cg['key']
                grade = cg['grade']
                for subject in self.subjects:
                    required = self._get_required_lessons(grade, subject)
                    created = self.diagnostic_counts[ck][subject]
                    if required > 0 and created == 0:
                        suggestions.append(Suggestion(
                            type="no_eligible_teacher",
                            message=f"Grade {grade} {cg['stream_name']} needs {required} lessons of {subject}, "
                                    f"but NO teacher is assigned to teach it.",
                            fixes=[
                                f"Assign a teacher to teach {subject} for Grade {grade} {cg['stream_name']}",
                                f"Remove {subject} requirement for this class",
                                f"Check if {subject} is blacklisted for Grade {grade}"
                            ],
                            priority=1,
                            metadata={"grade": grade, "subject": subject}
                        ))
            
            # Check for teacher never available
            for teacher, t_data in self.teachers.items():
                if not teacher_availability.get(teacher):
                    total = sum(int(a.get('lessons', 0)) for a in t_data.get('assignments', []))
                    if total > 0:
                        suggestions.append(Suggestion(
                            type="teacher_never_available",
                            message=f"{teacher} assigned {total} lessons but unavailable all days",
                            fixes=[
                                f"Remove unavailable days for {teacher}",
                                f"Reassign {teacher}'s lessons to other teachers",
                                f"Remove {teacher}'s assignments"
                            ],
                            priority=1,
                            metadata={"teacher": teacher}
                        ))
            
            # Capacity check
            total_required = sum(
                self._get_required_lessons(cg['grade'], s)
                for cg in self.class_groups
                for s in self.subjects
            )
            total_available = len(self.class_groups) * self.num_days * self.num_slots
            
            if total_required > total_available:
                shortage = total_required - total_available
                suggestions.append(Suggestion(
                    type="capacity_overload",
                    message=f"Total lessons ({total_required}) exceed slots ({total_available}) by {shortage}",
                    fixes=[
                        f"Reduce lessons by {shortage}",
                        f"Remove a stream to free {self.num_days * self.num_slots} slots",
                        f"LAST RESORT: Add extra working day"
                    ],
                    priority=2
                ))
            
            # Teacher overload
            for teacher, t_data in self.teachers.items():
                total = sum(int(a.get('lessons', 0)) for a in t_data.get('assignments', []))
                max_weekly = t_data.get('maxLessons')
                if max_weekly and total > max_weekly:
                    overload = total - max_weekly
                    suggestions.append(Suggestion(
                        type="teacher_overload",
                        message=f"{teacher} assigned {total} lessons but max is {max_weekly}",
                        fixes=[
                            f"Reduce {teacher}'s lessons by {overload}",
                            f"Increase {teacher}'s limit to {total}",
                            f"Reassign to other teachers"
                        ],
                        priority=2,
                        metadata={"teacher": teacher}
                    ))
            
            if not suggestions:
                suggestions.append(Suggestion(
                    type="complex_conflict",
                    message="Solver could not find solution due to constraint interactions",
                    fixes=[
                        "Reduce lessons gradually to isolate conflict",
                        "Check stream-specific assignments",
                        "Verify blacklist settings",
                        "Increase solver time limit"
                    ],
                    priority=3
                ))
            
            logger.info(f"Generated {len(suggestions)} suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return [Suggestion(
                type="analysis_error",
                message=f"Error during analysis: {str(e)}",
                fixes=["Check server logs", "Verify configuration"]
            )]
    
    def solve(self, time_limit: float = 300.0, num_workers: int = 8) -> Optional[Dict[str, Any]]:
        try:
            logger.info(f"Building model ({len(self.class_groups)} classes, {len(self.teachers)} teachers)")
            
            start_time = time.time()
            
            self._add_hard_constraints()
            self._add_soft_constraints()
            self._add_decision_strategy()
            
            # Objective: minimize penalties
            total_penalty = sum(weight * var for var, weight, _ in self.penalty_vars) if self.penalty_vars else 0
            self.model.Minimize(total_penalty)
            
            self.solver = cp_model.CpSolver()
            self.solver.parameters.max_time_in_seconds = time_limit
            self.solver.parameters.num_search_workers = num_workers
            
            logger.info(f"Starting solver (timeout={time_limit}s, workers={num_workers})")
            status = self.solver.Solve(self.model)
            
            solve_time = time.time() - start_time
            self.stats.solve_time = solve_time
            self.stats.status = self.solver.StatusName(status)
            self.stats.optimal = status == cp_model.OPTIMAL
            self.stats.wall_time = self.solver.WallTime()
            self.stats.branches = self.solver.NumBranches()
            
            logger.info(f"Solver finished in {solve_time:.2f}s: {self.stats.status}")
            
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                solution = self._extract_solution()
                cache_manager.set(self.config, self.__dict__)
                return solution
            
            return None
            
        except Exception as e:
            logger.error(f"Solver error: {str(e)}", exc_info=True)
            raise
    
    def _extract_solution(self) -> Dict[str, Any]:
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
                    if (self.common_session.get('enabled') and 
                        day == self.common_session.get('day', 'FRI') and 
                        slot_idx == int(self.common_session.get('slotIndex', 0))):
                        timetable[ck]["days"][day].append(None)
                        continue
                    
                    for teacher in self.x[ck][d_idx][slot_idx]:
                        for subject, var in self.x[ck][d_idx][slot_idx][teacher].items():
                            if self.solver.Value(var):
                                cell = {"subject": subject, "teacher": teacher, "grade": cg['grade']}
                                break
                        if cell:
                            break
                    
                    timetable[ck]["days"][day].append(cell)
        return timetable
    
    def get_violations(self) -> List[Dict[str, Any]]:
        if not self.solver:
            return []
        violations = []
        for var, weight, desc in self.penalty_vars:
            value = self.solver.Value(var)
            if value > 0:
                severity = (ConstraintSeverity.CRITICAL if weight >= 100000 else
                           ConstraintSeverity.HIGH if weight >= 50000 else
                           ConstraintSeverity.MEDIUM if weight >= 10000 else
                           ConstraintSeverity.LOW)
                violations.append(Violation(
                    description=desc, severity=severity, value=value, penalty=value * weight
                ).to_dict())
        return violations


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        if not config:
            return jsonify({"success": False, "message": "Invalid JSON", "suggestions": []}), 400
        
        logger.info(f"Generation request: {len(config.get('grades', []))} grades, "
                   f"{len(config.get('teachers', {}))} teachers")
        
        solver = TimetableSolver(config, use_cache=True)
        solution = solver.solve()
        
        if solution:
            violations = solver.get_violations()
            logger.info(f"Solution found with {len(violations)} violations")
            return jsonify({
                "success": True,
                "timetable": solution,
                "violations": violations,
                "stats": solver.stats.to_dict()
            })
        else:
            suggestions = solver._analyze_infeasibility()
            logger.info(f"No solution. Generated {len(suggestions)} suggestions")
            return jsonify({
                "success": False,
                "message": "Could not generate timetable. Each suggestion is a complete solution.",
                "suggestions": [s.to_dict() for s in suggestions],
                "stats": solver.stats.to_dict()
            })
            
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({"success": False, "message": str(e), "suggestions": []}), 400
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {str(e)}", "suggestions": []}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "timestamp": time.time(), "service": "EduSchedule Pro", "version": "4.0.0"})


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("EduSchedule Pro Backend v4.0.0 Starting")
    logger.info("  ✓ Caching enabled for repeated configurations")
    logger.info("  ✓ Decision strategies for efficient search")
    logger.info("  ✓ Exhaustive diagnostic logging")
    logger.info("  ✓ Specific actionable suggestions")
    logger.info("=" * 70)
    app.run(debug=False, host='0.0.0.0', port=5000)