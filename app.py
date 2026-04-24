"""
EduSchedule Pro Backend - Version 5.1.0
Advanced Timetable Generator

Features:
  - Pre-computed index for O(1) lookups
  - Clean separation: ModelBuilder / SolutionExtractor / InfeasibilityAnalyser
  - Serialisable solution cache
  - Correct slot constraint (== 1 ensures no empty class slots)
  - Prioritised, deduplicated infeasibility suggestions
  - Special teacher support (no workload limits)
  - Comprehensive diagnostic logging
  - SPECIFIC suggestions with exact numbers and teacher names
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class SolutionCache:
    def __init__(self, max_size: int = 200):
        self._store: Dict[str, Dict] = {}
        self._max = max_size

    def _hash(self, config: Dict) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, config: Dict) -> Optional[Dict]:
        key = self._hash(config)
        result = self._store.get(key)
        if result:
            logger.info("Cache HIT %s", key[:12])
        return result

    def put(self, config: Dict, result: Dict) -> None:
        if len(self._store) >= self._max:
            self._store.pop(next(iter(self._store)))
        self._store[self._hash(config)] = result


_cache = SolutionCache()


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Violation:
    description: str
    severity: Severity
    value: int
    penalty: int

    def to_dict(self) -> Dict:
        return {"description": self.description, "severity": self.severity.value, "value": self.value, "penalty": self.penalty}


@dataclass
class Suggestion:
    type: str
    message: str
    fixes: List[str]
    priority: int = 2
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d: Dict = {"type": self.type, "message": self.message, "fixes": self.fixes, "priority": self.priority}
        if self.metadata:
            d.update(self.metadata)
        return d


@dataclass
class SolverStats:
    total_variables: int = 0
    total_constraints: int = 0
    solve_time: float = 0.0
    status: str = "UNKNOWN"
    optimal: bool = False
    wall_time: float = 0.0
    branches: int = 0

    def to_dict(self) -> Dict:
        return {"totalVariables": self.total_variables, "totalConstraints": self.total_constraints, "solveTime": round(self.solve_time, 3), "status": self.status, "optimal": self.optimal, "wallTime": round(self.wall_time, 3), "branches": self.branches}


_REQUIRED_FIELDS = ("grades", "subjects", "teachers", "timeSlots", "workingDays")


def validate_config(config: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(config, dict):
        return False, "Configuration must be a JSON object"
    for f in _REQUIRED_FIELDS:
        if f not in config:
            return False, f"Missing required field: '{f}'"
    if not config["grades"]:
        return False, "'grades' must not be empty"
    if not config["subjects"]:
        return False, "'subjects' must not be empty"
    if not config["teachers"]:
        return False, "'teachers' must not be empty"
    return True, None


def preprocess_config(config: Dict) -> Dict:
    cfg = json.loads(json.dumps(config))
    blacklist: Set[str] = set(cfg.get("blacklist", []))
    sub_blacklist: Set[str] = set(cfg.get("subjectBlacklist", []))
    for teacher, t_data in cfg["teachers"].items():
        t_data["assignments"] = [
            a for a in t_data.get("assignments", [])
            if f"{teacher}|{a.get('grade')}" not in blacklist
            and f"{a.get('subject')}|{a.get('grade')}" not in sub_blacklist
        ]
    return cfg


@dataclass
class ScheduleIndex:
    teacher_assignments: Dict[str, List[Dict]] = field(default_factory=dict)
    required_lessons: Dict[Tuple[int, str], int] = field(default_factory=dict)
    teacher_avail_days: Dict[str, Set[int]] = field(default_factory=dict)
    var_index: Dict[str, Dict[str, List[Tuple[int, int, str, Any]]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    teacher_var_index: Dict[str, List[Tuple[str, int, int, str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def build_pre_vars(self, config: Dict, working_days: List[str]) -> None:
        teachers = config["teachers"]
        subjects = [s[0] if isinstance(s, (list, tuple)) else s for s in config["subjects"]]
        blacklist = set(config.get("blacklist", []))
        sub_blacklist = set(config.get("subjectBlacklist", []))
        for teacher, t_data in teachers.items():
            unavail = set(t_data.get("unavailDays", []))
            self.teacher_avail_days[teacher] = {i for i, d in enumerate(working_days) if d not in unavail}
            self.teacher_assignments[teacher] = [
                a for a in t_data.get("assignments", [])
                if f"{teacher}|{a.get('grade')}" not in blacklist
                and f"{a.get('subject')}|{a.get('grade')}" not in sub_blacklist
            ]
        grades = sorted(int(g) for g in config["grades"])
        for grade in grades:
            for subject in subjects:
                total = 0
                for teacher, assigns in self.teacher_assignments.items():
                    if teacher in {t for t in teachers if f"{t}|{grade}" in blacklist}:
                        continue
                    for a in assigns:
                        if int(a.get("grade", 0)) == grade and a.get("subject") == subject:
                            total += int(a.get("lessons", 0))
                self.required_lessons[(grade, subject)] = total


class ModelBuilder:
    W_SUBJECT_SHORTAGE = 100_000
    W_WEEKLY_OVERLOAD = 50_000
    W_DAILY_OVERLOAD = 10_000

    def __init__(self, config: Dict):
        self.config = config
        self.model = cp_model.CpModel()
        self.stats = SolverStats()
        rules = config.get("rules", {})
        self.grades: List[int] = sorted(int(g) for g in config["grades"])
        self.subjects: List[str] = [s[0] if isinstance(s, (list, tuple)) else s for s in config["subjects"]]
        self.teachers: Dict[str, Dict] = config["teachers"]
        self.time_slots: List[Dict] = config.get("timeSlots", [])
        self.working_days: List[str] = config.get("workingDays", ["MON", "TUE", "WED", "THU", "FRI"])
        self.target_grades: List[int] = config.get("targetGrades", self.grades)
        self.grade_streams: Dict[str, int] = config.get("gradeStreams", {})
        self.grade_stream_names: Dict[str, List[str]] = config.get("gradeStreamNames", {})
        self.common_session: Dict = config.get("commonSession", {"enabled": False})
        self.global_max_per_day: int = int(rules.get("maxTeacherPerDay", 8))
        self.no_repeat: bool = rules.get("ruleNoRepeat") == "1"
        self.lesson_slots = [s for s in self.time_slots if s.get("type") == "lesson"]
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)
        if self.num_slots == 0:
            raise ValueError("No lesson slots defined in 'timeSlots'")
        if self.num_days == 0:
            raise ValueError("No working days defined")
        self.class_groups: List[Dict] = self._build_class_groups()
        if not self.class_groups:
            raise ValueError("No class groups could be built from target grades")
        self.idx = ScheduleIndex()
        self.idx.build_pre_vars(config, self.working_days)
        self.x: Dict = {}
        self.penalties: List[Tuple[Any, int, str]] = []
        self.teacher_total: Dict[str, Any] = {}
        self._create_variables()

    def _build_class_groups(self) -> List[Dict]:
        groups = []
        for grade in self.target_grades:
            streams = int(self.grade_streams.get(str(grade), 1))
            names = self.grade_stream_names.get(str(grade), [])
            for si in range(streams):
                groups.append({"grade": grade, "stream_index": si, "stream_name": names[si] if si < len(names) else f"Stream {chr(65+si)}", "key": f"{grade}_{si}"})
        return groups

    def _create_variables(self) -> None:
        var_count = 0
        for cg in self.class_groups:
            ck, grade, si = cg["key"], cg["grade"], cg["stream_index"]
            self.x[ck] = {}
            for d_idx in range(self.num_days):
                self.x[ck][d_idx] = {}
                for slot_idx in range(self.num_slots):
                    self.x[ck][d_idx][slot_idx] = {}
                    for teacher, t_data in self.teachers.items():
                        if d_idx not in self.idx.teacher_avail_days[teacher]:
                            continue
                        for assign in self.idx.teacher_assignments[teacher]:
                            if int(assign.get("grade", 0)) != grade:
                                continue
                            assign_stream = assign.get("streamIndex")
                            if assign_stream is not None and int(assign_stream) != si:
                                continue
                            subject = assign.get("subject")
                            if not subject:
                                continue
                            slot_teacher = self.x[ck][d_idx][slot_idx].setdefault(teacher, {})
                            var = self.model.NewBoolVar(f"x_{ck}_{d_idx}_{slot_idx}_{teacher}_{subject}")
                            slot_teacher[subject] = var
                            var_count += 1
                            self.idx.var_index[ck][subject].append((d_idx, slot_idx, teacher, var))
                            self.idx.teacher_var_index[teacher].append((ck, d_idx, slot_idx, subject, var))
        for cg in self.class_groups:
            ck, grade = cg["key"], cg["grade"]
            for subject in self.subjects:
                required = self.idx.required_lessons.get((grade, subject), 0)
                created = len(self.idx.var_index[ck][subject])
                if required > 0 and created == 0:
                    logger.error("CRITICAL: Grade %s %s needs %d lessons of %s but 0 variables created", grade, cg["stream_name"], required, subject)
                elif required > 0 and created < required:
                    logger.warning("Grade %s %s needs %d lessons of %s but only %d vars created", grade, cg["stream_name"], required, subject, created)
        logger.info("Created %d decision variables", var_count)
        self.stats.total_variables = var_count

    def add_hard_constraints(self) -> None:
        count = 0
        for cg in self.class_groups:
            ck = cg["key"]
            for d_idx in range(self.num_days):
                for slot_idx in range(self.num_slots):
                    slot_vars = [var for t_vars in self.x[ck][d_idx][slot_idx].values() for var in t_vars.values()]
                    if slot_vars:
                        self.model.Add(sum(slot_vars) == 1)
                        count += 1
                    else:
                        logger.error("CRITICAL: Grade %s %s Day %d Slot %d has NO possible teachers!", cg["grade"], cg["stream_name"], d_idx, slot_idx)
        for teacher in self.teachers:
            for d_idx in range(self.num_days):
                for slot_idx in range(self.num_slots):
                    teacher_vars = [var for cg in self.class_groups for var in self.x[cg["key"]][d_idx][slot_idx].get(teacher, {}).values()]
                    if teacher_vars:
                        self.model.Add(sum(teacher_vars) <= 1)
                        count += 1
        logger.info("Added %d hard constraints", count)
        self.stats.total_constraints = count

    def add_soft_constraints(self) -> None:
        for cg in self.class_groups:
            ck, grade = cg["key"], cg["grade"]
            for subject in self.subjects:
                required = self.idx.required_lessons.get((grade, subject), 0)
                if required == 0:
                    continue
                s_vars = [v for _, _, _, v in self.idx.var_index[ck][subject]]
                if not s_vars:
                    continue
                shortage = self.model.NewIntVar(0, required, f"shortage_{ck}_{subject}")
                self.model.Add(sum(s_vars) + shortage == required)
                self.penalties.append((shortage, self.W_SUBJECT_SHORTAGE, f"Grade {grade} {cg['stream_name']}: missing {subject} lessons"))
        for teacher, t_data in self.teachers.items():
            if t_data.get("isSpecial", False):
                continue
            max_pd = int(t_data.get("maxPerDay", self.global_max_per_day))
            daily: Dict[int, List] = defaultdict(list)
            for _ck, d_idx, _s, _subj, var in self.idx.teacher_var_index[teacher]:
                daily[d_idx].append(var)
            for d_idx, dvars in daily.items():
                if not dvars:
                    continue
                overload = self.model.NewIntVar(0, len(dvars), f"daily_ol_{teacher}_{d_idx}")
                self.model.Add(sum(dvars) <= max_pd + overload)
                self.penalties.append((overload, self.W_DAILY_OVERLOAD, f"{teacher}: daily overload day {d_idx}"))
        for teacher, t_data in self.teachers.items():
            if t_data.get("isSpecial", False):
                continue
            max_weekly = t_data.get("maxLessons")
            all_vars = [v for _, _, _, _, v in self.idx.teacher_var_index[teacher]]
            if not all_vars:
                continue
            total_var = self.model.NewIntVar(0, len(all_vars), f"total_{teacher}")
            self.model.Add(total_var == sum(all_vars))
            self.teacher_total[teacher] = total_var
            if max_weekly:
                overload = self.model.NewIntVar(0, len(all_vars), f"week_ol_{teacher}")
                self.model.Add(sum(all_vars) <= int(max_weekly) + overload)
                self.penalties.append((overload, self.W_WEEKLY_OVERLOAD, f"{teacher}: weekly overload"))
        logger.info("Soft constraints added (%d penalty vars)", len(self.penalties))

    def set_objective(self) -> None:
        if not self.penalties:
            return
        penalty_vars, weights, _ = zip(*self.penalties)
        self.model.Minimize(cp_model.LinearExpr.WeightedSum(list(penalty_vars), list(weights)))

    def add_decision_strategy(self) -> None:
        all_vars = [var for cg in self.class_groups for d_idx in range(self.num_days) for slot_idx in range(self.num_slots) for t_vars in self.x[cg["key"]][d_idx][slot_idx].values() for var in t_vars.values()]
        if all_vars:
            self.model.AddDecisionStrategy(all_vars, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)
            logger.info("Decision strategy set for %d vars", len(all_vars))


class SolutionExtractor:
    def __init__(self, builder: "ModelBuilder", cp_solver: cp_model.CpSolver):
        self.b = builder
        self.s = cp_solver

    def extract(self) -> Dict:
        b, s = self.b, self.s
        timetable = {}
        for cg in b.class_groups:
            ck = cg["key"]
            timetable[ck] = {"grade": cg["grade"], "streamIndex": cg["stream_index"], "streamName": cg["stream_name"], "days": {}}
            for d_idx, day in enumerate(b.working_days):
                slots_out = []
                for slot_idx in range(b.num_slots):
                    cs = b.common_session
                    if cs.get("enabled") and day == cs.get("day", "FRI") and slot_idx == int(cs.get("slotIndex", 0)):
                        slots_out.append(None)
                        continue
                    cell = None
                    for teacher, subj_vars in b.x[ck][d_idx][slot_idx].items():
                        for subject, var in subj_vars.items():
                            if s.Value(var):
                                cell = {"subject": subject, "teacher": teacher, "grade": cg["grade"]}
                                break
                        if cell:
                            break
                    slots_out.append(cell)
                timetable[ck]["days"][day] = slots_out
        return timetable

    def get_violations(self) -> List[Dict]:
        violations = []
        for var, weight, desc in self.b.penalties:
            val = self.s.Value(var)
            if val > 0:
                severity = Severity.CRITICAL if weight >= 100_000 else Severity.HIGH if weight >= 50_000 else Severity.MEDIUM if weight >= 10_000 else Severity.LOW
                violations.append(Violation(desc, severity, val, val * weight).to_dict())
        return violations


class InfeasibilityAnalyser:
    def __init__(self, builder: "ModelBuilder"):
        self.b = builder

    def analyse(self) -> List[Suggestion]:
        b = self.b
        suggestions: List[Suggestion] = []
        seen: Set[str] = set()

        def add(s: Suggestion) -> None:
            key = s.type + s.message
            if key not in seen:
                seen.add(key)
                suggestions.append(s)

        # =====================================================================
        # PHASE 1: HARD CONSTRAINT VIOLATIONS (Most Specific)
        # =====================================================================

        # 1a) Check for empty slots (no possible teacher for a specific slot)
        for cg in b.class_groups:
            ck, grade, stream_name = cg["key"], cg["grade"], cg["stream_name"]
            empty_slots = []
            for d_idx in range(b.num_days):
                for slot_idx in range(b.num_slots):
                    vars_in_slot = [var for t_vars in b.x[ck][d_idx][slot_idx].values() for var in t_vars.values()]
                    if not vars_in_slot:
                        day_name = b.working_days[d_idx]
                        slot_time = b.lesson_slots[slot_idx]["time"] if slot_idx < len(b.lesson_slots) else f"Slot {slot_idx+1}"
                        empty_slots.append(f"{day_name} {slot_time}")

            if empty_slots:
                add(Suggestion(
                    type="empty_slots",
                    message=f"Grade {grade} {stream_name} has {len(empty_slots)} slot(s) with NO possible teacher.",
                    fixes=[
                        f"Empty slots: {', '.join(empty_slots[:5])}{'...' if len(empty_slots) > 5 else ''}",
                        f"Ensure at least one teacher is available and assigned to teach Grade {grade} during these times",
                        f"Check teacher availability settings for Grade {grade}"
                    ],
                    priority=1,
                    metadata={"grade": grade, "stream": stream_name, "empty_slots": len(empty_slots)}
                ))

        # 1b) Per-subject variable shortage (not enough possible assignments)
        for cg in b.class_groups:
            ck, grade, stream_name = cg["key"], cg["grade"], cg["stream_name"]
            for subject in b.subjects:
                required = b.idx.required_lessons.get((grade, subject), 0)
                created = len(b.idx.var_index[ck][subject])

                if required > 0 and created == 0:
                    add(Suggestion(
                        type="no_teacher_for_subject",
                        message=f"Grade {grade} {stream_name} needs {required} lessons of {subject} but NO teacher is assigned to teach it.",
                        fixes=[
                            f"Assign a teacher to teach {subject} for Grade {grade} {stream_name}",
                            f"Remove {subject} requirement for this class",
                            f"Check if {subject} is subject-blacklisted for Grade {grade}"
                        ],
                        priority=1,
                        metadata={"grade": grade, "subject": subject, "stream": stream_name}
                    ))
                elif required > 0 and created < required:
                    add(Suggestion(
                        type="insufficient_variables",
                        message=f"Grade {grade} {stream_name} needs {required} lessons of {subject} but only {created} assignments are possible.",
                        fixes=[
                            f"Increase teacher availability for {subject} (teachers are unavailable on some days)",
                            f"Reduce required lessons for {subject} from {required} to {created}",
                            f"Add another teacher who can teach {subject} for Grade {grade}"
                        ],
                        priority=1,
                        metadata={"grade": grade, "subject": subject, "required": required, "created": created}
                    ))

        # 1c) Teacher with zero available days but has assignments
        for teacher, t_data in b.teachers.items():
            avail_days = b.idx.teacher_avail_days.get(teacher, set())
            if not avail_days:
                total_assigned = sum(int(a.get("lessons", 0)) for a in b.idx.teacher_assignments.get(teacher, []))
                if total_assigned > 0:
                    add(Suggestion(
                        type="teacher_never_available",
                        message=f"{teacher} has {total_assigned} lessons assigned but is unavailable on ALL days.",
                        fixes=[
                            f"Remove unavailable days for {teacher} (currently marked unavailable every day)",
                            f"Reassign all {total_assigned} of {teacher}'s lessons to other teachers",
                            f"Remove {teacher}'s assignments entirely"
                        ],
                        priority=1,
                        metadata={"teacher": teacher, "lessons": total_assigned}
                    ))

        # 1d) Subject-requirement mismatch (grade needs subject but no teacher for that grade)
        for subject in b.subjects:
            for grade in b.grades:
                required = b.idx.required_lessons.get((grade, subject), 0)
                if required > 0:
                    # Find teachers assigned to this subject for this grade
                    teachers_for_this = []
                    for teacher, assigns in b.idx.teacher_assignments.items():
                        for a in assigns:
                            if a.get("grade") == grade and a.get("subject") == subject:
                                teachers_for_this.append(teacher)
                                break

                    if not teachers_for_this:
                        add(Suggestion(
                            type="grade_subject_unassigned",
                            message=f"Grade {grade} needs {required} lessons of {subject} but no teacher is assigned to teach {subject} for Grade {grade}.",
                            fixes=[
                                f"Assign a teacher to teach {subject} specifically for Grade {grade}",
                                f"Remove {subject} requirement for Grade {grade}",
                                f"Add a new teacher qualified to teach {subject}"
                            ],
                            priority=1,
                            metadata={"grade": grade, "subject": subject}
                        ))

        # =====================================================================
        # PHASE 2: SOFT CONSTRAINT VIOLATIONS (Capacity, Overload)
        # =====================================================================
        if not suggestions:
            # Check only if no hard conflicts found
            total_required = sum(b.idx.required_lessons.values())
            total_slots = len(b.class_groups) * b.num_days * b.num_slots

            if total_required > total_slots:
                shortage = total_required - total_slots
                # Find subjects with highest lesson counts
                subject_totals = defaultdict(int)
                for (g, s), req in b.idx.required_lessons.items():
                    subject_totals[s] += req
                high_volume = sorted(subject_totals.items(), key=lambda x: x[1], reverse=True)[:3]
                subject_names = [f"{s[0]} ({s[1]} lessons)" for s in high_volume]

                add(Suggestion(
                    type="capacity_overload",
                    message=f"Total required lessons ({total_required}) exceed available slots ({total_slots}) by {shortage}.",
                    fixes=[
                        f"Reduce lessons by {shortage}. Highest volume subjects: {', '.join(subject_names)}",
                        f"Remove or combine a stream to free {b.num_days * b.num_slots} slots",
                        f"LAST RESORT: Add an extra working day (adds {len(b.class_groups) * b.num_slots} slots)"
                    ],
                    priority=2
                ))

            # Teacher overload
            for teacher, t_data in b.teachers.items():
                max_weekly = t_data.get("maxLessons")
                if not max_weekly:
                    continue
                total = sum(int(a.get("lessons", 0)) for a in b.idx.teacher_assignments.get(teacher, []))
                if total > int(max_weekly):
                    over = total - int(max_weekly)
                    # Find other teachers who teach same subjects
                    teacher_subjects = set()
                    for a in b.idx.teacher_assignments.get(teacher, []):
                        teacher_subjects.add(a.get("subject"))

                    available_teachers = []
                    for other_t, other_assigns in b.idx.teacher_assignments.items():
                        if other_t == teacher:
                            continue
                        other_subjects = set(a.get("subject") for a in other_assigns)
                        if teacher_subjects.intersection(other_subjects):
                            other_total = sum(int(a.get("lessons", 0)) for a in other_assigns)
                            other_max = b.teachers[other_t].get("maxLessons")
                            if other_max and other_total < int(other_max):
                                available_teachers.append(other_t)

                    fixes = [f"Reduce {teacher}'s lessons by {over} (from {total} to {max_weekly})"]
                    if available_teachers:
                        fixes.append(f"Reassign {over} lessons to {available_teachers[0]} (teaches same subjects, has capacity)")
                    fixes.append(f"LAST RESORT: Increase {teacher}'s weekly max to {total}")

                    add(Suggestion(
                        type="teacher_overload",
                        message=f"{teacher} is assigned {total} lessons but weekly maximum is {max_weekly}.",
                        fixes=fixes,
                        priority=2,
                        metadata={"teacher": teacher}
                    ))

            # Daily limit check
            for teacher, t_data in b.teachers.items():
                if t_data.get("isSpecial", False):
                    continue
                max_pd = int(t_data.get("maxPerDay", b.global_max_per_day))
                avail_days = len(b.idx.teacher_avail_days.get(teacher, set()))
                total = sum(int(a.get("lessons", 0)) for a in b.idx.teacher_assignments.get(teacher, []))

                if avail_days > 0 and total / avail_days > max_pd:
                    daily_needed = total / avail_days
                    suggested_max = int(daily_needed) + 1
                    reduction = total - (max_pd * avail_days)

                    fixes = [f"Increase {teacher}'s daily max from {max_pd} to at least {suggested_max}"]
                    if reduction > 0:
                        fixes.append(f"Reduce {teacher}'s total lessons by {reduction} (from {total} to {max_pd * avail_days})")
                    fixes.append(f"Increase {teacher}'s available days (currently {avail_days} days)")

                    add(Suggestion(
                        type="daily_limit",
                        message=f"{teacher} needs {daily_needed:.1f} lessons/day but daily maximum is {max_pd}.",
                        fixes=fixes,
                        priority=2,
                        metadata={"teacher": teacher}
                    ))

        # =====================================================================
        # PHASE 3: FALLBACK (if nothing specific was found)
        # =====================================================================
        if not suggestions:
            # Last resort - check for common session conflicts
            if b.common_session.get("enabled"):
                effective_slots = b.num_days * b.num_slots - 1
                for cg in b.class_groups:
                    for subject in b.subjects:
                        required = b.idx.required_lessons.get((cg["grade"], subject), 0)
                        if required > effective_slots:
                            add(Suggestion(
                                type="common_session_conflict",
                                message=f"Grade {cg['grade']} {cg['stream_name']} needs {required} lessons of {subject} but only {effective_slots} slots available (common session takes one).",
                                fixes=[
                                    f"Reduce {subject} lessons to {effective_slots}",
                                    f"Disable common session or move it to a break slot",
                                    f"Add an extra working day"
                                ],
                                priority=2,
                                metadata={"grade": cg["grade"], "subject": subject}
                            ))

        # Final fallback
        if not suggestions:
            add(Suggestion(
                type="complex_conflict",
                message="Solver could not find a solution; constraints interact in a complex way.",
                fixes=[
                    "Try reducing the total number of lessons gradually to isolate the conflict",
                    "Check if any teacher is assigned to multiple grades with conflicting schedules",
                    "Verify stream-specific assignment indices match correctly",
                    "Review all blacklist entries for contradictions",
                    "Increase the solver time limit (current: 300s)"
                ],
                priority=3
            ))

        suggestions.sort(key=lambda s: s.priority)
        logger.info("Generated %d infeasibility suggestions", len(suggestions))
        return suggestions


def run_solver(config: Dict, time_limit: float = 300.0, num_workers: int = 8) -> Tuple[Optional[Dict], SolverStats, List[Dict], List[Dict]]:
    ok, err = validate_config(config)
    if not ok:
        raise ValueError(err)
    config = preprocess_config(config)
    builder = ModelBuilder(config)
    builder.add_hard_constraints()
    builder.add_soft_constraints()
    builder.set_objective()
    builder.add_decision_strategy()
    cp = cp_model.CpSolver()
    cp.parameters.max_time_in_seconds = time_limit
    cp.parameters.num_search_workers = num_workers
    t0 = time.time()
    logger.info("Solving: %d classes, %d teachers, %d slots/day, %d days", len(builder.class_groups), len(builder.teachers), builder.num_slots, builder.num_days)
    status = cp.Solve(builder.model)
    elapsed = time.time() - t0
    stats = builder.stats
    stats.solve_time = elapsed
    stats.status = cp.StatusName(status)
    stats.optimal = status == cp_model.OPTIMAL
    stats.wall_time = cp.WallTime()
    stats.branches = cp.NumBranches()
    logger.info("Status: %s in %.2fs", stats.status, elapsed)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        extractor = SolutionExtractor(builder, cp)
        timetable = extractor.extract()
        violations = extractor.get_violations()
        return timetable, stats, violations, []
    analyser = InfeasibilityAnalyser(builder)
    suggestions = analyser.analyse()
    return None, stats, [], suggestions


@app.route("/generate", methods=["POST"])
def generate():
    try:
        config = request.get_json(force=True, silent=True)
        if config is None:
            return jsonify({"success": False, "message": "Invalid or missing JSON body", "suggestions": []}), 400
        cached = _cache.get(config)
        if cached:
            return jsonify(cached)
        timetable, stats, violations, suggestions = run_solver(config)
        if timetable is not None:
            result = {"success": True, "timetable": timetable, "violations": violations, "stats": stats.to_dict()}
            _cache.put(config, result)
            return jsonify(result)
        return jsonify({"success": False, "message": "Could not generate a timetable. Review the suggestions below.", "suggestions": [s.to_dict() for s in suggestions], "stats": stats.to_dict()})
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"success": False, "message": str(exc), "suggestions": []}), 400
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {exc}", "suggestions": []}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.time(), "service": "EduSchedule Pro", "version": "5.1.0"})


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("EduSchedule Pro Backend v5.1.0")
    logger.info("  - No empty class slots (== 1 constraint)")
    logger.info("  - Special teacher support")
    logger.info("  - SPECIFIC infeasibility suggestions with exact details")
    logger.info("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)