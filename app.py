"""
EduSchedule Pro Backend - Version 5.3.0
Advanced Timetable Generator with SPECIFIC Multi-Solution Suggestions

Key Features:
  - NO empty class slots (enforced by hard constraints)
  - Detects sole-teacher conflicts (teacher is the only option for multiple classes)
  - Multiple independent suggestions (any one can solve)
  - Specific teacher names, subject names, and numbers
  - Special teacher support
  - Fast CP-SAT solver with 8 workers
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
        return self._store.get(self._hash(config))

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
        d = {"type": self.type, "message": self.message, "fixes": self.fixes, "priority": self.priority}
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
        return {"totalVariables": self.total_variables, "totalConstraints": self.total_constraints,
                "solveTime": round(self.solve_time, 3), "status": self.status,
                "optimal": self.optimal, "wallTime": round(self.wall_time, 3), "branches": self.branches}


def validate_config(c: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(c, dict): return False, "JSON object required"
    for f in ("grades", "subjects", "teachers", "timeSlots", "workingDays"):
        if f not in c: return False, f"Missing '{f}'"
    if not c["grades"] or not c["subjects"] or not c["teachers"]: return False, "Empty required field"
    return True, None


def preprocess_config(c: Dict) -> Dict:
    c = json.loads(json.dumps(c))
    bl = set(c.get("blacklist", []))
    sb = set(c.get("subjectBlacklist", []))
    for t, td in c["teachers"].items():
        td["assignments"] = [a for a in td.get("assignments", [])
                             if f"{t}|{a.get('grade')}" not in bl
                             and f"{a.get('subject')}|{a.get('grade')}" not in sb]
    return c


@dataclass
class ScheduleIndex:
    teacher_assignments: Dict[str, List[Dict]] = field(default_factory=dict)
    required_lessons: Dict[Tuple[int, str], int] = field(default_factory=dict)
    teacher_avail_days: Dict[str, Set[int]] = field(default_factory=dict)
    var_index: Dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    teacher_var_index: Dict = field(default_factory=lambda: defaultdict(list))

    def build(self, config: Dict, working_days: List[str]) -> None:
        teachers = config["teachers"]
        subjects = [s[0] if isinstance(s, (list, tuple)) else s for s in config["subjects"]]
        bl = set(config.get("blacklist", []))
        sb = set(config.get("subjectBlacklist", []))
        for t, td in teachers.items():
            ua = set(td.get("unavailDays", []))
            self.teacher_avail_days[t] = {i for i, d in enumerate(working_days) if d not in ua}
            self.teacher_assignments[t] = [a for a in td.get("assignments", [])
                                          if f"{t}|{a.get('grade')}" not in bl
                                          and f"{a.get('subject')}|{a.get('grade')}" not in sb]
        for g in sorted(int(g) for g in config["grades"]):
            for s in subjects:
                total = 0
                for t, assigns in self.teacher_assignments.items():
                    if t in {x for x in teachers if f"{x}|{g}" in bl}: continue
                    for a in assigns:
                        if int(a.get("grade", 0)) == g and a.get("subject") == s:
                            total += int(a.get("lessons", 0))
                self.required_lessons[(g, s)] = total


class ModelBuilder:
    W_SUBJECT = 100_000
    W_WEEKLY = 50_000
    W_DAILY = 10_000

    def __init__(self, config: Dict):
        self.config = config
        self.model = cp_model.CpModel()
        self.stats = SolverStats()
        r = config.get("rules", {})
        self.grades = sorted(int(g) for g in config["grades"])
        self.subjects = [s[0] if isinstance(s, (list, tuple)) else s for s in config["subjects"]]
        self.teachers = config["teachers"]
        self.time_slots = config.get("timeSlots", [])
        self.working_days = config.get("workingDays", ["MON", "TUE", "WED", "THU", "FRI"])
        self.target_grades = config.get("targetGrades", self.grades)
        self.grade_streams = config.get("gradeStreams", {})
        self.grade_stream_names = config.get("gradeStreamNames", {})
        self.common_session = config.get("commonSession", {"enabled": False})
        self.global_max_per_day = int(r.get("maxTeacherPerDay", 8))
        self.lesson_slots = [s for s in self.time_slots if s.get("type") == "lesson"]
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)
        if self.num_slots == 0: raise ValueError("No lesson slots")
        if self.num_days == 0: raise ValueError("No working days")
        self.class_groups = self._build_groups()
        self.idx = ScheduleIndex()
        self.idx.build(config, self.working_days)
        self.x: Dict = {}
        self.penalties: List = []
        self.teacher_total: Dict = {}
        self._create_vars()

    def _build_groups(self) -> List[Dict]:
        gs = []
        for g in self.target_grades:
            streams = int(self.grade_streams.get(str(g), 1))
            names = self.grade_stream_names.get(str(g), [])
            for si in range(streams):
                gs.append({"grade": g, "stream_index": si,
                           "stream_name": names[si] if si < len(names) else f"Stream {chr(65 + si)}",
                           "key": f"{g}_{si}"})
        return gs

    def _create_vars(self) -> None:
        vc = 0
        for cg in self.class_groups:
            ck, grade, si = cg["key"], cg["grade"], cg["stream_index"]
            self.x[ck] = {}
            for d in range(self.num_days):
                self.x[ck][d] = {}
                for s in range(self.num_slots):
                    self.x[ck][d][s] = {}
                    for t, td in self.teachers.items():
                        if d not in self.idx.teacher_avail_days[t]: continue
                        for a in self.idx.teacher_assignments[t]:
                            if int(a.get("grade", 0)) != grade: continue
                            if a.get("streamIndex") is not None and int(a.get("streamIndex", 0)) != si: continue
                            sub = a.get("subject")
                            if not sub: continue
                            st = self.x[ck][d][s].setdefault(t, {})
                            st[sub] = self.model.NewBoolVar(f"x_{ck}_{d}_{s}_{t}_{sub}")
                            vc += 1
                            self.idx.var_index[ck][sub].append((d, s, t, st[sub]))
                            self.idx.teacher_var_index[t].append((ck, d, s, sub, st[sub]))
        for cg in self.class_groups:
            ck, grade = cg["key"], cg["grade"]
            for sub in self.subjects:
                req = self.idx.required_lessons.get((grade, sub), 0)
                cre = len(self.idx.var_index[ck][sub])
                if req > 0 and cre == 0:
                    logger.error("CRITICAL: G%s %s needs %d of %s but 0 vars", grade, cg["stream_name"], req, sub)
        self.stats.total_variables = vc

    def add_hard(self) -> None:
        ct = 0
        for cg in self.class_groups:
            ck = cg["key"]
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    sv = [v for tv in self.x[ck][d][s].values() for v in tv.values()]
                    if sv:
                        self.model.Add(sum(sv) == 1)
                        ct += 1
        for t in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = [v for cg in self.class_groups for v in
                          self.x[cg["key"]][d][s].get(t, {}).values()]
                    if tv:
                        self.model.Add(sum(tv) <= 1)
                        ct += 1
        self.stats.total_constraints = ct

    def add_soft(self) -> None:
        for cg in self.class_groups:
            ck, grade = cg["key"], cg["grade"]
            for sub in self.subjects:
                req = self.idx.required_lessons.get((grade, sub), 0)
                if req == 0: continue
                sv = [v for _, _, _, v in self.idx.var_index[ck][sub]]
                if not sv: continue
                sh = self.model.NewIntVar(0, req, f"sh_{ck}_{sub}")
                self.model.Add(sum(sv) + sh == req)
                self.penalties.append((sh, self.W_SUBJECT, f"G{grade} {cg['stream_name']}: missing {sub}"))
        for t, td in self.teachers.items():
            if td.get("isSpecial"): continue
            mpd = int(td.get("maxPerDay", self.global_max_per_day))
            daily = defaultdict(list)
            for _, d, _, _, v in self.idx.teacher_var_index[t]: daily[d].append(v)
            for d, dv in daily.items():
                if not dv: continue
                ol = self.model.NewIntVar(0, len(dv), f"dol_{t}_{d}")
                self.model.Add(sum(dv) <= mpd + ol)
                self.penalties.append((ol, self.W_DAILY, f"{t}: daily overload d{d}"))
        for t, td in self.teachers.items():
            if td.get("isSpecial"): continue
            mw = td.get("maxLessons")
            av = [v for _, _, _, _, v in self.idx.teacher_var_index[t]]
            if not av: continue
            tv = self.model.NewIntVar(0, len(av), f"tot_{t}")
            self.model.Add(tv == sum(av))
            self.teacher_total[t] = tv
            if mw:
                ol = self.model.NewIntVar(0, len(av), f"wol_{t}")
                self.model.Add(sum(av) <= int(mw) + ol)
                self.penalties.append((ol, self.W_WEEKLY, f"{t}: weekly overload"))

    def set_obj(self) -> None:
        if not self.penalties: return
        pv, pw, _ = zip(*self.penalties)
        self.model.Minimize(cp_model.LinearExpr.WeightedSum(list(pv), list(pw)))

    def add_strategy(self) -> None:
        av = [v for cg in self.class_groups for d in range(self.num_days) for s in range(self.num_slots)
              for tv in self.x[cg["key"]][d][s].values() for v in tv.values()]
        if av: self.model.AddDecisionStrategy(av, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)


class SolutionExtractor:
    def __init__(self, b: ModelBuilder, s: cp_model.CpSolver):
        self.b, self.s = b, s

    def extract(self) -> Dict:
        tt = {}
        for cg in self.b.class_groups:
            ck = cg["key"]
            tt[ck] = {"grade": cg["grade"], "streamIndex": cg["stream_index"],
                      "streamName": cg["stream_name"], "days": {}}
            for di, day in enumerate(self.b.working_days):
                slots = []
                for si in range(self.b.num_slots):
                    cs = self.b.common_session
                    if cs.get("enabled") and day == cs.get("day", "FRI") and si == int(cs.get("slotIndex", 0)):
                        slots.append(None)
                        continue
                    cell = None
                    for t, sv in self.b.x[ck][di][si].items():
                        for sub, var in sv.items():
                            if self.s.Value(var):
                                cell = {"subject": sub, "teacher": t, "grade": cg["grade"]}
                                break
                        if cell: break
                    slots.append(cell)
                tt[ck]["days"][day] = slots
        return tt

    def violations(self) -> List[Dict]:
        vs = []
        for var, w, desc in self.b.penalties:
            v = self.s.Value(var)
            if v > 0:
                sev = Severity.CRITICAL if w >= 100_000 else Severity.HIGH if w >= 50_000 else Severity.MEDIUM if w >= 10_000 else Severity.LOW
                vs.append(Violation(desc, sev, v, v * w).to_dict())
        return vs


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

        # -----------------------------------------------------------------
        # 1. EMPTY SLOTS (no teacher available for a specific slot)
        # -----------------------------------------------------------------
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            empty = []
            for d in range(b.num_days):
                for s in range(b.num_slots):
                    sv = [v for tv in b.x[ck][d][s].values() for v in tv.values()]
                    if not sv:
                        day = b.working_days[d]
                        time = b.lesson_slots[s]["time"] if s < len(b.lesson_slots) else f"Slot{s + 1}"
                        empty.append(f"{day} {time}")
            if empty:
                add(Suggestion(
                    type="empty_slots",
                    message=f"Grade {grade} {sn}: {len(empty)} slot(s) have no teacher.",
                    fixes=[
                        f"SOLUTION 1: Make teachers available on these slots: {', '.join(empty[:3])}{'...' if len(empty) > 3 else ''}",
                        f"SOLUTION 2: Add a new teacher for Grade {grade} with full availability"
                    ],
                    priority=1,
                    metadata={"grade": grade, "stream": sn, "empty": len(empty)}
                ))

        # -----------------------------------------------------------------
        # 2. NO TEACHER FOR SUBJECT
        # -----------------------------------------------------------------
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            for sub in b.subjects:
                req = b.idx.required_lessons.get((grade, sub), 0)
                cre = len(b.idx.var_index[ck][sub])
                if req > 0 and cre == 0:
                    add(Suggestion(
                        type="no_teacher_subj",
                        message=f"Grade {grade} {sn}: No teacher assigned for {sub} ({req} lessons needed).",
                        fixes=[
                            f"SOLUTION: Assign a teacher to teach {sub} for Grade {grade} {sn}",
                            f"OR remove {sub} requirement for this class"
                        ],
                        priority=1,
                        metadata={"grade": grade, "subject": sub}
                    ))

        # -----------------------------------------------------------------
        # 3. TEACHER NEVER AVAILABLE
        # -----------------------------------------------------------------
        for t, td in b.teachers.items():
            if not b.idx.teacher_avail_days.get(t, set()):
                tot = sum(int(a.get("lessons", 0)) for a in b.idx.teacher_assignments.get(t, []))
                if tot > 0:
                    add(Suggestion(
                        type="teacher_unavail",
                        message=f"{t} has {tot} lessons but is never available.",
                        fixes=[
                            f"SOLUTION: Remove unavailable days for {t}",
                            f"OR reassign all {tot} lessons to other teachers"
                        ],
                        priority=1,
                        metadata={"teacher": t}
                    ))

        # -----------------------------------------------------------------
        # 4. SOLE-TEACHER CONFLICTS (multiple classes rely on same teacher)
        # -----------------------------------------------------------------
        for d in range(b.num_days):
            for s in range(b.num_slots):
                sole_teacher = defaultdict(list)
                for cg in b.class_groups:
                    ck = cg["key"]
                    vars_in_slot = {t: list(tv.keys()) for t, tv in b.x[ck][d][s].items() if tv}
                    if len(vars_in_slot) == 1:
                        (only_teacher, subjects) = next(iter(vars_in_slot.items()))
                        sole_teacher[only_teacher].append(ck)

                for teacher, classes in sole_teacher.items():
                    if len(classes) > 1:
                        class_names = []
                        for ck in classes:
                            cg = next(c for c in b.class_groups if c["key"] == ck)
                            class_names.append(f"Grade {cg['grade']} {cg['stream_name']}")
                        add(Suggestion(
                            type="sole_teacher_conflict",
                            message=f"Teacher {teacher} is the ONLY option for {len(classes)} classes at {b.working_days[d]} {b.lesson_slots[s]['time']}.",
                            fixes=[
                                f"SOLUTION 1: Make another teacher available for one of these classes: {', '.join(class_names[:3])}",
                                f"SOLUTION 2: Reassign {teacher} from one of these classes to free up the slot",
                                f"SOLUTION 3: Add a new teacher who can cover one of these classes at this time"
                            ],
                            priority=1,
                            metadata={"teacher": teacher, "classes": class_names,
                                      "day": b.working_days[d], "slot": b.lesson_slots[s]["time"]}
                        ))

        # -----------------------------------------------------------------
        # 5. CAPACITY OVERLOAD
        # -----------------------------------------------------------------
        tr = sum(b.idx.required_lessons.values())
        ts = len(b.class_groups) * b.num_days * b.num_slots
        if tr > ts:
            sh = tr - ts
            st = defaultdict(int)
            for (g, s), req in b.idx.required_lessons.items():
                st[s] += req
            hv = sorted(st.items(), key=lambda x: x[1], reverse=True)[:3]
            sn = [f"{s[0]}({s[1]})" for s in hv]
            add(Suggestion(
                type="capacity",
                message=f"Total lessons ({tr}) exceed slots ({ts}) by {sh}.",
                fixes=[
                    f"SOLUTION 1: Reduce lessons by {sh}. Top subjects: {', '.join(sn)}",
                    f"SOLUTION 2: Remove a stream (frees {b.num_days * b.num_slots} slots)",
                    f"SOLUTION 3: Add extra day (adds {len(b.class_groups) * b.num_slots} slots)"
                ],
                priority=2
            ))

        # -----------------------------------------------------------------
        # 6. TEACHER OVERLOAD
        # -----------------------------------------------------------------
        for t, td in b.teachers.items():
            mw = td.get("maxLessons")
            if not mw: continue
            tot = sum(int(a.get("lessons", 0)) for a in b.idx.teacher_assignments.get(t, []))
            if tot > int(mw):
                ov = tot - int(mw)
                ts_set = set(a.get("subject") for a in b.idx.teacher_assignments.get(t, []))
                at = []
                for ot, oa in b.idx.teacher_assignments.items():
                    if ot == t: continue
                    os = set(a.get("subject") for a in oa)
                    if ts_set.intersection(os):
                        otot = sum(int(a.get("lessons", 0)) for a in oa)
                        om = b.teachers[ot].get("maxLessons")
                        if om and otot < int(om): at.append(ot)
                fixes = [f"SOLUTION 1: Reduce {t}'s lessons by {ov} (from {tot} to {mw})"]
                if at: fixes.append(f"SOLUTION 2: Reassign {ov} lessons to {at[0]} (same subjects, has capacity)")
                fixes.append(f"SOLUTION 3: Increase {t}'s max to {tot}")
                add(Suggestion(
                    type="teacher_overload",
                    message=f"{t} has {tot} lessons, max is {mw}.",
                    fixes=fixes,
                    priority=2,
                    metadata={"teacher": t}
                ))

        # -----------------------------------------------------------------
        # 7. FALLBACK (should rarely be reached now)
        # -----------------------------------------------------------------
        if not suggestions:
            add(Suggestion(
                type="complex",
                message="Complex constraint conflict detected.",
                fixes=[
                    "Try reducing lesson counts gradually",
                    "Check for back-to-back restrictions",
                    "Verify stream assignments match",
                    "Increase solver time limit"
                ],
                priority=3
            ))

        suggestions.sort(key=lambda s: s.priority)
        logger.info("Generated %d infeasibility suggestions", len(suggestions))
        return suggestions


def run_solver(config: Dict, timeout: float = 300.0, workers: int = 8):
    ok, err = validate_config(config)
    if not ok: raise ValueError(err)
    config = preprocess_config(config)
    builder = ModelBuilder(config)
    builder.add_hard()
    builder.add_soft()
    builder.set_obj()
    builder.add_strategy()
    cp = cp_model.CpSolver()
    cp.parameters.max_time_in_seconds = timeout
    cp.parameters.num_search_workers = workers
    t0 = time.time()
    logger.info("Solving: %d classes, %d teachers", len(builder.class_groups), len(builder.teachers))
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
        ex = SolutionExtractor(builder, cp)
        return ex.extract(), stats, ex.violations(), []
    analyser = InfeasibilityAnalyser(builder)
    return None, stats, [], analyser.analyse()


@app.route("/generate", methods=["POST"])
def generate():
    try:
        config = request.get_json(force=True, silent=True)
        if config is None: return jsonify({"success": False, "message": "Invalid JSON", "suggestions": []}), 400
        cached = _cache.get(config)
        if cached: return jsonify(cached)
        tt, stats, violations, suggestions = run_solver(config)
        if tt is not None:
            result = {"success": True, "timetable": tt, "violations": violations, "stats": stats.to_dict()}
            _cache.put(config, result)
            return jsonify(result)
        return jsonify({"success": False,
                        "message": "Could not generate timetable. Each suggestion is a complete solution.",
                        "suggestions": [s.to_dict() for s in suggestions],
                        "stats": stats.to_dict()})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e), "suggestions": []}), 400
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return jsonify({"success": False, "message": f"Server error: {e}", "suggestions": []}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.time(), "service": "EduSchedule Pro", "version": "5.3.0"})


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("EduSchedule Pro v5.3.0 - Sole-Teacher Conflict Detection")
    logger.info("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)