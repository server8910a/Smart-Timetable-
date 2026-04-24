"""
EduSchedule Pro Backend — Version 6.4.9 (Always feasible – automatic reduction)
"""

from __future__ import annotations

import json, sys, time, uuid, signal, logging, hashlib, threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from flask import Flask, Response, request, jsonify, stream_with_context
from flask_cors import CORS
from ortools.sat.python import cp_model

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Metrics ──────────────────────────────────────────────────────────
_metrics: Dict[str, int | float] = defaultdict(int)
_metrics_lock = threading.Lock()

def inc(key: str, n: int | float = 1) -> None:
    with _metrics_lock:
        _metrics[key] += n

# ── LRU Cache with TTL ──────────────────────────────────────────────
class _Entry:
    __slots__ = ("value", "ts")
    def __init__(self, value, ts):
        self.value = value; self.ts = ts

class SolutionCache:
    def __init__(self, max_size=500, ttl=3600.0):
        self._store: Dict[str, _Entry] = {}
        self._order: list[str] = []
        self._max = max_size; self._ttl = ttl
        self._lock = threading.Lock()

    def _key(self, config: Dict) -> str:
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()

    def get(self, config: Dict):
        k = self._key(config)
        with self._lock:
            e = self._store.get(k)
            if not e or (time.time() - e.ts > self._ttl):
                return None
            if k in self._order:
                self._order.remove(k); self._order.append(k)
            return e.value

    def put(self, config: Dict, result: Dict):
        k = self._key(config)
        with self._lock:
            if k in self._store:
                self._order.remove(k)
            elif len(self._store) >= self._max:
                oldest = self._order.pop(0); del self._store[oldest]
            self._store[k] = _Entry(result, time.time())
            self._order.append(k)

    def stats(self):
        with self._lock:
            return {"size": len(self._store), "max": self._max, "ttl": self._ttl}

_cache = SolutionCache()

# ── Domain types ─────────────────────────────────────────────────────
class Severity(str, Enum):
    CRITICAL = "Critical"; HIGH = "High"; MEDIUM = "Medium"; LOW = "Low"

@dataclass
class Violation:
    description: str; severity: Severity; value: int; penalty: int; coverage_pct: float = 0.0
    def to_dict(self):
        return {"description": self.description, "severity": self.severity.value,
                "value": self.value, "penalty": self.penalty, "coveragePct": round(self.coverage_pct,1)}

@dataclass
class Suggestion:
    type: str; message: str; fixes: List[str]; priority: int = 2
    effort: int = 2; impact: int = 2; metadata: Optional[Dict] = None
    def score(self) -> float:
        return (self.impact * 3.0) / self.effort - self.priority * 0.5
    def to_dict(self):
        d = {"type": self.type, "message": self.message, "fixes": self.fixes,
             "priority": self.priority, "effort": self.effort, "impact": self.impact}
        if self.metadata: d.update(self.metadata)
        return d

@dataclass
class SolverStats:
    total_variables: int = 0; total_constraints: int = 0; solve_time: float = 0.0
    status: str = "UNKNOWN"; optimal: bool = False; wall_time: float = 0.0
    branches: int = 0; objective: int = 0
    def to_dict(self):
        return {"totalVariables": self.total_variables, "totalConstraints": self.total_constraints,
                "solveTime": round(self.solve_time,3), "status": self.status,
                "optimal": self.optimal, "wallTime": round(self.wall_time,3),
                "branches": self.branches, "objective": self.objective}

# ── Validation ───────────────────────────────────────────────────────
_REQUIRED_TOP = ("grades","subjects","teachers","timeSlots","workingDays")

def validate_config(c):
    errors = []
    if not isinstance(c, dict): return False, "JSON object required", []
    for f in _REQUIRED_TOP:
        if f not in c: errors.append({"field":f,"error":f"Missing '{f}'"})
    if errors: return False, "Validation failed", errors
    if not c["grades"]: errors.append({"field":"grades","error":"Empty"})
    if not c["subjects"]: errors.append({"field":"subjects","error":"Empty"})
    if not c["teachers"]: errors.append({"field":"teachers","error":"Empty"})
    lesson_slots = [s for s in c.get("timeSlots",[]) if s.get("type")=="lesson"]
    if not lesson_slots: errors.append({"field":"timeSlots","error":"No lesson slots"})
    if not c.get("workingDays"): errors.append({"field":"workingDays","error":"Empty"})
    if errors: return False, "Validation failed", errors
    return True, None, []

def preprocess_config(c):
    c = json.loads(json.dumps(c))
    bl = set(c.get("blacklist",[])); sb = set(c.get("subjectBlacklist",[]))
    for t,td in c["teachers"].items():
        td["assignments"] = [a for a in td.get("assignments",[])
                             if f"{t}|{a.get('grade')}" not in bl
                             and f"{a.get('subject')}|{a.get('grade')}" not in sb]
    return c

# ── Schedule Index ──────────────────────────────────────────────────
@dataclass
class ScheduleIndex:
    teacher_assignments: Dict[str, List[Dict]] = field(default_factory=dict)
    required_lessons: Dict[Tuple[int,str], int] = field(default_factory=dict)
    teacher_avail_days: Dict[str, Set[int]] = field(default_factory=dict)
    var_index: Dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    teacher_var_index: Dict = field(default_factory=lambda: defaultdict(list))
    class_required: Dict[Tuple[int,int], Dict[str,int]] = field(default_factory=dict)

    def build(self, config, working_days):
        teachers = config["teachers"]
        subjects = [s[0] if isinstance(s,(list,tuple)) else s for s in config["subjects"]]
        bl = set(config.get("blacklist",[])); sb = set(config.get("subjectBlacklist",[]))
        for t,td in teachers.items():
            ua = set(td.get("unavailDays",[]))
            self.teacher_avail_days[t] = {i for i,d in enumerate(working_days) if d not in ua}
            self.teacher_assignments[t] = [a for a in td.get("assignments",[])
                                          if f"{t}|{a.get('grade')}" not in bl
                                          and f"{a.get('subject')}|{a.get('grade')}" not in sb]
        # Grade‑subject totals (overall)
        for g in sorted(int(g) for g in config["grades"]):
            for s in subjects:
                total = 0
                for t,assigns in self.teacher_assignments.items():
                    for a in assigns:
                        if int(a.get("grade",0))==g and a.get("subject")==s:
                            total += int(a.get("lessons",0))
                self.required_lessons[(g,s)] = total

        # Per‑(grade, stream_index) requirements
        self.class_required = defaultdict(lambda: defaultdict(int))
        for t,assigns in self.teacher_assignments.items():
            for a in assigns:
                g = int(a.get("grade", 0))
                si = int(a.get("streamIndex", 0))
                sub = a.get("subject")
                self.class_required[(g, si)][sub] += int(a.get("lessons", 0))

# ── Progress callback ────────────────────────────────────────────────
class SolveProgressCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, queue):
        super().__init__(); self._queue = queue; self._start = time.time()
    def on_solution_callback(self):
        self._queue.append({
            "type": "solution",
            "objective": int(self.ObjectiveValue()),
            "bound": int(self.BestObjectiveBound()),
            "elapsed": round(time.time()-self._start,2)
        })

# ── Model Builder (always feasible) ─────────────────────────────────
class ModelBuilder:
    W_MISSING_LESSON  = 500_000
    W_SPREAD          = 20_000
    W_BACK_TO_BACK    = 15_000
    W_WEEKLY_OVERLOAD = 50_000
    W_DAILY_OVERLOAD  = 10_000
    W_DAILY_PRIORITY  = 300

    def __init__(self, config, cid="?"):
        self.config = config; self.cid = cid
        self.model = cp_model.CpModel(); self.stats = SolverStats()
        r = config.get("rules",{})
        self.grades = sorted(int(g) for g in config["grades"])
        self.subjects = [s[0] if isinstance(s,(list,tuple)) else s for s in config["subjects"]]
        self.high_priority = set(config.get("highPrioritySubjects",[]))
        self.teachers = config["teachers"]
        self.time_slots = config.get("timeSlots",[])
        self.working_days = config.get("workingDays",["MON","TUE","WED","THU","FRI"])
        self.target_grades = config.get("targetGrades", self.grades)
        self.grade_streams = config.get("gradeStreams",{})
        self.grade_stream_names = config.get("gradeStreamNames",{})
        self.common_session = config.get("commonSession",{"enabled":False})
        self.global_max_per_day = int(r.get("maxTeacherPerDay",8))
        self.blacklist = set(config.get("blacklist", []))
        self.subject_blacklist = set(config.get("subjectBlacklist", []))
        self.penalise_back_to_back_subjects: Set[str] = set(r.get("noBackToBack",[]))
        self.double_lessons: Dict[str,int] = r.get("doubleLesson",{})
        self.lesson_slots = [s for s in self.time_slots if s.get("type")=="lesson"]
        self.num_slots = len(self.lesson_slots)
        self.num_days = len(self.working_days)
        if self.num_slots==0: raise ValueError("No lesson slots")
        if self.num_days==0: raise ValueError("No working days")
        self.class_groups = self._build_groups()
        self.idx = ScheduleIndex(); self.idx.build(config, self.working_days)
        self.x: Dict = {}; self.penalties: List = []; self.teacher_total: Dict = {}
        self._create_vars()

    def _build_groups(self):
        gs = []
        for g in self.target_grades:
            streams = int(self.grade_streams.get(str(g),1))
            names = self.grade_stream_names.get(str(g),[])
            for si in range(streams):
                gs.append({"grade":g,"stream_index":si,
                           "stream_name":names[si] if si<len(names) else f"Stream {chr(65+si)}",
                           "key":f"{g}_{si}"})
        return gs

    def _create_vars(self):
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
                            if int(a.get("grade",0)) != grade: continue
                            if a.get("streamIndex") is not None and int(a["streamIndex"]) != si: continue
                            sub = a.get("subject")
                            if not sub: continue
                            sv = self.x[ck][d][s].setdefault(t,{})
                            sv[sub] = self.model.NewBoolVar(f"x_{ck}_{d}_{s}_{t}_{sub}")
                            vc += 1
                            self.idx.var_index[ck][sub].append((d,s,t,sv[sub]))
                            self.idx.teacher_var_index[t].append((ck,d,s,sub,sv[sub]))
        self.stats.total_variables = vc

    def add_hard(self):
        ct = 0
        for cg in self.class_groups:
            ck = cg["key"]
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    sv = [v for tv in self.x[ck][d][s].values() for v in tv.values()]
                    if sv:
                        # Relaxed: at most one lesson per slot (free periods allowed)
                        self.model.Add(sum(sv) <= 1); ct += 1
        for t in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = [v for cg in self.class_groups for v in self.x[cg["key"]][d][s].get(t,{}).values()]
                    if tv:
                        self.model.Add(sum(tv) <= 1); ct += 1
        self.stats.total_constraints = ct

    def add_soft(self):
        # Per‑stream missing lessons
        for cg in self.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            stream_req = self.idx.class_required.get((grade, cg["stream_index"]), {})
            for sub in self.subjects:
                req = stream_req.get(sub, 0)
                if req == 0: continue
                sv = [v for _,_,_,v in self.idx.var_index[ck][sub]]
                if not sv: continue
                sh = self.model.NewIntVar(0, req, f"sh_{ck}_{sub}")
                self.model.Add(sum(sv) + sh == req)
                self.penalties.append((sh, self.W_MISSING_LESSON, f"G{grade} {sn}: missing {sub}"))

        # (High‑priority daily penalty, daily/weekly overload remain unchanged)
        # … keep all existing soft constraints …

        logger.info("Soft constraints added (%d penalty vars)", len(self.penalties))

    def set_obj(self):
        if not self.penalties: return
        pv,pw,_ = zip(*self.penalties)
        self.model.Minimize(cp_model.LinearExpr.WeightedSum(list(pv),list(pw)))

    def add_strategy(self):
        av = [v for cg in self.class_groups for d in range(self.num_days)
              for s in range(self.num_slots) for tv in self.x[cg["key"]][d][s].values() for v in tv.values()]
        if av: self.model.AddDecisionStrategy(av, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)

# (SolutionExtractor, routes, etc. remain the same; only the hard constraint changed)