"""
EduSchedule Pro Backend — Version 7.4.2 (Never fails, auto‑adjusts, max reduction 2)
"""

from __future__ import annotations

import json, sys, time, uuid, signal, logging, hashlib, threading, copy
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
        for g in sorted(int(g) for g in config["grades"]):
            for s in subjects:
                total = 0
                for t,assigns in self.teacher_assignments.items():
                    for a in assigns:
                        if int(a.get("grade",0))==g and a.get("subject")==s:
                            total += int(a.get("lessons",0))
                self.required_lessons[(g,s)] = total

        self.class_required = defaultdict(lambda: defaultdict(int))
        for t,assigns in self.teacher_assignments.items():
            for a in assigns:
                g = int(a.get("grade", 0))
                si = int(a.get("streamIndex", 0))
                sub = a.get("subject")
                self.class_required[(g, si)][sub] += int(a.get("lessons", 0))

# ── Model Builder (unchanged) ─────────────────────────────────────
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

    def add_hard(self, relaxed=False):
        ct = 0
        for cg in self.class_groups:
            ck = cg["key"]
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    sv = [v for tv in self.x[ck][d][s].values() for v in tv.values()]
                    if sv:
                        if relaxed:
                            self.model.Add(sum(sv) <= 1)
                        else:
                            self.model.Add(sum(sv) == 1)
                        ct += 1
        for t in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = [v for cg in self.class_groups for v in self.x[cg["key"]][d][s].get(t,{}).values()]
                    if tv:
                        self.model.Add(sum(tv) <= 1); ct += 1
        self.stats.total_constraints = ct

    def add_soft(self):
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

        for cg in self.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            for sub in self.subjects:
                if sub not in self.high_priority: continue
                stream_req = self.idx.class_required.get((grade, cg["stream_index"]), {})
                req = stream_req.get(sub, 0)
                if req == 0: continue
                for d in range(self.num_days):
                    daily_vars = [v for (day,s,t,v) in self.idx.var_index[ck][sub] if day==d]
                    if not daily_vars: continue
                    taught_today = self.model.NewBoolVar(f"pri_{ck}_{sub}_{d}")
                    self.model.Add(sum(daily_vars) >= 1).OnlyEnforceIf(taught_today)
                    self.model.Add(sum(daily_vars) == 0).OnlyEnforceIf(taught_today.Not())
                    slack = self.model.NewIntVar(0, 1, f"pslack_{ck}_{sub}_{d}")
                    self.model.Add(1 - taught_today == slack)
                    self.penalties.append((slack, self.W_DAILY_PRIORITY, f"G{grade} {sn}: {sub} not daily d{d}"))

        for t,td in self.teachers.items():
            if td.get("isSpecial"): continue
            mpd = int(td.get("maxPerDay", self.global_max_per_day))
            daily = defaultdict(list)
            for _,d,_,_,v in self.idx.teacher_var_index[t]: daily[d].append(v)
            for d,dv in daily.items():
                if not dv: continue
                ol = self.model.NewIntVar(0, len(dv), f"dol_{t}_{d}")
                self.model.Add(sum(dv) <= mpd + ol)
                self.penalties.append((ol, self.W_DAILY_OVERLOAD, f"{t}: daily overload d{d}"))

        for t,td in self.teachers.items():
            if td.get("isSpecial"): continue
            mw = td.get("maxLessons")
            av = [v for _,_,_,_,v in self.idx.teacher_var_index[t]]
            if not av: continue
            tv = self.model.NewIntVar(0, len(av), f"tot_{t}")
            self.model.Add(tv == sum(av)); self.teacher_total[t] = tv
            if mw:
                ol = self.model.NewIntVar(0, len(av), f"wol_{t}")
                self.model.Add(sum(av) <= int(mw) + ol)
                self.penalties.append((ol, self.W_WEEKLY_OVERLOAD, f"{t}: weekly overload"))

        logger.info("Soft constraints added (%d penalty vars)", len(self.penalties))

    def set_obj(self):
        if not self.penalties: return
        pv,pw,_ = zip(*self.penalties)
        self.model.Minimize(cp_model.LinearExpr.WeightedSum(list(pv),list(pw)))

    def add_strategy(self):
        av = [v for cg in self.class_groups for d in range(self.num_days)
              for s in range(self.num_slots) for tv in self.x[cg["key"]][d][s].values() for v in tv.values()]
        if av: self.model.AddDecisionStrategy(av, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)

# ── Solution Extractor ───────────────────────────────────────────────
class SolutionExtractor:
    def __init__(self, b, s): self.b=b; self.s=s
    def extract(self):
        tt = {}
        for cg in self.b.class_groups:
            ck = cg["key"]
            tt[ck] = {"grade":cg["grade"],"streamIndex":cg["stream_index"],
                      "streamName":cg["stream_name"],"days":{}}
            for di,day in enumerate(self.b.working_days):
                slots = []
                for si in range(self.b.num_slots):
                    cs = self.b.common_session
                    if cs.get("enabled") and day==cs.get("day","FRI") and si==int(cs.get("slotIndex",0)):
                        slots.append(None); continue
                    cell = None
                    for t,sv in self.b.x[ck][di][si].items():
                        for sub,var in sv.items():
                            if self.s.Value(var):
                                cell = {"subject":sub,"teacher":t,"grade":cg["grade"]}; break
                        if cell: break
                    if cell is None:
                        cell = {"subject":"FREE","teacher":"","grade":cg["grade"]}
                    slots.append(cell)
                tt[ck]["days"][day] = slots
        return tt

    def violations(self):
        total_req = max(1, sum(self.b.idx.required_lessons.values()))
        vs = []
        for var,w,desc in self.b.penalties:
            v = self.s.Value(var)
            if v>0:
                sev = (Severity.CRITICAL if w>=500_000 else Severity.HIGH if w>=50_000
                       else Severity.MEDIUM if w>=10_000 else Severity.LOW)
                cov = round(v/total_req*100,1) if "missing" in desc else 0.0
                vs.append(Violation(desc,sev,v,v*w,cov).to_dict())
        return vs

# ── Feasibility check ───────────────────────────────────────────────
def is_feasible(config, timeout=5.0):
    try:
        builder = ModelBuilder(config)
        builder.add_hard(relaxed=False)
        builder.add_strategy()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout
        solver.parameters.num_search_workers = 2
        status = solver.Solve(builder.model)
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    except Exception:
        return False

# ── Auto‑Reducer (max 2 reduction, never fails) ────────────────────
def auto_reduce_config(original_config, MAX_REDUCTION=2):
    cfg = copy.deepcopy(original_config)
    high_priority = set(cfg.get("highPrioritySubjects", []))
    working_days = cfg.get("workingDays", ["MON","TUE","WED","THU","FRI"])
    num_days = len(working_days)
    lesson_slots = [s for s in cfg.get("timeSlots", []) if s.get("type") == "lesson"]
    num_slots = len(lesson_slots)
    max_slots_per_class = num_days * num_slots

    original_lessons = {}
    for t, td in cfg["teachers"].items():
        for a in td.get("assignments", []):
            key = (t, a["grade"], a.get("streamIndex", 0), a["subject"])
            original_lessons[key] = int(a["lessons"])

    reductions_log = []

    def find_assignment(teacher, grade, si, subject):
        for a in cfg["teachers"][teacher]["assignments"]:
            if int(a.get("grade",0)) == grade and int(a.get("streamIndex",0)) == si and a.get("subject") == subject:
                return a
        return None

    for _ in range(500):
        if is_feasible(cfg, timeout=5.0):
            return cfg, reductions_log, False  # feasible, no relaxation needed

        idx = ScheduleIndex()
        idx.build(cfg, working_days)

        # 1. Teacher over‑capacity
        for t, td in cfg["teachers"].items():
            if td.get("isSpecial"): continue
            avail_days = idx.teacher_avail_days.get(t, set())
            av_slots = len(avail_days) * num_slots
            total = sum(int(a.get("lessons", 0)) for a in td.get("assignments", []))
            if total > av_slots:
                candidates = []
                for a in td["assignments"]:
                    if a["lessons"] <= 1: continue
                    key = (t, a["grade"], a.get("streamIndex", 0), a["subject"])
                    orig = original_lessons.get(key, a["lessons"])
                    if (orig - a["lessons"]) < MAX_REDUCTION:
                        candidates.append((a, orig))
                if candidates:
                    candidates.sort(key=lambda x: (x[0]["subject"] in high_priority, -x[0]["lessons"]))
                    a, _ = candidates[0]
                    a["lessons"] -= 1
                    reductions_log.append(f"{a['subject']} (teacher {t}, Grade {a['grade']}): {a['lessons']+1}→{a['lessons']}")
                    break
        else:
            # 2. Class over‑capacity
            for (grade, si), reqs in idx.class_required.items():
                total = sum(reqs.values())
                if total > max_slots_per_class:
                    candidates = []
                    for sub, cnt in reqs.items():
                        if cnt == 0: continue
                        for t in cfg["teachers"]:
                            a = find_assignment(t, grade, si, sub)
                            if a and a["lessons"] > 1:
                                key = (t, grade, si, sub)
                                orig = original_lessons.get(key, a["lessons"])
                                if (orig - a["lessons"]) < MAX_REDUCTION:
                                    candidates.append((a, orig))
                                break
                    if candidates:
                        candidates.sort(key=lambda x: (x[0]["subject"] in high_priority, -x[0]["lessons"]))
                        a, _ = candidates[0]
                        a["lessons"] -= 1
                        reductions_log.append(f"{a['subject']} Grade {grade} Stream {si}: {a['lessons']+1}→{a['lessons']}")
                        break
            else:
                # 3. Global reduction of the heaviest low‑priority subject
                global_candidates = []
                for t, td in cfg["teachers"].items():
                    for a in td["assignments"]:
                        if a["lessons"] <= 1: continue
                        key = (t, a["grade"], a.get("streamIndex", 0), a["subject"])
                        orig = original_lessons.get(key, a["lessons"])
                        if (orig - a["lessons"]) < MAX_REDUCTION:
                            global_candidates.append((a, orig))
                if global_candidates:
                    global_candidates.sort(key=lambda x: (x[0]["subject"] in high_priority, -x[0]["lessons"]))
                    a, _ = global_candidates[0]
                    a["lessons"] -= 1
                    reductions_log.append(f"{a['subject']} (global): {a['lessons']+1}→{a['lessons']}")
                else:
                    break  # nothing more to reduce

    # If loop ends, still infeasible → use relaxed model later
    return cfg, reductions_log, True  # need relaxation

# ── Core runner (always returns success) ────────────────────────────
def run_solver(config, timeout=300.0, workers=8, cid="?", progress=None):
    ok,err,errors = validate_config(config)
    if not ok: raise ValueError(f"{err}: {errors}")
    original = preprocess_config(config)

    # Step 1: Try original
    builder = ModelBuilder(original, cid=cid)
    builder.add_hard(relaxed=False); builder.add_soft(); builder.set_obj(); builder.add_strategy()
    cp = cp_model.CpSolver()
    cp.parameters.max_time_in_seconds = min(timeout, 50.0)
    cp.parameters.num_search_workers = workers
    t0 = time.time()
    if progress:
        cb = SolveProgressCallback(progress)
        status = cp.Solve(builder.model, cb)
    else:
        status = cp.Solve(builder.model)
    elapsed = time.time()-t0
    stats = builder.stats
    stats.solve_time = elapsed
    stats.status = cp.StatusName(status)
    stats.optimal = (status == cp_model.OPTIMAL)
    stats.wall_time = cp.WallTime()
    stats.branches = cp.NumBranches()
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        stats.objective = int(cp.ObjectiveValue()) if cp.ObjectiveValue() != 0 else 0
        ex = SolutionExtractor(builder, cp)
        return ex.extract(), stats, ex.violations(), [], original, False

    # Step 2: Auto‑reduce (max 2 cuts)
    reduced_cfg, reductions, need_relax = auto_reduce_config(original, MAX_REDUCTION=2)
    builder2 = ModelBuilder(reduced_cfg, cid=cid)
    builder2.add_hard(relaxed=False); builder2.add_soft(); builder2.set_obj(); builder2.add_strategy()
    cp2 = cp_model.CpSolver()
    cp2.parameters.max_time_in_seconds = timeout - elapsed
    cp2.parameters.num_search_workers = workers
    t1 = time.time()
    if progress:
        cb2 = SolveProgressCallback(progress)
        status2 = cp2.Solve(builder2.model, cb2)
    else:
        status2 = cp2.Solve(builder2.model)
    elapsed2 = time.time()-t1
    stats2 = builder2.stats
    stats2.solve_time = elapsed2
    stats2.status = cp2.StatusName(status2)
    stats2.optimal = (status2 == cp_model.OPTIMAL)
    stats2.wall_time = cp2.WallTime()
    stats2.branches = cp2.NumBranches()
    if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        stats2.objective = int(cp2.ObjectiveValue()) if cp2.ObjectiveValue() != 0 else 0
        ex2 = SolutionExtractor(builder2, cp2)
        return ex2.extract(), stats2, ex2.violations(), reductions, reduced_cfg, False

    # Step 3: Relaxed model (free slots allowed) – guaranteed feasible
    logger.info("Switching to relaxed model (free slots allowed).")
    builder3 = ModelBuilder(reduced_cfg, cid=cid)
    builder3.add_hard(relaxed=True); builder3.add_soft(); builder3.set_obj(); builder3.add_strategy()
    cp3 = cp_model.CpSolver()
    cp3.parameters.max_time_in_seconds = timeout - elapsed - elapsed2
    cp3.parameters.num_search_workers = workers
    t2 = time.time()
    if progress:
        cb3 = SolveProgressCallback(progress)
        status3 = cp3.Solve(builder3.model, cb3)
    else:
        status3 = cp3.Solve(builder3.model)
    stats3 = builder3.stats
    stats3.solve_time = time.time()-t2
    stats3.status = cp3.StatusName(status3)
    stats3.optimal = (status3 == cp_model.OPTIMAL)
    stats3.wall_time = cp3.WallTime()
    stats3.branches = cp3.NumBranches()
    # This must be feasible
    stats3.objective = int(cp3.ObjectiveValue()) if cp3.ObjectiveValue() != 0 else 0
    ex3 = SolutionExtractor(builder3, cp3)
    return ex3.extract(), stats3, ex3.violations(), reductions, reduced_cfg, True  # relaxed

# ── Routes ─────────────────────────────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate():
    cid = uuid.uuid4().hex[:7]
    try:
        config = request.get_json(force=True, silent=True)
        if config is None: return jsonify({"success":False,"message":"Invalid JSON"}), 400
        cached = _cache.get(config)
        if cached: inc("cache_hits"); return jsonify(cached)
        inc("cache_misses")
        timeout = float(request.args.get("timeout",300))
        workers = int(request.args.get("workers",8))
        tt, stats, violations, reductions, final_cfg, relaxed = run_solver(config, timeout=timeout, workers=workers, cid=cid)
        # Guaranteed: tt is never None
        result = {
            "success": True,
            "timetable": tt,
            "violations": violations,
            "stats": stats.to_dict(),
            "reductions": reductions,
            "relaxed": relaxed
        }
        _cache.put(config, result)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"success":False,"message":str(e)}), 400
    except Exception as e:
        logger.error("Error: %s",e,exc_info=True); inc("errors")
        return jsonify({"success":False,"message":f"Server error: {e}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","version":"7.4.2"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)