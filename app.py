"""
EduSchedule Pro Backend — Version 6.1.0
Advanced Timetable Generator: Maximum Power Edition

New in v6.1:
  • Empty‑slot analysis now suggests exact teachers and days to adjust.
  • All previous v6.0 features retained.
  • Fixed logging for Python 3.14 compatibility.
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

# ──────────────────────────────────────────────────────────────────
# Logging (Python 3.14 compatible)
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────
_metrics: Dict[str, int | float] = defaultdict(int)
_metrics_lock = threading.Lock()

def inc(key: str, n: int | float = 1) -> None:
    with _metrics_lock:
        _metrics[key] += n

# ──────────────────────────────────────────────────────────────────
# LRU Cache with TTL
# ──────────────────────────────────────────────────────────────────
class _Entry:
    __slots__ = ("value", "ts")
    def __init__(self, value: Any, ts: float):
        self.value = value; self.ts = ts

class SolutionCache:
    def __init__(self, max_size: int = 500, ttl: float = 3600.0):
        self._store: Dict[str, _Entry] = {}
        self._order: list[str] = []
        self._max = max_size; self._ttl = ttl
        self._lock = threading.Lock()

    def _key(self, config: Dict) -> str:
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()

    def get(self, config: Dict) -> Optional[Dict]:
        k = self._key(config)
        with self._lock:
            e = self._store.get(k)
            if e is None: return None
            if time.time() - e.ts > self._ttl:
                self._evict(k); return None
            if k in self._order:
                self._order.remove(k); self._order.append(k)
            return e.value

    def put(self, config: Dict, result: Dict) -> None:
        k = self._key(config)
        with self._lock:
            if k in self._store: self._order.remove(k)
            elif len(self._store) >= self._max:
                oldest = self._order.pop(0); del self._store[oldest]
            self._store[k] = _Entry(result, time.time())
            self._order.append(k)

    def _evict(self, k: str) -> None:
        self._store.pop(k, None)
        if k in self._order: self._order.remove(k)

    def stats(self) -> Dict:
        with self._lock:
            return {"size": len(self._store), "max": self._max, "ttl": self._ttl}

_cache = SolutionCache()

# ──────────────────────────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────────────────────────
class Severity(str, Enum):
    CRITICAL = "Critical"; HIGH = "High"; MEDIUM = "Medium"; LOW = "Low"

@dataclass
class Violation:
    description: str; severity: Severity; value: int; penalty: int; coverage_pct: float = 0.0
    def to_dict(self) -> Dict:
        return {"description": self.description, "severity": self.severity.value,
                "value": self.value, "penalty": self.penalty, "coveragePct": round(self.coverage_pct,1)}

@dataclass
class Suggestion:
    type: str; message: str; fixes: List[str]; priority: int = 2
    effort: int = 2; impact: int = 2; metadata: Optional[Dict] = None
    def score(self) -> float: return (self.impact * 3.0) / self.effort - self.priority * 0.5
    def to_dict(self) -> Dict:
        d = {"type": self.type, "message": self.message, "fixes": self.fixes,
             "priority": self.priority, "effort": self.effort, "impact": self.impact}
        if self.metadata: d.update(self.metadata)
        return d

@dataclass
class SolverStats:
    total_variables: int = 0; total_constraints: int = 0; solve_time: float = 0.0
    status: str = "UNKNOWN"; optimal: bool = False; wall_time: float = 0.0
    branches: int = 0; objective: int = 0
    def to_dict(self) -> Dict:
        return {"totalVariables": self.total_variables, "totalConstraints": self.total_constraints,
                "solveTime": round(self.solve_time,3), "status": self.status,
                "optimal": self.optimal, "wallTime": round(self.wall_time,3),
                "branches": self.branches, "objective": self.objective}

# ──────────────────────────────────────────────────────────────────
# Config validation & preprocessing
# ──────────────────────────────────────────────────────────────────
_REQUIRED_TOP = ("grades","subjects","teachers","timeSlots","workingDays")

def validate_config(c: Any) -> Tuple[bool, Optional[str], List[Dict]]:
    errors: List[Dict] = []
    if not isinstance(c, dict): return False, "JSON object required", []
    for f in _REQUIRED_TOP:
        if f not in c: errors.append({"field":f, "error":f"Missing '{f}'"})
    if errors: return False, "Validation failed", errors
    if not c["grades"]: errors.append({"field":"grades","error":"Empty"})
    if not c["subjects"]: errors.append({"field":"subjects","error":"Empty"})
    if not c["teachers"]: errors.append({"field":"teachers","error":"Empty"})
    lesson_slots = [s for s in c.get("timeSlots",[]) if s.get("type")=="lesson"]
    if not lesson_slots: errors.append({"field":"timeSlots","error":"No lesson slots"})
    if not c.get("workingDays"): errors.append({"field":"workingDays","error":"Empty"})
    if errors: return False, "Validation failed", errors
    return True, None, []

def preprocess_config(c: Dict) -> Dict:
    c = json.loads(json.dumps(c))
    bl = set(c.get("blacklist",[]))
    sb = set(c.get("subjectBlacklist",[]))
    for t, td in c["teachers"].items():
        td["assignments"] = [a for a in td.get("assignments",[])
                             if f"{t}|{a.get('grade')}" not in bl
                             and f"{a.get('subject')}|{a.get('grade')}" not in sb]
    return c

# ──────────────────────────────────────────────────────────────────
# Schedule Index
# ──────────────────────────────────────────────────────────────────
@dataclass
class ScheduleIndex:
    teacher_assignments: Dict[str, List[Dict]] = field(default_factory=dict)
    required_lessons: Dict[Tuple[int, str], int] = field(default_factory=dict)
    teacher_avail_days: Dict[str, Set[int]] = field(default_factory=dict)
    var_index: Dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    teacher_var_index: Dict = field(default_factory=lambda: defaultdict(list))

    def build(self, config: Dict, working_days: List[str]) -> None:
        teachers = config["teachers"]
        subjects = [s[0] if isinstance(s,(list,tuple)) else s for s in config["subjects"]]
        bl = set(config.get("blacklist",[]))
        sb = set(config.get("subjectBlacklist",[]))
        for t, td in teachers.items():
            ua = set(td.get("unavailDays",[]))
            self.teacher_avail_days[t] = {i for i,d in enumerate(working_days) if d not in ua}
            self.teacher_assignments[t] = [a for a in td.get("assignments",[])
                                          if f"{t}|{a.get('grade')}" not in bl
                                          and f"{a.get('subject')}|{a.get('grade')}" not in sb]
        for g in sorted(int(g) for g in config["grades"]):
            for s in subjects:
                total = 0
                for t, assigns in self.teacher_assignments.items():
                    for a in assigns:
                        if int(a.get("grade",0))==g and a.get("subject")==s:
                            total += int(a.get("lessons",0))
                self.required_lessons[(g,s)] = total

# ──────────────────────────────────────────────────────────────────
# Progress callback for SSE
# ──────────────────────────────────────────────────────────────────
class SolveProgressCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, queue: list):
        super().__init__(); self._queue = queue; self._start = time.time()
    def on_solution_callback(self) -> None:
        self._queue.append({"type":"solution",
                            "objective": int(self.ObjectiveValue()),
                            "bound": int(self.BestObjectiveBound()),
                            "elapsed": round(time.time()-self._start,2)})

# ──────────────────────────────────────────────────────────────────
# Model Builder (v6)
# ──────────────────────────────────────────────────────────────────
class ModelBuilder:
    W_MISSING_LESSON  = 500_000
    W_SPREAD          = 20_000
    W_BACK_TO_BACK    = 15_000
    W_WEEKLY_OVERLOAD = 50_000
    W_DAILY_OVERLOAD  = 10_000

    def __init__(self, config: Dict, cid: str = "?"):
        self.config = config; self.cid = cid
        self.model = cp_model.CpModel(); self.stats = SolverStats()
        r = config.get("rules",{})
        self.grades = sorted(int(g) for g in config["grades"])
        self.subjects = [s[0] if isinstance(s,(list,tuple)) else s for s in config["subjects"]]
        self.teachers = config["teachers"]
        self.time_slots = config.get("timeSlots",[])
        self.working_days = config.get("workingDays",["MON","TUE","WED","THU","FRI"])
        self.target_grades = config.get("targetGrades", self.grades)
        self.grade_streams = config.get("gradeStreams",{})
        self.grade_stream_names = config.get("gradeStreamNames",{})
        self.common_session = config.get("commonSession",{"enabled":False})
        self.global_max_per_day = int(r.get("maxTeacherPerDay",8))
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

    def _build_groups(self) -> List[Dict]:
        gs = []
        for g in self.target_grades:
            streams = int(self.grade_streams.get(str(g),1))
            names = self.grade_stream_names.get(str(g),[])
            for si in range(streams):
                gs.append({"grade":g,"stream_index":si,
                           "stream_name":names[si] if si<len(names) else f"Stream {chr(65+si)}",
                           "key":f"{g}_{si}"})
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

    def add_hard(self) -> None:
        ct = 0
        for cg in self.class_groups:
            ck = cg["key"]
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    sv = [v for tv in self.x[ck][d][s].values() for v in tv.values()]
                    if sv:
                        self.model.Add(sum(sv) == 1); ct += 1
        for t in self.teachers:
            for d in range(self.num_days):
                for s in range(self.num_slots):
                    tv = [v for cg in self.class_groups for v in self.x[cg["key"]][d][s].get(t,{}).values()]
                    if tv:
                        self.model.Add(sum(tv) <= 1); ct += 1
        self.stats.total_constraints = ct

    def add_soft(self) -> None:
        for cg in self.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            for sub in self.subjects:
                req = self.idx.required_lessons.get((grade,sub),0)
                if req==0: continue
                sv = [v for _,_,_,v in self.idx.var_index[ck][sub]]
                if not sv: continue
                sh = self.model.NewIntVar(0,req,f"sh_{ck}_{sub}")
                self.model.Add(sum(sv)+sh==req)
                self.penalties.append((sh,self.W_MISSING_LESSON,f"G{grade} {sn}: missing {sub}"))
        for t,td in self.teachers.items():
            if td.get("isSpecial"): continue
            mpd = int(td.get("maxPerDay",self.global_max_per_day))
            daily = defaultdict(list)
            for _,d,_,_,v in self.idx.teacher_var_index[t]: daily[d].append(v)
            for d,dv in daily.items():
                if not dv: continue
                ol = self.model.NewIntVar(0,len(dv),f"dol_{t}_{d}")
                self.model.Add(sum(dv)<=mpd+ol)
                self.penalties.append((ol,self.W_DAILY_OVERLOAD,f"{t}: daily overload d{d}"))
        for t,td in self.teachers.items():
            if td.get("isSpecial"): continue
            mw = td.get("maxLessons")
            av = [v for _,_,_,_,v in self.idx.teacher_var_index[t]]
            if not av: continue
            tv = self.model.NewIntVar(0,len(av),f"tot_{t}")
            self.model.Add(tv==sum(av)); self.teacher_total[t]=tv
            if mw:
                ol = self.model.NewIntVar(0,len(av),f"wol_{t}")
                self.model.Add(sum(av)<=int(mw)+ol)
                self.penalties.append((ol,self.W_WEEKLY_OVERLOAD,f"{t}: weekly overload"))
        logger.info("Soft constraints added (%d penalty vars)", len(self.penalties))

    def set_obj(self) -> None:
        if not self.penalties: return
        pv,pw,_ = zip(*self.penalties)
        self.model.Minimize(cp_model.LinearExpr.WeightedSum(list(pv),list(pw)))

    def add_strategy(self) -> None:
        av = [v for cg in self.class_groups for d in range(self.num_days)
              for s in range(self.num_slots) for tv in self.x[cg["key"]][d][s].values() for v in tv.values()]
        if av: self.model.AddDecisionStrategy(av, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)

# ──────────────────────────────────────────────────────────────────
# Solution Extractor
# ──────────────────────────────────────────────────────────────────
class SolutionExtractor:
    def __init__(self, b: ModelBuilder, s: cp_model.CpSolver): self.b, self.s = b, s
    def extract(self) -> Dict:
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
                    slots.append(cell)
                tt[ck]["days"][day] = slots
        return tt
    def violations(self) -> List[Dict]:
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

# ──────────────────────────────────────────────────────────────────
# Infeasibility Analyser (v6.1 – improved empty slot fixes)
# ──────────────────────────────────────────────────────────────────
class InfeasibilityAnalyser:
    def __init__(self, builder: ModelBuilder): self.b = builder

    def analyse(self) -> List[Suggestion]:
        b = self.b
        suggestions: List[Suggestion] = []
        seen: Set[str] = set()

        def add(s: Suggestion) -> None:
            key = s.type+"|"+s.message
            if key not in seen: seen.add(key); suggestions.append(s)

        # ── 1. Empty slots with specific teacher-availability fixes ──
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            empty = []
            grade_teachers = set()
            for t, assigns in b.idx.teacher_assignments.items():
                if any(int(a.get("grade",0))==grade for a in assigns):
                    grade_teachers.add(t)
            for d in range(b.num_days):
                day_unavailable = [t for t in grade_teachers if d not in b.idx.teacher_avail_days.get(t,set())]
                for s in range(b.num_slots):
                    sv = [v for tv in b.x[ck][d][s].values() for v in tv.values()]
                    if not sv:
                        day  = b.working_days[d]
                        time = b.lesson_slots[s].get("time",f"Slot{s+1}") if s<len(b.lesson_slots) else f"Slot{s+1}"
                        fix_teachers = day_unavailable[:3]
                        if fix_teachers:
                            fix_str = f"Make {', '.join(fix_teachers)} available on {day}"
                        else:
                            fix_str = f"Add a new teacher for Grade {grade} or adjust availability on {day}"
                        empty.append({"slot":f"{day} {time}","fix_str":fix_str})

            if empty:
                sample = ", ".join(e["slot"] for e in empty[:4]) + ("…" if len(empty)>4 else "")
                fix_details = [e["fix_str"] for e in empty]
                add(Suggestion(
                    type="empty_slots",
                    message=f"Grade {grade} {sn}: {len(empty)} slot(s) have no eligible teacher.",
                    fixes=[
                        (f"SOLUTION A — Adjust teacher availability:\n" +
                         "\n".join(f"  • {fd}" for fd in fix_details[:5]) +
                         (f"\n  … and {len(fix_details)-5} more" if len(fix_details)>5 else ""))
                        if len(fix_details)>1 else fix_details[0],
                        f"SOLUTION B — Add a new teacher assigned to Grade {grade} {sn} with full availability",
                        f"SOLUTION C — Remove these time slots: {sample}"
                    ],
                    priority=1, effort=2, impact=3,
                    metadata={"grade":grade,"stream":sn,"emptyCount":len(empty),
                              "sampleSlots":[e["slot"] for e in empty[:4]]}
                ))

        # ── 2. No teacher for subject ──
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            for sub in b.subjects:
                req = b.idx.required_lessons.get((grade,sub),0)
                cre = len(b.idx.var_index[ck][sub])
                if req>0 and cre==0:
                    possible = [t for t,assigns in b.idx.teacher_assignments.items()
                                if any(a.get("subject")==sub for a in assigns)]
                    hint = f"Teachers who teach {sub} elsewhere: {', '.join(possible[:3]) if possible else 'none'}"
                    add(Suggestion(
                        type="no_teacher_subj",
                        message=f"Grade {grade} {sn}: No teacher assigned for {sub} ({req} lessons needed).",
                        fixes=[
                            f"SOLUTION A — Assign fix: assign one of [{', '.join(possible[:3])}] to teach {sub} for Grade {grade} {sn}" if possible else f"SOLUTION A — Hire a new teacher for {sub}",
                            f"SOLUTION B — Remove {sub} from Grade {grade}'s curriculum",
                            f"SOLUTION C — Merge fix: teach Grade {grade} {sub} alongside another grade in same slot"
                        ],
                        priority=1, effort=1, impact=3,
                        metadata={"grade":grade,"subject":sub,"required":req,"hint":hint}
                    ))

        # ── 3. Teacher never available ──
        for t,td in b.teachers.items():
            if not b.idx.teacher_avail_days.get(t,set()):
                tot = sum(int(a.get("lessons",0)) for a in b.idx.teacher_assignments.get(t,[]))
                if tot>0:
                    add(Suggestion(
                        type="teacher_unavail",
                        message=f"{t} has {tot} lessons but is unavailable on ALL working days.",
                        fixes=[
                            f"SOLUTION A — Remove unavailable days for {t}",
                            f"SOLUTION B — Reassign all {tot} lessons to other teachers",
                            f"SOLUTION C — Remove {t}'s assignments"
                        ],
                        priority=1, effort=1, impact=3,
                        metadata={"teacher":t,"totalLessons":tot}
                    ))

        # ── 4. Sole-teacher conflicts ──
        conflict_edges = 0
        for d in range(b.num_days):
            for s in range(b.num_slots):
                sole: Dict[str, List[str]] = defaultdict(list)
                for cg in b.class_groups:
                    ck = cg["key"]
                    opts = {t: list(tv.keys()) for t,tv in b.x[ck][d][s].items() if tv}
                    if len(opts)==1:
                        only_t = next(iter(opts))
                        sole[only_t].append(ck)
                for teacher, classes in sole.items():
                    if len(classes)>1:
                        conflict_edges += 1
                        class_names = []
                        for ck in classes:
                            cg_ref = next(c for c in b.class_groups if c["key"]==ck)
                            class_names.append(f"Grade {cg_ref['grade']} {cg_ref['stream_name']}")
                        day_lbl = b.working_days[d]
                        slot_lbl = b.lesson_slots[s].get("time",f"Slot{s+1}") if s<len(b.lesson_slots) else f"Slot{s+1}"
                        add(Suggestion(
                            type="sole_teacher_conflict",
                            message=f"{teacher} is the ONLY option for {len(classes)} classes at {day_lbl} {slot_lbl}: {', '.join(class_names[:3])}" + ("…" if len(class_names)>3 else ""),
                            fixes=[
                                f"SOLUTION A — Backup teacher: assign a second teacher who can cover one of [{', '.join(class_names[:2])}] at this slot",
                                f"SOLUTION B — Reassign {teacher}: move {teacher} from one of these classes",
                                f"SOLUTION C — Remove {day_lbl} from {teacher}'s schedule and redistribute"
                            ],
                            priority=1, effort=2, impact=3,
                            metadata={"teacher":teacher,"classes":class_names,"day":day_lbl,"slot":slot_lbl}
                        ))
        if conflict_edges:
            density = round(conflict_edges / max(1, b.num_days*b.num_slots) * 100, 1)
            logger.info("Conflict graph density: %.1f%%", density)

        # ── 5. Capacity overload ──
        total_req = sum(b.idx.required_lessons.values())
        total_slots = len(b.class_groups) * b.num_days * b.num_slots
        if total_req > total_slots:
            shortage = total_req - total_slots
            sub_totals = defaultdict(int)
            for (g,s),req in b.idx.required_lessons.items(): sub_totals[s] += req
            top3 = sorted(sub_totals.items(), key=lambda x:x[1], reverse=True)[:3]
            top3_str = ", ".join(f"{s}({n})" for s,n in top3)
            add(Suggestion(
                type="capacity",
                message=f"Total required lessons ({total_req}) exceed available slots ({total_slots}) by {shortage}.",
                fixes=[
                    f"SOLUTION A — Reduce lessons by {shortage}. Heaviest subjects: {top3_str}",
                    f"SOLUTION B — Add a working day (gains {len(b.class_groups)*b.num_slots} slots)",
                    f"SOLUTION C — Add a lesson slot per day (gains {len(b.class_groups)*b.num_days} slots)",
                ],
                priority=2, effort=3, impact=3
            ))

        # ── 6. Teacher overload ──
        for t,td in b.teachers.items():
            mw = td.get("maxLessons")
            if not mw: continue
            tot = sum(int(a.get("lessons",0)) for a in b.idx.teacher_assignments.get(t,[]))
            if tot <= int(mw): continue
            ov = tot - int(mw)
            t_subjects = {a.get("subject") for a in b.idx.teacher_assignments.get(t,[])}
            alternates = []
            for ot,oa in b.idx.teacher_assignments.items():
                if ot==t: continue
                os = {a.get("subject") for a in oa}
                overlap = t_subjects & os
                if not overlap: continue
                ot_tot = sum(int(a.get("lessons",0)) for a in oa)
                om = b.teachers[ot].get("maxLessons")
                if om and ot_tot < int(om):
                    cap = int(om) - ot_tot
                    alternates.append((ot,cap,", ".join(list(overlap)[:2])))
            fixes = [f"SOLUTION A — Reduce {t}'s lessons by {ov} (from {tot} → {mw})"]
            if alternates:
                ot,cap,shared = alternates[0]
                fixes.append(f"SOLUTION B — Move {min(ov,cap)} lessons to {ot} (has {cap} capacity, shares: {shared})")
            fixes.append(f"SOLUTION C — Increase {t}'s weekly max from {mw} to {tot}")
            add(Suggestion(
                type="teacher_overload",
                message=f"{t} has {tot} lessons but max is {mw} (over by {ov}).",
                fixes=fixes, priority=2, effort=1, impact=2,
                metadata={"teacher":t,"assigned":tot,"max":int(mw),"over":ov}
            ))

        # ── 7. MIS probe ──
        if not suggestions:
            suggestions.extend(self._mis_probe())

        # ── 8. Final fallback ──
        if not suggestions:
            add(Suggestion(
                type="complex",
                message="Complex interlocking constraint conflict — no single root cause identified.",
                fixes=[
                    "Gradually reduce lesson counts subject-by-subject",
                    "Temporarily remove back-to-back restrictions",
                    "Verify stream assignment indices",
                    "Try extended solver time limit (600 s)"
                ],
                priority=3, effort=3, impact=2
            ))

        suggestions.sort(key=lambda s: (-s.score(), s.priority))
        logger.info("Generated %d infeasibility suggestions", len(suggestions))
        return suggestions

    def _mis_probe(self) -> List[Suggestion]:
        b = self.b
        out = []
        def try_relaxed(extra_ub: int) -> bool:
            m = cp_model.CpModel()
            x = {}
            for cg in b.class_groups:
                ck = cg["key"]
                x[ck] = {}
                for d in range(b.num_days):
                    x[ck][d] = {}
                    for s in range(b.num_slots):
                        x[ck][d][s] = {}
                        for t,tv in b.x[ck][d][s].items():
                            for sub,_ in tv.items():
                                sv = x[ck][d][s].setdefault(t,{})
                                sv[sub] = m.NewBoolVar(f"r_{ck}_{d}_{s}_{t}_{sub}")
            for cg in b.class_groups:
                ck = cg["key"]
                for d in range(b.num_days):
                    for s in range(b.num_slots):
                        sv = [v for tv in x[ck][d][s].values() for v in tv.values()]
                        if sv: m.Add(sum(sv)==1)
            for t in b.teachers:
                for d in range(b.num_days):
                    for s in range(b.num_slots):
                        tv = [v for cg in b.class_groups for v in x[cg["key"]][d][s].get(t,{}).values()]
                        if tv: m.Add(sum(tv)<=extra_ub)
            cp = cp_model.CpSolver()
            cp.parameters.max_time_in_seconds = 5.0
            cp.parameters.num_search_workers = 4
            status = cp.Solve(m)
            return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        if try_relaxed(2):
            out.append(Suggestion(
                type="mis_teacher_conflict",
                message="Schedule becomes feasible if teachers could teach 2 classes at once → teacher availability is the bottleneck.",
                fixes=[
                    "SOLUTION A — Add more teachers (especially for peak-conflict slots)",
                    "SOLUTION B — Reduce the number of parallel streams",
                    "SOLUTION C — Stagger grade timetables so peak slots don't overlap"
                ],
                priority=1, effort=2, impact=3
            ))
        return out

# ──────────────────────────────────────────────────────────────────
# Pre‑solve analyser (instant, no CP)
# ──────────────────────────────────────────────────────────────────
def pre_solve_analyse(config: Dict) -> Dict:
    ok, msg, errors = validate_config(config)
    if not ok: return {"feasible":False,"errors":errors,"warnings":[],"summary":msg}
    config = preprocess_config(config)
    idx = ScheduleIndex()
    wd = config.get("workingDays",[])
    idx.build(config, wd)
    ls = [s for s in config.get("timeSlots",[]) if s.get("type")=="lesson"]
    grades = sorted(int(g) for g in config["grades"])
    subjects = [s[0] if isinstance(s,(list,tuple)) else s for s in config["subjects"]]
    total_slots = len(grades)*len(wd)*len(ls) if grades else 0
    total_req = sum(idx.required_lessons.values())
    warnings = []; errors_out = []
    if total_req > total_slots:
        errors_out.append({"type":"capacity","message":f"Required ({total_req}) > slots ({total_slots})"})
    for g in grades:
        for sub in subjects:
            req = idx.required_lessons.get((g,sub),0)
            has = any(int(a.get("grade",0))==g and a.get("subject")==sub
                      for assigns in idx.teacher_assignments.values() for a in assigns)
            if req>0 and not has:
                errors_out.append({"type":"no_teacher","message":f"Grade {g} needs {sub} but no teacher assigned"})
    for t,avail in idx.teacher_avail_days.items():
        if not avail:
            tot = sum(int(a.get("lessons",0)) for a in idx.teacher_assignments.get(t,[]))
            if tot>0: errors_out.append({"type":"teacher_unavail","message":f"{t} has {tot} lessons but zero available days"})
    feasible = len(errors_out)==0
    return {"feasible":feasible,"errors":errors_out,"warnings":warnings,
            "totalSlots":total_slots,"totalRequired":total_req,
            "utilizationPct":round(total_req/max(1,total_slots)*100,1),
            "summary":"OK" if feasible else f"{len(errors_out)} blocking issue(s) found"}

# ──────────────────────────────────────────────────────────────────
# Core solver runner
# ──────────────────────────────────────────────────────────────────
def run_solver(config: Dict, timeout: float=300.0, workers: int=8, cid: str="?", progress: Optional[list]=None):
    ok, err, errors = validate_config(config)
    if not ok: raise ValueError(f"{err}: {errors}")
    config = preprocess_config(config)
    builder = ModelBuilder(config, cid=cid)
    builder.add_hard(); builder.add_soft(); builder.set_obj(); builder.add_strategy()
    cp = cp_model.CpSolver()
    cp.parameters.max_time_in_seconds = timeout
    cp.parameters.num_search_workers  = workers
    cp.parameters.search_branching    = cp_model.PORTFOLIO
    t0 = time.time()
    logger.info("Solving: %d classes, %d teachers", len(builder.class_groups), len(builder.teachers))
    if progress is not None:
        cb = SolveProgressCallback(progress)
        status = cp.SolveWithSolutionCallback(builder.model, cb)
    else:
        status = cp.Solve(builder.model)
    elapsed = time.time()-t0
    stats = builder.stats
    stats.solve_time = elapsed
    stats.status = cp.StatusName(status)
    stats.optimal = (status==cp_model.OPTIMAL)
    stats.wall_time = cp.WallTime()
    stats.branches = cp.NumBranches()
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        stats.objective = int(cp.ObjectiveValue())
    logger.info("Status=%s obj=%d %.2fs", stats.status, stats.objective, elapsed)
    inc("solves_total"); inc(f"status_{stats.status}"); inc("solve_seconds", elapsed)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        ex = SolutionExtractor(builder, cp)
        return ex.extract(), stats, ex.violations(), []
    analyser = InfeasibilityAnalyser(builder)
    return None, stats, [], analyser.analyse()

# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate():
    cid = uuid.uuid4().hex[:7]
    try:
        config = request.get_json(force=True, silent=True)
        if config is None: return jsonify({"success":False,"message":"Invalid JSON","suggestions":[]}), 400
        cached = _cache.get(config)
        if cached: inc("cache_hits"); return jsonify(cached)
        inc("cache_misses")
        timeout = float(request.args.get("timeout",300))
        workers = int(request.args.get("workers",8))
        tt, stats, violations, suggestions = run_solver(config, timeout=timeout, workers=workers, cid=cid)
        if tt is not None:
            result = {"success":True,"timetable":tt,"violations":violations,"stats":stats.to_dict()}
            _cache.put(config, result); return jsonify(result)
        return jsonify({"success":False,"message":"Could not generate timetable. Each suggestion is an independent fix.",
                        "suggestions":[s.to_dict() for s in suggestions],"stats":stats.to_dict()})
    except ValueError as e: return jsonify({"success":False,"message":str(e),"suggestions":[]}), 400
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True); inc("errors")
        return jsonify({"success":False,"message":f"Server error: {e}","suggestions":[]}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        config = request.get_json(force=True, silent=True)
        if config is None: return jsonify({"error":"Invalid JSON"}), 400
        return jsonify(pre_solve_analyse(config))
    except Exception as e:
        logger.error("Analyze error: %s", e, exc_info=True)
        return jsonify({"error":str(e)}), 500

@app.route("/stream", methods=["POST"])
def stream():
    config = request.get_json(force=True, silent=True)
    if config is None: return Response("data: {\"error\": \"Invalid JSON\"}\n\n", mimetype="text/event-stream")
    cid = uuid.uuid4().hex[:7]
    timeout = float(request.args.get("timeout",300))
    workers = int(request.args.get("workers",8))
    progress = []
    def generate_events() -> Generator[str,None,None]:
        result_holder = [None]; error_holder = [None]
        def solve_thread():
            try:
                tt, stats, violations, suggestions = run_solver(config, timeout=timeout, workers=workers, cid=cid, progress=progress)
                result_holder[0] = (tt, stats, violations, suggestions)
            except Exception as e: error_holder[0] = str(e)
        t = threading.Thread(target=solve_thread, daemon=True); t.start()
        sent = 0
        while t.is_alive() or sent < len(progress):
            while sent < len(progress):
                evt = progress[sent]; yield f"event: progress\ndata: {json.dumps(evt)}\n\n"; sent += 1
            time.sleep(0.5)
        t.join()
        if error_holder[0]: yield f"event: error\ndata: {json.dumps({'message':error_holder[0]})}\n\n"; return
        tt, stats, violations, suggestions = result_holder[0]
        if tt is not None:
            payload = {"success":True,"timetable":tt,"violations":violations,"stats":stats.to_dict()}
        else:
            payload = {"success":False,"suggestions":[s.to_dict() for s in suggestions],"stats":stats.to_dict()}
        yield f"event: result\ndata: {json.dumps(payload)}\n\n"
    return Response(stream_with_context(generate_events()), mimetype="text/event-stream")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","timestamp":time.time(),"service":"EduSchedule Pro","version":"6.1.0","cache":_cache.stats()})

@app.route("/metrics", methods=["GET"])
def metrics():
    lines = ["# EduSchedule Pro metrics"]
    with _metrics_lock:
        for k,v in _metrics.items(): lines.append(f"eduscheduler_{k} {v}")
    return Response("\n".join(lines)+"\n", mimetype="text/plain")

def _on_sigterm(signum, frame): logger.info("SIGTERM received – shutting down"); sys.exit(0)
signal.signal(signal.SIGTERM, _on_sigterm)

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("EduSchedule Pro v6.1.0 — Maximum Power Edition")
    logger.info("="*60)
    app.run(debug=False, host="0.0.0.0", port=5000)