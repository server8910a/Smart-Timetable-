"""
EduSchedule Pro Backend — Version 6.4.4 (Precise, actionable infeasibility diagnostics)
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

# ── Model Builder ─────────────────────────────────────────────────
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
        # High‑priority subjects (need daily teaching) – taken from config
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
                        self.model.Add(sum(sv) == 1); ct += 1
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

        # Daily priority – high‑priority subjects should appear on most days
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

        # Daily overload
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

        # Weekly overload
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

# ── ULTRA‑SPECIFIC INFEASIBILITY ANALYSER ───────────────────────────
class InfeasibilityAnalyser:
    def __init__(self, builder: ModelBuilder): self.b = builder

    def analyse(self) -> List[Suggestion]:
        b = self.b
        suggestions: List[Suggestion] = []
        seen: Set[str] = set()

        def add(s: Suggestion):
            key = s.type + "|" + s.message
            if key not in seen:
                seen.add(key)
                suggestions.append(s)

        t_assign = b.idx.teacher_assignments
        t_total_lessons = {t: sum(int(a.get("lessons",0)) for a in assigns)
                           for t,assigns in t_assign.items()}
        teacher_avail_days = b.idx.teacher_avail_days

        # ── 1. Per‑class capacity (each stream must fit into weekly slots) ──
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            stream_req = b.idx.class_required.get((grade, cg["stream_index"]), {})
            total_class_lessons = sum(stream_req.values())
            max_slots_per_class = b.num_days * b.num_slots
            if total_class_lessons > max_slots_per_class:
                short = total_class_lessons - max_slots_per_class
                # Identify the heaviest subjects in this stream to suggest reduction
                top_subs = sorted(stream_req.items(), key=lambda x: x[1], reverse=True)
                top_subs_str = ", ".join(f"{s}({n})" for s,n in top_subs[:4])
                fixes = [f"SOLUTION A — Reduce total lessons for Grade {grade} {sn} from {total_class_lessons} to {max_slots_per_class} or fewer."]
                fixes.append(f"SOLUTION B — Reduce these specific subjects: {top_subs_str} (by at least {short} total).")
                fixes.append(f"SOLUTION C — Add a working day or more lesson slots per day to increase capacity.")
                add(Suggestion(
                    type="class_capacity",
                    message=f"Grade {grade} {sn} requires {total_class_lessons} lessons but has only {max_slots_per_class} available slots (overflow {short}).",
                    fixes=fixes,
                    priority=1, effort=2, impact=3,
                    metadata={"grade":grade,"stream":sn,"required":total_class_lessons,"maxSlots":max_slots_per_class}
                ))

        # ── 2. Per‑Teacher capacity (teacher total lessons vs. available slots) ──
        for t,td in b.teachers.items():
            if td.get("isSpecial"): continue
            avail_days = teacher_avail_days.get(t, set())
            # A teacher can teach at most one lesson per slot.
            total_avail_slots = len(avail_days) * b.num_slots
            total_req_t = t_total_lessons.get(t, 0)
            if total_req_t > total_avail_slots:
                ov = total_req_t - total_avail_slots
                fixes = [
                    f"SOLUTION A — Reduce {t}'s lessons by {ov} (from {total_req_t} to {total_avail_slots}).",
                    f"SOLUTION B — Make {t} available on more days (currently {len(avail_days)} out of {b.num_days}).",
                    f"SOLUTION C — Mark {t} as 'Special' to bypass this limit."
                ]
                add(Suggestion(
                    type="teacher_capacity",
                    message=f"{t} must teach {total_req_t} lessons but has only {total_avail_slots} available slots in the week.",
                    fixes=fixes,
                    priority=1, effort=1, impact=3,
                    metadata={"teacher":t,"totalLessons":total_req_t,"availableSlots":total_avail_slots}
                ))

        # ── 3. Global capacity shortage ──
        total_req = sum(b.idx.required_lessons.values())
        total_slots = len(b.class_groups) * b.num_days * b.num_slots
        if total_req > total_slots:
            shortage = total_req - total_slots
            sub_totals = defaultdict(int)
            for (g,s),req in b.idx.required_lessons.items(): sub_totals[s] += req
            top5 = sorted(sub_totals.items(), key=lambda x:x[1], reverse=True)[:5]
            top5_str = ", ".join(f"{s}({n})" for s,n in top5)
            fixes = [f"SOLUTION A — Reduce overall lessons by {shortage}. Heaviest subjects: {top5_str}"]
            for sub, n in top5[:3]:
                reduce_by = min(n, max(1, shortage // len(top5)))
                fixes.append(f"SOLUTION A{sub[0]} — Reduce {sub} by {reduce_by} lesson(s)")
            fixes.append(f"SOLUTION B — Add a working day (gains {len(b.class_groups)*b.num_slots} slots)")
            fixes.append(f"SOLUTION C — Add a lesson slot per day (gains {len(b.class_groups)*b.num_days} slots)")
            add(Suggestion(
                type="capacity_overload",
                message=f"Total required lessons ({total_req}) exceed all available slots ({total_slots}) by {shortage}.",
                fixes=fixes,
                priority=2, effort=3, impact=3,
                metadata={"totalRequired":total_req,"totalSlots":total_slots,"shortage":shortage}
            ))

        # ── 4. Empty slots (no teacher for that class at that slot) ──
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            grade_teachers = {t for t,assigns in t_assign.items() if any(int(a.get("grade",0))==grade for a in assigns)}
            empty_slots = []
            for d in range(b.num_days):
                day_unavail = [t for t in grade_teachers if d not in teacher_avail_days.get(t,set())]
                for s in range(b.num_slots):
                    sv = [v for tv in b.x[ck][d][s].values() for v in tv.values()]
                    if not sv:
                        day_name = b.working_days[d]
                        time_str = b.lesson_slots[s].get("time",f"Slot{s+1}") if s<len(b.lesson_slots) else f"Slot{s+1}"
                        empty_slots.append({
                            "day": day_name,
                            "time": time_str,
                            "fix_candidates": day_unavail[:5]
                        })
            if empty_slots:
                collect_fixes = set()
                for e in empty_slots:
                    for t in e["fix_candidates"]:
                        collect_fixes.add(f"Make {t} available on {e['day']}")
                fix_lines = list(collect_fixes)[:6]
                add(Suggestion(
                    type="empty_slots",
                    message=f"Grade {grade} {sn} has {len(empty_slots)} slot(s) with zero eligible teachers.",
                    fixes=[
                        f"SOLUTION A — Adjust availability: {'; '.join(fix_lines)}",
                        f"SOLUTION B — Add a new teacher assigned to Grade {grade} {sn} with full availability",
                        f"SOLUTION C — Remove these time slots: {', '.join(f'{e['day']} {e['time']}' for e in empty_slots[:5])}"
                    ],
                    priority=1, effort=2, impact=3,
                    metadata={"grade":grade,"stream":sn,"emptyCount":len(empty_slots)}
                ))

        # ── 5. Missing teacher for a required subject ──
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            stream_req = b.idx.class_required.get((grade, cg["stream_index"]), {})
            for sub in b.subjects:
                req = stream_req.get(sub, 0)
                cre = len(b.idx.var_index[ck][sub])
                if req > 0 and cre == 0:
                    possible = [t for t,assigns in t_assign.items() if any(a.get("subject")==sub for a in assigns)]
                    hint = possible and f"Teachers who teach {sub}: {', '.join(possible[:3])}" or "none"
                    add(Suggestion(
                        type="no_teacher_subj",
                        message=f"Grade {grade} {sn}: No teacher assigned for {sub} ({req} lessons needed).",
                        fixes=[
                            f"SOLUTION A — Assign one of [{', '.join(possible[:3])}] to teach {sub} for Grade {grade} {sn}" if possible else f"SOLUTION A — Hire a teacher for {sub}",
                            f"SOLUTION B — Remove {sub} from Grade {grade}'s curriculum",
                            f"SOLUTION C — Merge classes: teach {sub} for Grade {grade} together with another grade"
                        ],
                        priority=1, effort=1, impact=3,
                        metadata={"grade":grade,"subject":sub,"required":req}
                    ))

        # ── 6. Teacher never available at all ──
        for t,td in b.teachers.items():
            if not teacher_avail_days.get(t,set()):
                tot = t_total_lessons.get(t,0)
                if tot > 0:
                    add(Suggestion(
                        type="teacher_unavail",
                        message=f"{t} has {tot} lessons but is unavailable on all working days.",
                        fixes=[
                            f"SOLUTION A — Remove unavailable days for {t}",
                            f"SOLUTION B — Reassign {t}'s {tot} lessons to other teachers",
                            f"SOLUTION C — Remove {t}'s assignments"
                        ],
                        priority=1, effort=1, impact=3,
                        metadata={"teacher":t,"totalLessons":tot}
                    ))

        # ── 7. Teacher overload (weekly max) ──
        for t,td in b.teachers.items():
            if td.get("isSpecial"): continue
            mw = td.get("maxLessons")
            if not mw: continue
            tot = t_total_lessons.get(t,0)
            if tot > int(mw):
                ov = tot - int(mw)
                t_subjects = {a.get("subject") for a in t_assign.get(t,[])}
                alternates = []
                for ot,oa in t_assign.items():
                    if ot == t: continue
                    os = {a.get("subject") for a in oa}
                    overlap = t_subjects & os
                    if not overlap: continue
                    ot_tot = t_total_lessons[ot]
                    om = b.teachers[ot].get("maxLessons")
                    if om and ot_tot < int(om):
                        cap = int(om) - ot_tot
                        alternates.append((ot, cap, list(overlap)[:3]))
                fixes = [f"SOLUTION A — Reduce {t}'s lessons by {ov} (from {tot} → {mw})"]
                if alternates:
                    for ot, cap, shared in alternates[:3]:
                        move = min(ov, cap)
                        fixes.append(f"SOLUTION B — Move up to {move} lessons to {ot} (shares {', '.join(shared)}, has {cap} free)")
                fixes.append(f"SOLUTION C — Increase {t}'s weekly max from {mw} to {tot}")
                fixes.append(f"SOLUTION D — Mark {t} as 'Special' (no workload limits)")
                add(Suggestion(
                    type="teacher_overload",
                    message=f"{t} has {tot} lessons but weekly max is {mw}.",
                    fixes=fixes,
                    priority=2, effort=1, impact=2,
                    metadata={"teacher":t,"assigned":tot,"max":int(mw),"over":ov}
                ))

        # ── 8. Subject impossible in one week for a stream ──
        for cg in b.class_groups:
            ck,grade,sn = cg["key"],cg["grade"],cg["stream_name"]
            max_slots = b.num_days * b.num_slots
            stream_req = b.idx.class_required.get((grade, cg["stream_index"]), {})
            for sub in b.subjects:
                req = stream_req.get(sub,0)
                if req > max_slots:
                    add(Suggestion(
                        type="subject_impossible",
                        message=f"Grade {grade} {sn} requires {req} lessons of {sub}, but only {max_slots} slots exist.",
                        fixes=[f"SOLUTION A — Reduce {sub} from {req} to {max_slots} or fewer",
                               f"SOLUTION B — Add more lesson slots (current {b.num_slots}/day)"],
                        priority=1, effort=1, impact=3,
                        metadata={"grade":grade,"subject":sub,"required":req,"maxSlots":max_slots}
                    ))

        # ── 9. Daily availability bottleneck ──
        day_shortage = defaultdict(lambda: {"unavail_teachers":[], "affected_classes":0})
        for cg in b.class_groups:
            ck = cg["key"]
            for d in range(b.num_days):
                sv_count = sum(1 for s in range(b.num_slots) for tv in b.x[ck][d][s].values() if tv)
                if sv_count == 0:
                    day_shortage[d]["affected_classes"] += 1
                    grade_teachers = [t for t,assigns in t_assign.items()
                                     if any(int(a.get("grade",0))==cg["grade"] for a in assigns)]
                    for t in grade_teachers:
                        if d not in teacher_avail_days.get(t,set()):
                            if t not in day_shortage[d]["unavail_teachers"]:
                                day_shortage[d]["unavail_teachers"].append(t)

        for d, data in day_shortage.items():
            if data["affected_classes"] >= max(1, len(b.class_groups)*0.3):
                day_name = b.working_days[d]
                teachers_to_fix = data["unavail_teachers"][:5]
                if teachers_to_fix:
                    add(Suggestion(
                        type="daily_bottleneck",
                        message=f"{data['affected_classes']} classes have no coverage on {day_name} because key teachers are unavailable.",
                        fixes=[
                            f"SOLUTION A — Make {', '.join(teachers_to_fix)} available on {day_name}",
                            f"SOLUTION B — Add substitute teachers for {day_name}",
                            f"SOLUTION C — Remove {day_name} from the timetable"
                        ],
                        priority=1, effort=2, impact=3,
                        metadata={"day":day_name,"affectedClasses":data["affected_classes"],
                                  "unavailableTeachers":teachers_to_fix}
                    ))

        # ── 10. Subject schedule crowding ──
        for sub in b.subjects:
            total_lessons_sub = sum(b.idx.required_lessons.get((g,sub),0) for g in b.grades)
            avg_per_day = total_lessons_sub / (b.num_days * len(b.class_groups)) if b.class_groups else 0
            if avg_per_day > 1.5:
                add(Suggestion(
                    type="subject_crowding",
                    message=f"{sub} requires {total_lessons_sub} total lessons across all classes ({avg_per_day:.1f} per class per day).",
                    fixes=[
                        f"SOLUTION A — Reduce {sub} lessons by at least {int(total_lessons_sub*0.2)}",
                        f"SOLUTION B — Use double periods for {sub} to reduce slot pressure",
                        f"SOLUTION C — Add more lesson slots per day"
                    ],
                    priority=2, effort=2, impact=2,
                    metadata={"subject":sub,"totalLessons":total_lessons_sub,"avgPerDay":round(avg_per_day,2)}
                ))

        # ── 11. Stream merging suggestion ──
        grade_streams = defaultdict(list)
        for cg in b.class_groups:
            grade_streams[cg["grade"]].append(cg["stream_name"])
        for grade, streams in grade_streams.items():
            if len(streams) >= 2:
                grade_req = sum(b.idx.required_lessons.get((grade,sub),0) for sub in b.subjects)
                one_stream_slots = b.num_days * b.num_slots
                if grade_req <= one_stream_slots * 1.1:
                    add(Suggestion(
                        type="stream_merge",
                        message=f"Grade {grade} has {len(streams)} streams ({', '.join(streams)}), but total lessons ({grade_req}) nearly fit one stream ({one_stream_slots} slots).",
                        fixes=[
                            f"SOLUTION A — Merge streams of Grade {grade} into one class",
                            f"SOLUTION B — Reduce lessons to require only one stream",
                            f"SOLUTION C — Keep streams but reduce lesson count to relieve pressure"
                        ],
                        priority=2, effort=3, impact=2,
                        metadata={"grade":grade,"streams":streams,"required":grade_req,"oneStreamSlots":one_stream_slots}
                    ))

        # ── 12. Teacher assignment trimming ──
        for t,td in b.teachers.items():
            if td.get("isSpecial"): continue
            assigns = t_assign.get(t,[])
            unique_grades = set(int(a.get("grade",0)) for a in assigns)
            unique_subjects = set(a.get("subject") for a in assigns)
            total_lessons = t_total_lessons.get(t,0)
            if len(unique_grades) >= 3 and len(unique_subjects) >= 4 and total_lessons > 20:
                add(Suggestion(
                    type="teacher_scope_trim",
                    message=f"{t} teaches {len(unique_grades)} grades and {len(unique_subjects)} different subjects.",
                    fixes=[
                        f"SOLUTION A — Reduce {t}'s grades to 1‑2 (currently {', '.join(str(g) for g in sorted(unique_grades)[:3])})",
                        f"SOLUTION B — Reduce {t}'s subjects to 1‑2 (currently {', '.join(list(unique_subjects)[:4])})",
                        f"SOLUTION C — Make {t} a 'Special' teacher to bypass workload limits"
                    ],
                    priority=2, effort=2, impact=2,
                    metadata={"teacher":t,"grades":sorted(unique_grades),"subjects":list(unique_subjects)[:5]}
                ))

        # ── 13. Blacklist conflict ──
        for t,td in b.teachers.items():
            for cg in b.class_groups:
                grade = cg["grade"]
                subj_taught = {a.get("subject") for a in t_assign.get(t,[]) if int(a.get("grade",0))==grade}
                if subj_taught and f"{t}|{grade}" in b.blacklist:
                    add(Suggestion(
                        type="blacklist_conflict",
                        message=f"{t} is blacklisted from Grade {grade} but teaches them {', '.join(subj_taught)}.",
                        fixes=[f"SOLUTION A — Remove blacklist entry '{t}|{grade}'",
                               f"SOLUTION B — Remove {t}'s assignments for Grade {grade}"],
                        priority=1, effort=1, impact=3,
                        metadata={"teacher":t,"grade":grade}
                    ))

        # ── Fallback (should rarely trigger now) ──
        if not suggestions:
            add(Suggestion(
                type="complex",
                message="Complex constraint conflict — no single root cause identified.",
                fixes=["Increase solver timeout (try 600 seconds).",
                       "Check if a teacher is assigned to a subject they are not available for.",
                       "Try enabling 'Special' status for all teachers temporarily to see if workload limits are the cause."],
                priority=3, effort=3, impact=2
            ))

        suggestions.sort(key=lambda s: (-s.score(), s.priority))
        logger.info("Generated %d infeasibility suggestions", len(suggestions))
        return suggestions

# ── Pre‑solve analyser ──────────────────────────────────────────────
def pre_solve_analyse(config):
    ok,msg,errors = validate_config(config)
    if not ok: return {"feasible":False,"errors":errors,"warnings":[],"summary":msg}
    config = preprocess_config(config)
    idx = ScheduleIndex()
    idx.build(config, config.get("workingDays",[]))
    ls = [s for s in config.get("timeSlots",[]) if s.get("type")=="lesson"]
    grades = sorted(int(g) for g in config["grades"])
    subjects = [s[0] if isinstance(s,(list,tuple)) else s for s in config["subjects"]]
    total_slots = len(grades)*len(config["workingDays"])*len(ls) if grades else 0
    total_req = sum(idx.required_lessons.values())
    errors_out = []
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
    return {"feasible":feasible,"errors":errors_out,"warnings":[],"summary":"OK" if feasible else f"{len(errors_out)} blocking issues"}

# ── Core solver runner ──────────────────────────────────────────────
def run_solver(config, timeout=300.0, workers=8, cid="?", progress=None):
    ok,err,errors = validate_config(config)
    if not ok: raise ValueError(f"{err}: {errors}")
    config = preprocess_config(config)
    builder = ModelBuilder(config, cid=cid)
    builder.add_hard(); builder.add_soft(); builder.set_obj(); builder.add_strategy()
    cp = cp_model.CpSolver()
    cp.parameters.max_time_in_seconds = timeout
    cp.parameters.num_search_workers = workers
    t0=time.time()
    logger.info("Solving: %d classes, %d teachers", len(builder.class_groups), len(builder.teachers))
    if progress is not None:
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
        stats.objective = int(cp.ObjectiveValue())
    logger.info("Status=%s obj=%d %.2fs", stats.status, stats.objective, elapsed)
    inc("solves_total"); inc(f"status_{stats.status}"); inc("solve_seconds", elapsed)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        ex = SolutionExtractor(builder, cp)
        return ex.extract(), stats, ex.violations(), []
    analyser = InfeasibilityAnalyser(builder)
    return None, stats, [], analyser.analyse()

# ── Routes ─────────────────────────────────────────────────────────
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
        return jsonify({
            "success":False,
            "message":"Could not generate timetable. Use the suggestions below to adjust your data.",
            "suggestions":[s.to_dict() for s in suggestions],
            "stats":stats.to_dict()
        })
    except ValueError as e: return jsonify({"success":False,"message":str(e),"suggestions":[]}), 400
    except Exception as e:
        logger.error("Error: %s",e,exc_info=True); inc("errors")
        return jsonify({"success":False,"message":f"Server error: {e}","suggestions":[]}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        config = request.get_json(force=True, silent=True)
        if config is None: return jsonify({"error":"Invalid JSON"}),400
        return jsonify(pre_solve_analyse(config))
    except Exception as e:
        logger.error("Analyze error: %s",e,exc_info=True)
        return jsonify({"error":str(e)}),500

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
                result_holder[0] = run_solver(config, timeout=timeout, workers=workers, cid=cid, progress=progress)
            except Exception as e:
                error_holder[0] = str(e)
        t = threading.Thread(target=solve_thread, daemon=True); t.start()
        sent = 0
        while t.is_alive() or sent < len(progress):
            while sent < len(progress):
                evt = progress[sent]
                yield f"event: progress\ndata: {json.dumps(evt)}\n\n"
                sent += 1
            time.sleep(0.5)
        t.join()
        if error_holder[0]:
            yield f"event: error\ndata: {json.dumps({'message':error_holder[0]})}\n\n"; return
        tt, stats, violations, suggestions = result_holder[0]
        if tt is not None:
            payload = {"success":True,"timetable":tt,"violations":violations,"stats":stats.to_dict()}
        else:
            payload = {"success":False,"suggestions":[s.to_dict() for s in suggestions],"stats":stats.to_dict()}
        yield f"event: result\ndata: {json.dumps(payload)}\n\n"
    return Response(stream_with_context(generate_events()), mimetype="text/event-stream")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":"ok","timestamp":time.time(),
        "service":"EduSchedule Pro","version":"6.4.4",
        "cache":_cache.stats()
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    lines = ["# EduSchedule Pro metrics"]
    with _metrics_lock:
        for k,v in _metrics.items(): lines.append(f"eduscheduler_{k} {v}")
    return Response("\n".join(lines)+"\n", mimetype="text/plain")

def _on_sigterm(signum, frame):
    logger.info("SIGTERM received – shutting down"); sys.exit(0)
signal.signal(signal.SIGTERM, _on_sigterm)

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("EduSchedule Pro v6.4.4 – Specific per‑teacher/per‑class capacity")
    logger.info("="*60)
    app.run(debug=False, host="0.0.0.0", port=5000)