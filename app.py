"""
EduSchedule Pro Backend — Version 6.4.8 (Precise subject‑level reductions only)
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

# ── SPECIFIC‑REDUCTION INFEASIBILITY ANALYSER (v6.4.8) ──────────────
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
        t_total_lessons = {t: sum(int(a.get("lessons",0)) for a in assigns) for t,assigns in t_assign.items()}
        teacher_avail_days = b.idx.teacher_avail_days
        num_days = b.num_days
        num_slots = b.num_slots

        # Helper to format subject reduction strings
        def subject_reduction_suggestions(per_class_subs: Dict[str, int], overage: int) -> List[str]:
            """Given {subject: current_lessons} for a class, suggest reductions totalling overage."""
            total = sum(per_class_subs.values())
            if total == 0: return []
            # Sort subjects descending by lesson count
            sorted_subs = sorted(per_class_subs.items(), key=lambda x: x[1], reverse=True)
            plan = []
            remaining = overage
            for sub, cnt in sorted_subs:
                if remaining <= 0: break
                reduce = min(cnt, remaining)
                if reduce > 0:
                    plan.append(f"Reduce {sub} from {cnt} to {cnt - reduce}")
                remaining -= reduce
            return plan

        # ── 1. Per‑class capacity overflow ──
        for cg in b.class_groups:
            grade = cg["grade"]; sn = cg["stream_name"]
            stream_req = b.idx.class_required.get((grade, cg["stream_index"]), {})
            total_class_lessons = sum(stream_req.values())
            max_slots_per_class = num_days * num_slots
            if total_class_lessons > max_slots_per_class:
                overage = total_class_lessons - max_slots_per_class
                reductions = subject_reduction_suggestions(stream_req, overage)
                fixes = [f"SOLUTION — Reduce total lessons for Grade {grade} {sn} by {overage}."]
                if reductions:
                    fixes.append("Suggested reductions: " + "; ".join(reductions))
                add(Suggestion(
                    type="class_capacity",
                    message=f"Grade {grade} {sn} requires {total_class_lessons} lessons but only {max_slots_per_class} slots available.",
                    fixes=fixes,
                    priority=1, effort=2, impact=3,
                    metadata={"grade":grade,"stream":sn}
                ))

        # ── 2. Per‑teacher weekly slot capacity ──
        for t,td in b.teachers.items():
            if td.get("isSpecial"): continue
            avail_days = teacher_avail_days.get(t, set())
            total_avail_slots = len(avail_days) * num_slots
            total_req_t = t_total_lessons.get(t, 0)
            if total_req_t > total_avail_slots:
                overage = total_req_t - total_avail_slots
                # Gather all subjects this teacher teaches, with their lesson counts
                sub_counts = defaultdict(int)
                for a in t_assign.get(t, []):
                    sub_counts[a.get("subject")] += int(a.get("lessons", 0))
                reductions = subject_reduction_suggestions(dict(sub_counts), overage)
                fixes = [f"SOLUTION — Reduce {t}'s weekly lessons by {overage}."]
                if reductions:
                    fixes.append("Suggested reductions: " + "; ".join(reductions))
                fixes.append("Alternative: Increase {t}'s available days or mark as Special (low priority).")
                add(Suggestion(
                    type="teacher_capacity",
                    message=f"{t} has {total_req_t} lesson requirements but only {total_avail_slots} available slots in the week.",
                    fixes=fixes,
                    priority=1, effort=1, impact=3,
                    metadata={"teacher":t}
                ))

        # ── 3. Global capacity ──
        total_req = sum(b.idx.required_lessons.values())
        total_slots = len(b.class_groups) * num_days * num_slots
        if total_req > total_slots:
            overage = total_req - total_slots
            # Find heaviest subjects across all classes to suggest proportional reduction
            global_sub_counts = defaultdict(int)
            for (g,s),req in b.idx.required_lessons.items(): global_sub_counts[s] += req
            reductions = subject_reduction_suggestions(dict(global_sub_counts), overage)
            fixes = [f"SOLUTION — Reduce overall weekly lessons by {overage}."]
            if reductions:
                fixes.append("Suggested reductions overall: " + "; ".join(reductions))
            add(Suggestion(
                type="global_capacity",
                message=f"Total required lessons ({total_req}) exceed all available slots ({total_slots}) by {overage}.",
                fixes=fixes,
                priority=1, effort=3, impact=3
            ))

        # ── 4. Per‑slot class‑teacher mismatch ──
        # For each day/slot, count classes that have any eligible teacher (i.e. have vars) and count distinct teachers available.
        # If #classes > #teachers, impossible. We'll suggest reducing lessons for one or more classes that contribute.
        overloaded_slots = []
        for d in range(num_days):
            day_name = b.working_days[d]
            for s in range(num_slots):
                slot_lbl = b.lesson_slots[s].get("time", f"Slot{s+1}")
                # classes that can be taught at this slot (have at least one teacher)
                classes_keys = []
                for cg in b.class_groups:
                    ck = cg["key"]
                    if ck in b.x and d in b.x[ck] and s in b.x[ck][d] and b.x[ck][d][s]:
                        classes_keys.append(ck)
                # distinct teachers available for any of these classes at this slot
                avail_teachers = set()
                for ck in classes_keys:
                    for t in b.x[ck][d][s]:
                        avail_teachers.add(t)
                num_classes = len(classes_keys)
                num_teachers = len(avail_teachers)
                if num_classes > num_teachers:
                    overloaded_slots.append((day_name, slot_lbl, num_classes, num_teachers, classes_keys))

        # Process overloaded slots, suggesting reductions in the classes involved
        if overloaded_slots:
            # For each overloaded slot, we need to reduce lessons in one or more classes.
            # We'll aggregate the heaviest subjects across all affected classes and suggest reductions.
            class_subject_map = {}  # class_key -> {subject: total lessons}
            for cg in b.class_groups:
                stream_req = b.idx.class_required.get((cg["grade"], cg["stream_index"]), {})
                class_subject_map[cg["key"]] = dict(stream_req)
            # We'll pick the first overloaded slot as example (or combine multiple)
            # Let's take the most overloaded slot.
            worst = max(overloaded_slots, key=lambda x: x[2] - x[3])
            day_name, slot_lbl, num_classes, num_teachers, involved_classes = worst
            # For involved classes, sum their total lessons
            total_lessons_involved = sum(
                sum(class_subject_map.get(ck, {}).values()) for ck in involved_classes
            )
            # The number of excess classes = num_classes - num_teachers
            excess = num_classes - num_teachers
            # We need to reduce total weekly lessons of some classes so that on this day the load drops.
            # A simple heuristic: we want to reduce total lessons across involved classes by some amount;
            # we can suggest reducing the subject with the highest lesson count in the class with the most lessons.
            # Better: list the top few subject/class combinations that can be reduced.
            suggestions_list = []
            # Gather all (class, subject, current_lessons)
            candidates = []
            for ck in involved_classes:
                cg = next((cg for cg in b.class_groups if cg["key"] == ck), None)
                if not cg: continue
                grade = cg["grade"]; sn = cg["stream_name"]
                subs = class_subject_map.get(ck, {})
                for sub, cnt in subs.items():
                    candidates.append((ck, grade, sn, sub, cnt))
            # Sort by lesson count descending
            candidates.sort(key=lambda x: x[4], reverse=True)
            # We need to reduce enough lessons to bring classes count down? Not exactly, but we can suggest reducing
            # the total weekly load so that the daily peak lowers. We'll pick enough subjects to cut at least `excess` lessons.
            reduction_plan = []
            remaining = excess
            for ck, grade, sn, sub, cnt in candidates:
                if remaining <= 0: break
                reduce = min(cnt, remaining)
                reduction_plan.append(f"Reduce {sub} in Grade {grade} {sn} from {cnt} to {cnt - reduce}")
                remaining -= reduce
            if reduction_plan:
                fixes = [
                    f"On {day_name} {slot_lbl}, {num_classes} classes require a lesson but only {num_teachers} teachers are available.",
                    f"SOLUTION — Reduce weekly lessons in the affected classes to lighten this day."
                ]
                fixes.extend(reduction_plan)
                add(Suggestion(
                    type="slot_overload",
                    message=f"Overcrowded slot: {day_name} {slot_lbl} needs {num_classes} teachers but only {num_teachers} available.",
                    fixes=fixes,
                    priority=1, effort=2, impact=3,
                    metadata={"day":day_name,"slot":slot_lbl}
                ))

        # ── 5. Empty slots (class has no teacher for that slot at all) ──
        for cg in b.class_groups:
            ck, grade, sn = cg["key"], cg["grade"], cg["stream_name"]
            empty_slots = []
            for d in range(num_days):
                for s in range(num_slots):
                    sv = [v for tv in b.x[ck][d][s].values() for v in tv.values()]
                    if not sv:
                        day_name = b.working_days[d]
                        time_str = b.lesson_slots[s].get("time", f"Slot{s+1}")
                        empty_slots.append({"day": day_name, "time": time_str})
            if empty_slots:
                # Suggest reducing lessons for that class (maybe remove a subject entirely if needed)
                stream_req = b.idx.class_required.get((grade, cg["stream_index"]), {})
                reductions = subject_reduction_suggestions(stream_req, len(empty_slots))
                fixes = [
                    f"SOLUTION — Reduce lessons for Grade {grade} {sn} to eliminate empty slots (or assign a teacher)."
                ]
                if reductions:
                    fixes.append("Subject reductions to consider: " + "; ".join(reductions))
                add(Suggestion(
                    type="empty_slots",
                    message=f"Grade {grade} {sn} has {len(empty_slots)} slot(s) with no eligible teacher.",
                    fixes=fixes,
                    priority=1, effort=2, impact=3,
                    metadata={"grade":grade,"stream":sn}
                ))

        # ── 6. Missing teacher for a required subject ──
        for cg in b.class_groups:
            grade = cg["grade"]; sn = cg["stream_name"]
            stream_req = b.idx.class_required.get((grade, cg["stream_index"]), {})
            for sub in b.subjects:
                req = stream_req.get(sub, 0)
                if req > 0 and len(b.idx.var_index[cg["key"]][sub]) == 0:
                    add(Suggestion(
                        type="no_teacher_subj",
                        message=f"Grade {grade} {sn} needs {sub} ({req} lessons) but no teacher is assigned.",
                        fixes=[
                            f"SOLUTION A — Reduce {sub} to 0 lessons.",
                            f"SOLUTION B — Assign a teacher to {sub} for Grade {grade} {sn}."
                        ],
                        priority=1, effort=1, impact=3,
                        metadata={"grade":grade,"subject":sub}
                    ))

        # ── 7. Teacher weekly overload relative to maxLessons ──
        for t,td in b.teachers.items():
            if td.get("isSpecial"): continue
            mw = td.get("maxLessons")
            if mw:
                tot = t_total_lessons.get(t, 0)
                if tot > int(mw):
                    overage = tot - int(mw)
                    sub_counts = defaultdict(int)
                    for a in t_assign.get(t, []): sub_counts[a.get("subject")] += int(a.get("lessons", 0))
                    reductions = subject_reduction_suggestions(dict(sub_counts), overage)
                    fixes = [f"SOLUTION — Reduce {t}'s lessons by {overage}."]
                    if reductions:
                        fixes.append("Suggested reductions: " + "; ".join(reductions))
                    add(Suggestion(
                        type="teacher_overload",
                        message=f"{t} has {tot} lesson requirements but weekly max is {mw}.",
                        fixes=fixes,
                        priority=1, effort=1, impact=2,
                        metadata={"teacher":t}
                    ))

        # ── 8. Fallback: if still no suggestions, try a global proportional reduction ──
        if not suggestions:
            # Calculate total teacher‑slot supply vs. demand
            total_avail_slots_all = sum(
                len(teacher_avail_days.get(t, set())) * num_slots
                for t in b.teachers if not b.teachers[t].get("isSpecial")
            )
            if total_req > total_avail_slots_all:
                overage = total_req - total_avail_slots_all
                global_sub_counts = defaultdict(int)
                for (g,s),req in b.idx.required_lessons.items(): global_sub_counts[s] += req
                reductions = subject_reduction_suggestions(dict(global_sub_counts), overage)
                fixes = [f"SOLUTION — Reduce overall weekly lessons by {overage} across all subjects."]
                if reductions:
                    fixes.append("Suggested reductions: " + "; ".join(reductions))
                add(Suggestion(
                    type="global_teacher_deficit",
                    message=f"Total required lessons ({total_req}) exceed total available teacher‑slots ({total_avail_slots_all}) by {overage}.",
                    fixes=fixes,
                    priority=1, effort=3, impact=3
                ))
            else:
                add(Suggestion(
                    type="complex",
                    message="Unable to pinpoint exact bottleneck. The problem is likely a combination of teacher daily limits and subject distribution.",
                    fixes=[
                        "Reduce the total number of lessons across all subjects by 10–20%.",
                        "Try marking all teachers as 'Special' temporarily to see if workload limits are the blocker.",
                        "Increase solver timeout to 600 seconds."
                    ],
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
            has = any(int(a.get("grade",0))==g and a.get("subject")==sub for assigns in idx.teacher_assignments.values() for a in assigns)
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
        "service":"EduSchedule Pro","version":"6.4.8",
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
    logger.info("EduSchedule Pro v6.4.8 – Precise subject‑level reductions")
    logger.info("="*60)
    app.run(debug=False, host="0.0.0.0", port=5000)