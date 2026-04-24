#!/usr/bin/env python3
"""
School Timetable Fixer – Corrects teacher capacity calculation.
Never suggests reducing all subjects to zero unless there are truly zero teachers.
"""

from typing import Dict, List, Tuple

# ============================================================
# CONFIGURATION – EDIT THIS TO MATCH YOUR SCHOOL
# ============================================================

# Required weekly lessons per subject
required_lessons: Dict[str, int] = {
    "MATHEMATICS": 30,
    "ENGLISH": 30,
    "CREATIVE ARTS": 30,
    "KISWAHILI": 24,
    "SCIENCE": 24,
    "CRE": 24,
    "AGRICULTURE": 24,
    "SOCIAL STUDIES": 24,
    "PRETECHNICAL": 12,
}

# List of teachers: (name, max_lessons_per_week)
teachers: List[Tuple[str, int]] = [
    # -------- ADD YOUR REAL TEACHERS HERE --------
    ("Mr. Kamau", 30),
    ("Ms. Atieno", 30),
    ("Mr. Otieno", 28),
    ("Mrs. Muthoni", 32),
    ("Dr. Sharma", 30),
    # --------------------------------------------
]

# ============================================================
# CORE FUNCTIONS
# ============================================================

def total_teacher_slots(teachers: List[Tuple[str, int]]) -> int:
    """Return sum of max weekly lessons across all teachers."""
    if not teachers:
        return 0
    return sum(load for _, load in teachers)

def proportional_reduction(required: Dict[str, int], available_slots: int) -> Dict[str, int]:
    """
    Reduce required lessons proportionally until they fit into available slots.
    Never sets a subject below zero, and preserves as much as possible.
    """
    total_req = sum(required.values())
    if total_req <= available_slots:
        return required.copy()
    
    deficit = total_req - available_slots
    reduced = required.copy()
    remaining_deficit = deficit
    
    # First pass: proportional reduction based on current load
    subjects_sorted = sorted(required.items(), key=lambda x: x[1], reverse=True)
    for subj, req in subjects_sorted:
        if remaining_deficit <= 0:
            break
        # share = ceil(req * deficit / total_req) but ensure integer
        share = int(round(req * deficit / total_req))
        share = min(share, reduced[subj])          # never go negative
        if share > 0:
            reduced[subj] -= share
            remaining_deficit -= share
    
    # Second pass: if still deficit (due to rounding), remove 1 lesson at a time
    # from the largest remaining subjects
    if remaining_deficit > 0:
        subjects_by_load = sorted(reduced.items(), key=lambda x: x[1], reverse=True)
        for subj, current in subjects_by_load:
            if remaining_deficit <= 0:
                break
            if current > 0:
                reduced[subj] -= 1
                remaining_deficit -= 1
    
    # Final sanity check – ensure no negative
    for subj in reduced:
        reduced[subj] = max(0, reduced[subj])
    
    return reduced

def print_report(required: Dict[str, int], available_slots: int):
    """Display feasibility status and recommended adjustments."""
    total_req = sum(required.values())
    print(f"📊 TOTAL REQUIRED LESSONS: {total_req}")
    print(f"👩‍🏫 TOTAL TEACHER SLOTS:   {available_slots}\n")
    
    if total_req <= available_slots:
        print("✅ FEASIBLE – Current requirements fit within teacher capacity.")
        return
    
    deficit = total_req - available_slots
    print(f"❌ NOT FEASIBLE – Short by {deficit} lessons.\n")
    print("🔧 Suggested reductions (proportional to subject loads):\n")
    
    reduced = proportional_reduction(required, available_slots)
    new_total = sum(reduced.values())
    
    for subj in required:
        old = required[subj]
        new = reduced[subj]
        if new < old:
            print(f"   • {subj}: {old} → {new}  (reduce by {old - new})")
        else:
            print(f"   • {subj}: {old} → {new}  (keep same)")
    
    print(f"\n📌 NEW TOTAL LESSONS: {new_total}")
    print(f"📌 NEW DEFICIT:      {new_total - available_slots} (negative means spare capacity)")

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    # Handle the critical error: zero teachers
    if not teachers:
        print("\n🚨 CRITICAL ERROR: No teachers defined in the system!\n")
        print("   The error 'total available teacher-slots = 0' occurs because")
        print("   the teacher list is empty. Please add at least one teacher.\n")
        print("   Example: teachers = [('Ms. Akinyi', 30)]")
        return
    
    total_slots = total_teacher_slots(teachers)
    print("\n" + "="*50)
    print("    TIMETABLE FEASIBILITY CHECK")
    print("="*50 + "\n")
    print_report(required_lessons, total_slots)
    print("\n" + "="*50)

if __name__ == "__main__":
    main()