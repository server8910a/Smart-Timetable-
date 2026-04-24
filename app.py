#!/usr/bin/env python3
"""
School Timetable Fixer - Web API (Flask)
Deployable on Render, Heroku, etc.
"""

from flask import Flask, request, jsonify, render_template_string
from typing import Dict, List, Tuple

app = Flask(__name__)

# ------------------------------------------------------------
# CONFIGURATION (edit as needed)
# ------------------------------------------------------------
DEFAULT_REQUIRED = {
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

DEFAULT_TEACHERS = [
    ("Mr. Kamau", 30),
    ("Ms. Atieno", 30),
    ("Mr. Otieno", 28),
    ("Mrs. Muthoni", 32),
    ("Dr. Sharma", 30),
]

# ------------------------------------------------------------
# CORE LOGIC (same as before)
# ------------------------------------------------------------
def total_teacher_slots(teachers: List[Tuple[str, int]]) -> int:
    return sum(load for _, load in teachers)

def proportional_reduction(required: Dict[str, int], available_slots: int) -> Dict[str, int]:
    total_req = sum(required.values())
    if total_req <= available_slots:
        return required.copy()
    
    deficit = total_req - available_slots
    reduced = required.copy()
    remaining_deficit = deficit
    
    subjects_sorted = sorted(required.items(), key=lambda x: x[1], reverse=True)
    for subj, req in subjects_sorted:
        if remaining_deficit <= 0:
            break
        share = int(round(req * deficit / total_req))
        share = min(share, reduced[subj])
        if share > 0:
            reduced[subj] -= share
            remaining_deficit -= share
    
    if remaining_deficit > 0:
        subjects_by_load = sorted(reduced.items(), key=lambda x: x[1], reverse=True)
        for subj, current in subjects_by_load:
            if remaining_deficit <= 0:
                break
            if current > 0:
                reduced[subj] -= 1
                remaining_deficit -= 1
    
    for subj in reduced:
        reduced[subj] = max(0, reduced[subj])
    
    return reduced

# ------------------------------------------------------------
# WEB ENDPOINTS
# ------------------------------------------------------------
@app.route('/')
def home():
    """Simple HTML form to test the fixer."""
    return render_template_string('''
    <!doctype html>
    <html>
    <head><title>Timetable Fixer</title></head>
    <body>
        <h2>School Timetable Feasibility Checker</h2>
        <form action="/check" method="post">
            <label>Total required lessons (JSON):</label><br>
            <textarea name="required" rows="6" cols="50">{{ required|tojson }}</textarea><br><br>
            <label>Teachers (JSON list of [name, max_lessons]):</label><br>
            <textarea name="teachers" rows="6" cols="50">{{ teachers|tojson }}</textarea><br><br>
            <input type="submit" value="Check">
        </form>
        <hr>
        <p><strong>Example:</strong><br>
        Required: {"MATHEMATICS":30,"ENGLISH":30,"CREATIVE ARTS":30,...}<br>
        Teachers: [["Mr. Kamau",30],["Ms. Atieno",30],...]
        </p>
    </body>
    </html>
    ''', required=DEFAULT_REQUIRED, teachers=DEFAULT_TEACHERS)

@app.route('/check', methods=['POST'])
def check():
    try:
        required = request.form.get('required')
        teachers_data = request.form.get('teachers')
        if not required or not teachers_data:
            return jsonify({"error": "Both required and teachers are required"}), 400
        
        import json
        required_dict = json.loads(required)
        teachers_list = json.loads(teachers_data)
        # Convert teachers list to tuples
        teachers = [(name, int(load)) for name, load in teachers_list]
        
        total_slots = total_teacher_slots(teachers)
        total_req = sum(required_dict.values())
        
        if total_req <= total_slots:
            return jsonify({
                "feasible": True,
                "total_required": total_req,
                "total_slots": total_slots,
                "message": "✅ Feasible – enough teacher capacity."
            })
        
        reduced = proportional_reduction(required_dict, total_slots)
        new_total = sum(reduced.values())
        deficit = new_total - total_slots
        
        return jsonify({
            "feasible": False,
            "total_required": total_req,
            "total_slots": total_slots,
            "deficit": total_req - total_slots,
            "suggested_reductions": {
                subj: {"original": required_dict[subj], "new": reduced[subj]}
                for subj in required_dict
            },
            "new_total": new_total,
            "new_deficit": deficit,
            "message": f"⚠️ Need to reduce by {total_req - total_slots} lessons. Suggested reductions applied."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# HEALTH CHECK (for Render)
# ------------------------------------------------------------
@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=5000, debug=True)