from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)  # allow requests from any origin (useful for separate frontend)

# -------------------- DEFAULT DATA (only for initial form display) --------------------
DEFAULT_REQUIRED = {
    "MATHEMATICS": 30,
    "ENGLISH": 30,
    "CREATIVE ARTS": 30,
    "KISWAHILI": 24,
    "SCIENCE": 24,
    "CRE": 24,
    "AGRICULTURE": 24,
    "SOCIAL STUDIES": 24,
    "PRETECHNICAL": 12
}

DEFAULT_TEACHERS = [
    ["Mr. Kamau", 30],
    ["Ms. Atieno", 30],
    ["Mr. Otieno", 28],
    ["Mrs. Muthoni", 32],
    ["Dr. Sharma", 30]
]

# -------------------- CORE LOGIC --------------------
def total_teacher_slots(teachers):
    """teachers: list of [name, max_lessons]"""
    return sum(load for _, load in teachers)

def proportional_reduction(required, available_slots):
    """
    required: dict {subject: lessons}
    available_slots: int
    Returns a new dict with reduced lessons (never negative).
    """
    total_req = sum(required.values())
    if total_req <= available_slots:
        return required.copy()

    deficit = total_req - available_slots
    reduced = required.copy()
    remaining_deficit = deficit

    # First pass: proportional reduction
    subjects_sorted = sorted(required.items(), key=lambda x: x[1], reverse=True)
    for subj, req in subjects_sorted:
        if remaining_deficit <= 0:
            break
        share = int(round(req * deficit / total_req))
        share = min(share, reduced[subj])
        if share > 0:
            reduced[subj] -= share
            remaining_deficit -= share

    # Second pass: clean up rounding errors (remove 1 at a time from largest)
    if remaining_deficit > 0:
        subjects_by_load = sorted(reduced.items(), key=lambda x: x[1], reverse=True)
        for subj, current in subjects_by_load:
            if remaining_deficit <= 0:
                break
            if current > 0:
                reduced[subj] -= 1
                remaining_deficit -= 1

    # Ensure no negatives
    for subj in reduced:
        reduced[subj] = max(0, reduced[subj])

    return reduced

# -------------------- WEB ROUTES --------------------
@app.route('/')
def home():
    """Serve the HTML form with default data prefilled."""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>School Timetable Fixer</title>
        <style>
            body { font-family: sans-serif; margin: 2rem; }
            textarea { width: 100%; max-width: 500px; font-family: monospace; }
            button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
            pre { background: #f4f4f4; padding: 1rem; border-radius: 5px; overflow-x: auto; }
            .error { color: red; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <h1>📅 School Timetable Feasibility Checker</h1>
        <p>Enter your required lessons and teacher details in JSON format.</p>

        <div>
            <label><strong>Required lessons (JSON):</strong></label><br>
            <textarea id="required" rows="8" cols="60">{{ required | tojson }}</textarea>
        </div>

        <div style="margin-top: 1rem;">
            <label><strong>Teachers (JSON list of [name, max_lessons]):</strong></label><br>
            <textarea id="teachers" rows="6" cols="60">{{ teachers | tojson }}</textarea>
        </div>

        <div style="margin-top: 1rem;">
            <button onclick="checkFeasibility()">✅ Check Feasibility</button>
        </div>

        <h2>Result:</h2>
        <pre id="result">Click the button to see results.</pre>

        <script>
            async function checkFeasibility() {
                const requiredText = document.getElementById('required').value;
                const teachersText = document.getElementById('teachers').value;
                const resultPre = document.getElementById('result');

                // Basic validation
                if (!requiredText.trim() || !teachersText.trim()) {
                    resultPre.innerHTML = '<span class="error">Error: Both fields are required.</span>';
                    return;
                }

                resultPre.innerText = "⏳ Checking...";

                try {
                    const response = await fetch('/check', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            required: requiredText,
                            teachers: teachersText
                        })
                    });

                    const data = await response.json();
                    if (response.ok) {
                        resultPre.innerText = JSON.stringify(data, null, 2);
                    } else {
                        resultPre.innerHTML = `<span class="error">Server error: ${data.error || 'Unknown error'}</span>`;
                    }
                } catch (err) {
                    resultPre.innerHTML = `<span class="error">❌ Failed to fetch: ${err.message}</span>`;
                }
            }
        </script>
    </body>
    </html>
    ''', required=DEFAULT_REQUIRED, teachers=DEFAULT_TEACHERS)

@app.route('/check', methods=['POST'])
def check():
    """
    Expects JSON body:
    {
        "required": "{...json string or object}",
        "teachers": "[...json string or array]"
    }
    Returns reduced schedule or feasibility status.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must be JSON"}), 400

        required_raw = data.get('required')
        teachers_raw = data.get('teachers')

        if not required_raw or not teachers_raw:
            return jsonify({"error": "Missing 'required' or 'teachers' field"}), 400

        # Parse if they are strings (from textarea) or already dict/list
        if isinstance(required_raw, str):
            required = json.loads(required_raw)
        else:
            required = required_raw

        if isinstance(teachers_raw, str):
            teachers_list = json.loads(teachers_raw)
        else:
            teachers_list = teachers_raw

        # Validate each teacher: [name, max_lessons] where max_lessons is number
        teachers = []
        for item in teachers_list:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return jsonify({"error": f"Invalid teacher entry: {item}. Expected [name, max_lessons]"}), 400
            try:
                load = int(item[1])
            except (ValueError, TypeError):
                return jsonify({"error": f"Teacher load must be a number: {item}"}), 400
            teachers.append([item[0], load])

        total_slots = total_teacher_slots(teachers)
        total_req = sum(required.values())

        # Feasibility check
        if total_req <= total_slots:
            return jsonify({
                "feasible": True,
                "message": "✅ Current requirements fit within teacher capacity.",
                "total_required": total_req,
                "total_slots": total_slots
            })

        # Not feasible: compute reductions
        reduced = proportional_reduction(required, total_slots)
        new_total = sum(reduced.values())
        deficit = total_req - total_slots

        # Build a clean response with reduction details
        reductions = {}
        for subj in required:
            original = required[subj]
            new_val = reduced[subj]
            if new_val != original:
                reductions[subj] = {"original": original, "new": new_val, "reduce_by": original - new_val}
            else:
                reductions[subj] = {"original": original, "new": new_val, "reduce_by": 0}

        return jsonify({
            "feasible": False,
            "message": f"⚠️ Not feasible. Need to reduce total lessons by {deficit}.",
            "total_required": total_req,
            "total_slots": total_slots,
            "deficit": deficit,
            "suggested_reductions": reductions,
            "new_total_lessons": new_total,
            "new_deficit": new_total - total_slots
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return {"status": "ok", "service": "timetable-fixer"}

if __name__ == '__main__':
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)