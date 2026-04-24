from flask import Flask, request, jsonify, render_template_string
import json
import os

app = Flask(__name__)

# -------------------- CONFIGURATION --------------------
DEFAULT_REQUIRED = {
    "MATHEMATICS": 30, "ENGLISH": 30, "CREATIVE ARTS": 30,
    "KISWAHILI": 24, "SCIENCE": 24, "CRE": 24,
    "AGRICULTURE": 24, "SOCIAL STUDIES": 24, "PRETECHNICAL": 12
}
DEFAULT_TEACHERS = [
    ["Mr. Kamau", 30], ["Ms. Atieno", 30], ["Mr. Otieno", 28],
    ["Mrs. Muthoni", 32], ["Dr. Sharma", 30]
]

def total_teacher_slots(teachers):
    return sum(load for _, load in teachers)

def proportional_reduction(required, available_slots):
    total_req = sum(required.values())
    if total_req <= available_slots:
        return required.copy()
    deficit = total_req - available_slots
    reduced = required.copy()
    remaining_deficit = deficit
    # proportional share
    for subj, req in sorted(required.items(), key=lambda x: x[1], reverse=True):
        if remaining_deficit <= 0:
            break
        share = int(round(req * deficit / total_req))
        share = min(share, reduced[subj])
        reduced[subj] -= share
        remaining_deficit -= share
    # mop up
    if remaining_deficit > 0:
        for subj in sorted(reduced, key=reduced.get, reverse=True):
            if remaining_deficit <= 0:
                break
            if reduced[subj] > 0:
                reduced[subj] -= 1
                remaining_deficit -= 1
    return {k: max(0, v) for k, v in reduced.items()}

# -------------------- ROUTES --------------------
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Timetable Fixer</title>
        <style>
            body { font-family: Arial; margin: 2rem; }
            textarea { width: 100%; max-width: 500px; font-family: monospace; }
            button { padding: 10px 20px; font-size: 16px; }
            pre { background: #f0f0f0; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <h1>📊 School Timetable Fixer</h1>
        <div>
            <label><strong>Required lessons (JSON)</strong></label><br>
            <textarea id="required" rows="8" cols="60">{{ required | tojson }}</textarea>
        </div>
        <div style="margin-top: 15px;">
            <label><strong>Teachers (JSON list of [name, max_lessons])</strong></label><br>
            <textarea id="teachers" rows="6" cols="60">{{ teachers | tojson }}</textarea>
        </div>
        <div style="margin-top: 15px;">
            <button onclick="checkNow()">🔍 Check Feasibility</button>
        </div>
        <h2>Result</h2>
        <pre id="result">Click the button to see results.</pre>

        <script>
            async function checkNow() {
                const requiredText = document.getElementById('required').value;
                const teachersText = document.getElementById('teachers').value;
                const resultPre = document.getElementById('result');
                
                if (!requiredText.trim() || !teachersText.trim()) {
                    resultPre.innerHTML = '<span class="error">❌ Both fields are required.</span>';
                    return;
                }

                resultPre.innerText = "⏳ Sending request...";

                try {
                    const response = await fetch(window.location.origin + '/check', {
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
                        resultPre.innerHTML = `<span class="error">Server error: ${data.error || 'Unknown'}</span>`;
                    }
                } catch (err) {
                    resultPre.innerHTML = `<span class="error">❌ Failed to fetch: ${err.message}<br>Check if the server is running and you are connected to the internet.</span>`;
                }
            }
        </script>
    </body>
    </html>
    ''', required=DEFAULT_REQUIRED, teachers=DEFAULT_TEACHERS)

@app.route('/check', methods=['POST'])
def check():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        required_raw = data.get('required')
        teachers_raw = data.get('teachers')
        if not required_raw or not teachers_raw:
            return jsonify({"error": "Missing 'required' or 'teachers'"}), 400

        # Parse JSON strings if necessary
        required = json.loads(required_raw) if isinstance(required_raw, str) else required_raw
        teachers_list = json.loads(teachers_raw) if isinstance(teachers_raw, str) else teachers_raw

        # Convert teachers to list of [name, load]
        teachers = []
        for t in teachers_list:
            if not isinstance(t, (list, tuple)) or len(t) < 2:
                return jsonify({"error": f"Invalid teacher: {t}"}), 400
            teachers.append([t[0], int(t[1])])

        total_slots = total_teacher_slots(teachers)
        total_req = sum(required.values())

        if total_req <= total_slots:
            return jsonify({
                "feasible": True,
                "message": "✅ Feasible – enough teacher slots",
                "total_required": total_req,
                "total_slots": total_slots
            })

        reduced = proportional_reduction(required, total_slots)
        new_total = sum(reduced.values())
        return jsonify({
            "feasible": False,
            "message": f"⚠️ Not feasible. Need to reduce by {total_req - total_slots} lessons.",
            "total_required": total_req,
            "total_slots": total_slots,
            "deficit": total_req - total_slots,
            "suggested_reductions": {
                subj: {"original": required[subj], "new": reduced[subj]}
                for subj in required
            },
            "new_total": new_total
        })
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)