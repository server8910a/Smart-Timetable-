import json
import sys
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class TimetableSolver:
    # ... (Keep all your existing TimetableSolver code exactly the same) ...
    # ... (No changes needed inside the solver class) ...

    def get_ai_explanation(self, violations=None):
        if not GEMINI_API_KEY:
            return "AI explanation unavailable (API key not set)."

        summary = {
            "grades": len(self.grades),
            "subjects": len(self.subjects),
            "teachers": len(self.teachers),
            "total_required": sum(self._get_required_lessons(g, s) for g in self.grades for s in self.subjects),
            "total_slots": len(self.class_groups) * self.num_days * self.num_slots,
            "violations": violations if violations else []
        }

        prompt = f"""
A school timetable was generated with the following soft constraint violations:
{json.dumps(summary['violations'], indent=2)}

Total required lessons: {summary['total_required']}, total available slots: {summary['total_slots']}.
Grades: {summary['grades']}, Subjects: {summary['subjects']}, Teachers: {summary['teachers']}.

Provide a brief, professional explanation of why these violations occurred and suggest 1-2 practical adjustments to eliminate them in future generations.
"""
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            response = requests.post(GEMINI_URL, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"AI API error: {response.status_code}"
        except Exception as e:
            return f"AI explanation error: {str(e)}"


@app.route('/generate', methods=['POST'])
def generate():
    try:
        config = request.json
        solver = TimetableSolver(config)
        solution = solver.solve()

        if solution:
            violations = solver.get_violations_summary()
            ai_explanation = None
            if violations:
                ai_explanation = solver.get_ai_explanation(violations)
            return jsonify({
                "success": True,
                "timetable": solution,
                "violations": violations,
                "ai_explanation": ai_explanation
            })
        else:
            ai_msg = solver.get_ai_explanation() if GEMINI_API_KEY else None
            return jsonify({
                "success": False,
                "message": "No feasible timetable found even with soft constraints.",
                "ai_explanation": ai_msg
            })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)})
    except Exception as e:
        return jsonify({"success": False, "message": "Internal error: " + str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)