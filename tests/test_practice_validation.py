"""Offline tests for Eloquify's adaptive-practice validation helpers."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from practice_engine import validate_exercise
from practice_evaluator import validate_evaluation
import speech_analyzer
from speech_analyzer import _json_safe


class PracticeValidationTests(unittest.TestCase):
    def exercise(self):
        return {
            "focus_area": "grammar",
            "evidence": ["The transcript contains the phrase 'she go'."],
            "exercise_title": "Practice subject-verb agreement",
            "context": "The latest transcript contains a specific agreement pattern to practice.",
            "practice_prompt": "Describe your usual school morning using third-person singular once.",
            "instructions": ["Speak in complete thoughts.", "Use a new example."],
            "reference_answer": "She goes to school early.",
            "success_criteria": ["Use a correct third-person singular verb."],
            "comparison_metric": "grammar_pattern",
        }

    def test_exercise_validation_adds_tracking_fields(self):
        item = validate_exercise(self.exercise(), "/tmp/recording.m4a")
        self.assertEqual(item["focus_area"], "grammar")
        self.assertTrue(item["practice_id"])
        self.assertEqual(item["source_recording"], "/tmp/recording.m4a")
        self.assertIn("generated_at", item)

    def test_exercise_rejects_unknown_focus(self):
        item = self.exercise()
        item["focus_area"] = "pronunciation"
        with self.assertRaises(ValueError):
            validate_exercise(item, "/tmp/recording.m4a")

    def test_evaluation_must_match_success_criteria(self):
        exercise = self.exercise()
        result = {
            "overall_feedback": "You answered the task clearly.",
            "task_completion": "achieved",
            "strengths": ["You used a relevant example."],
            "improvements": [],
            "success_criteria_results": [{
                "criterion": exercise["success_criteria"][0],
                "status": "met",
                "evidence": "The response used 'she goes'.",
            }],
            "next_step": "Try the same pattern in another sentence.",
        }
        checked = validate_evaluation(result, exercise)
        self.assertEqual(checked["task_completion"], "achieved")
        self.assertIn("evaluated_at", checked)

    def test_evaluation_rejects_mismatched_criteria(self):
        exercise = self.exercise()
        result = {
            "overall_feedback": "Good effort.",
            "task_completion": "achieved",
            "strengths": [],
            "improvements": [],
            "success_criteria_results": [{
                "criterion": "A different criterion",
                "status": "met",
                "evidence": "Evidence",
            }],
            "next_step": "Try again.",
        }
        with self.assertRaises(ValueError):
            validate_evaluation(result, exercise)

    def test_json_safe_handles_paths_and_nonfinite_floats(self):
        result = _json_safe({"path": __file__, "nan": math.nan, "inf": math.inf, "items": (1, 2)})
        self.assertIsInstance(result["path"], str)
        self.assertIsNone(result["nan"])
        self.assertIsNone(result["inf"])
        self.assertEqual(result["items"], [1, 2])

    def test_latest_analysis_is_written_as_valid_json(self):
        previous = speech_analyzer.PROJECT_DIR
        with tempfile.TemporaryDirectory() as temporary:
            speech_analyzer.PROJECT_DIR = Path(temporary)
            try:
                target = speech_analyzer.save_latest_analysis({"recording": Path("speech.m4a"), "value": math.nan})
                loaded = json.loads(target.read_text(encoding="utf-8"))
                self.assertEqual(loaded["recording"], "speech.m4a")
                self.assertIsNone(loaded["value"])
                self.assertEqual(target.name, "latest_analysis.json")
            finally:
                speech_analyzer.PROJECT_DIR = previous


if __name__ == "__main__":
    unittest.main()
