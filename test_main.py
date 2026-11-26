import tempfile
import unittest
from pathlib import Path
from main import REQUIREMENTS, classify_requirement, build_report

class DecisionHelperTests(unittest.TestCase):
    def by_name(self, name):
        return next(r for r in REQUIREMENTS if r["name"] == name)

    def test_gst_is_no_ai(self):
        decision, _ = classify_requirement(self.by_name("Calculate GST on an invoice"))
        self.assertEqual(decision, "No AI (rules/code)")

    def test_pan_validation_is_no_ai(self):
        decision, _ = classify_requirement(self.by_name("Validate PAN number format"))
        self.assertEqual(decision, "No AI (rules/code)")

    def test_photo_defects_use_deep_learning(self):
        decision, _ = classify_requirement(self.by_name("Detect defects in product photos"))
        self.assertEqual(decision, "Deep Learning")

    def test_churn_uses_classical_ml(self):
        decision, _ = classify_requirement(self.by_name("Predict customers likely to cancel next month"))
        self.assertEqual(decision, "Classical ML")

    def test_hr_policy_uses_generative_ai(self):
        decision, _ = classify_requirement(self.by_name("Answer employee questions from the HR policy PDF"))
        self.assertEqual(decision, "Generative AI")

    def test_report_has_one_row_per_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = build_report(output_path=Path(tmp) / "report.csv")
            self.assertEqual(len(rows), len(REQUIREMENTS))

if __name__ == "__main__":
    unittest.main()
