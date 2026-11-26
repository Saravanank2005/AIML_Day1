import tempfile
import unittest
from pathlib import Path
from main import AI_CATEGORIES, LIFECYCLE, REQUIREMENTS, build_report, classify_requirement


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

    def test_sales_trends_use_analytics_bi(self):
        decision, _ = classify_requirement(self.by_name("Track monthly regional sales trends"))
        self.assertEqual(decision, "Analytics/BI")

    def test_lifecycle_covers_ai_requirements(self):
        for req in REQUIREMENTS:
            decision, _ = classify_requirement(req)
            if decision in AI_CATEGORIES:
                self.assertIn(req["name"], LIFECYCLE)
                self.assertEqual(len(LIFECYCLE[req["name"]]), 5)

    def test_invalid_requirement_raises_error(self):
        with self.assertRaises(ValueError):
            classify_requirement({"id": 99, "name": "Broken"})

    def test_report_has_one_row_per_requirement(self):
        self.assertGreaterEqual(len(REQUIREMENTS), 10)
        with tempfile.TemporaryDirectory() as tmp:
            rows = build_report(output_path=Path(tmp) / "report.csv")
            self.assertEqual(len(rows), len(REQUIREMENTS))


if __name__ == "__main__":
    unittest.main()
