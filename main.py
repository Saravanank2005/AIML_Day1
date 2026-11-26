"""
AI Decision Helper: classifies requirements into No AI (rules/code), Analytics/BI,
Classical ML, Deep Learning, or Generative AI.

Category basis:
- AI: Broad field of systems performing tasks associated with human intelligence.
- ML (Classical ML): Learns patterns from structured data to predict or group outcomes.
- Deep Learning: Multi-layer neural networks for complex inputs like images, audio, or video.
- Generative AI: Models that create text, summaries, or grounded answers (e.g. RAG).
"""

import csv
from pathlib import Path

OUTPUT_CSV = Path(__file__).with_name("decision_report.csv")
AI_CATEGORIES = {"Classical ML", "Deep Learning", "Generative AI"}
REQUIRED_BOOLS = (
    "fixed_rules_possible",
    "needs_learning",
    "unstructured_input",
    "needs_generated_content",
    "data_available",
    "prediction_or_pattern_discovery",
    "aggregation_or_trends",
)

REQUIREMENTS = [
    {
        "id": 1, "name": "Validate PAN number format",
        "fixed_rules_possible": True, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Format check alone does not prove a PAN is genuine.",
        "next_step": "Implement a regex check with valid/invalid unit tests.",
    },
    {
        "id": 2, "name": "Predict customers likely to cancel next month",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": False,
        "risk": "Stale or biased history can skew churn scores.",
        "next_step": "Define churn and build a labelled customer history table.",
    },
    {
        "id": 3, "name": "Summarise long support emails",
        "fixed_rules_possible": False, "needs_learning": False,
        "unstructured_input": True, "input_type": "text",
        "needs_generated_content": True, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "May omit key details or hallucinate; redact personal data.",
        "next_step": "Collect redacted sample emails and define summary format.",
    },
    {
        "id": 4, "name": "Detect defects in product photos",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": True, "input_type": "image",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": False,
        "risk": "Lighting changes and rare defects can cause misses.",
        "next_step": "Build a labelled photo dataset covering normal and defect cases.",
    },
    {
        "id": 5, "name": "Calculate GST on an invoice",
        "fixed_rules_possible": True, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Wrong tax rates or rounding rules cause compliance errors.",
        "next_step": "Verify tax slabs and implement decimal-safe unit tests.",
    },
    {
        "id": 6, "name": "Forecast weekly product demand",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": True,
        "risk": "Stockouts and promotions can distort demand history.",
        "next_step": "Prepare weekly sales data and a seasonal-naive baseline.",
    },
    {
        "id": 7, "name": "Answer employee questions from the HR policy PDF",
        "fixed_rules_possible": False, "needs_learning": False,
        "unstructured_input": True, "input_type": "text",
        "needs_generated_content": True, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Outdated policy PDFs or ungrounded answers mislead staff.",
        "next_step": "Prototype RAG with strict citations to approved policy sections.",
    },
    {
        "id": 8, "name": "Sort invoices by amount",
        "fixed_rules_possible": True, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Mixed currencies or null amounts can break ordering.",
        "next_step": "Use standard sorting with currency and null checks.",
    },
    {
        "id": 9, "name": "Flag potentially fraudulent transactions",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": False,
        "risk": "Class imbalance and false positives can block valid users.",
        "next_step": "Align with fraud analysts on labels and review thresholds.",
    },
    {
        "id": 10, "name": "Track monthly regional sales trends",
        "fixed_rules_possible": False, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": True,
        "risk": "Inconsistent region mappings can skew monthly totals.",
        "next_step": "Build a SQL summary and BI dashboard by region and month.",
    },
]

LIFECYCLE = {
    "Predict customers likely to cancel next month": [
        ("Problem framing", "Define churn window and retention action.", "Product Owner"),
        ("Data", "Build dated customer feature and churn label table.", "Data Engineer"),
        ("Model", "Train and calibrate a classification model.", "Data Scientist"),
        ("Deployment", "Publish weekly churn risk scores to CRM.", "ML Engineer"),
        ("Monitoring", "Track drift, precision/recall, and retention lift.", "ML Engineer"),
    ],
    "Summarise long support emails": [
        ("Problem framing", "Set summary length and accuracy rules.", "Product Owner"),
        ("Data", "Collect redacted sample emails for evaluation.", "Data Analyst"),
        ("Model", "Select LLM and tune factuality prompts.", "AI Engineer"),
        ("Deployment", "Embed draft summary with agent review in helpdesk.", "Software Engineer"),
        ("Monitoring", "Audit hallucinations, omissions, latency, and cost.", "Support Lead"),
    ],
    "Detect defects in product photos": [
        ("Problem framing", "Define defect classes and miss tolerance.", "Quality Manager"),
        ("Data", "Capture and label representative product photos.", "Data Engineer"),
        ("Model", "Train and evaluate an image defect detector.", "Data Scientist"),
        ("Deployment", "Run inference on the inspection camera feed.", "ML Engineer"),
        ("Monitoring", "Sample predictions across lighting and product lines.", "Quality Engineer"),
    ],
    "Forecast weekly product demand": [
        ("Problem framing", "Set forecast horizon, SKU level, and error metric.", "Demand Planner"),
        ("Data", "Clean weekly sales, promo, and stockout history.", "Data Engineer"),
        ("Model", "Benchmark ML model against seasonal-naive baseline.", "Data Scientist"),
        ("Deployment", "Feed weekly forecasts to replenishment planning.", "ML Engineer"),
        ("Monitoring", "Review weekly forecast error and bias.", "Demand Planner"),
    ],
    "Answer employee questions from the HR policy PDF": [
        ("Problem framing", "Define supported HR topics and citation rules.", "HR Product Owner"),
        ("Data", "Chunk and permission-tag versioned HR PDFs.", "Data Engineer"),
        ("Model", "Build and test RAG retrieval with grounded answers.", "AI Engineer"),
        ("Deployment", "Launch behind employee SSO authentication.", "Software Engineer"),
        ("Monitoring", "Audit citations and flag policy document updates.", "HR Operations Lead"),
    ],
    "Flag potentially fraudulent transactions": [
        ("Problem framing", "Define fraud types and false-positive cost.", "Fraud Product Owner"),
        ("Data", "Join transactions with confirmed fraud outcomes.", "Data Engineer"),
        ("Model", "Train imbalanced classifier using precision-recall.", "Data Scientist"),
        ("Deployment", "Route high-risk scores to analyst review queue.", "ML Engineer"),
        ("Monitoring", "Monitor false alarms and emerging fraud patterns.", "Fraud Ops Lead"),
    ],
}


def validate_requirement(req):
    """Validate requirement structure and attribute types."""
    if not isinstance(req, dict):
        raise ValueError("Requirement must be a dictionary.")
    for key in ("id", "name", "input_type", "risk", "next_step"):
        if key not in req or req[key] in (None, ""):
            raise ValueError(f"Missing required field: {key}")
    for key in REQUIRED_BOOLS:
        if key not in req or not isinstance(req[key], bool):
            raise ValueError(f"Field '{key}' must be a boolean.")


def classify_requirement(req):
    """Return (category, reasons) for a single requirement."""
    validate_requirement(req)
    reasons = []

    if req["needs_generated_content"]:
        category = "Generative AI"
        reasons.append("Needs generated natural-language output from unstructured content.")
        if "PDF" in req["name"]:
            reasons.append("Use RAG to ground answers in policy citations.")
    elif req["unstructured_input"] and req["needs_learning"] and req["input_type"] == "image":
        category = "Deep Learning"
        reasons.append("Requires learning visual patterns from unstructured image data.")
    elif req["needs_learning"] and req["prediction_or_pattern_discovery"]:
        category = "Classical ML"
        reasons.append("Needs pattern learning on historical structured data for prediction.")
    elif req["aggregation_or_trends"]:
        category = "Analytics/BI"
        reasons.append("Needs aggregation and trend reporting without predictive modelling.")
    elif req["fixed_rules_possible"]:
        category = "No AI (rules/code)"
        reasons.append("Deterministic logic can be solved with explicit rules or code.")
    else:
        category = "Analytics/BI"
        reasons.append("Start with descriptive reporting before considering ML.")

    if category in AI_CATEGORIES and not req["data_available"]:
        reasons.append("Data is missing; collect and label data before modelling.")
    elif category in AI_CATEGORIES and req["data_available"]:
        reasons.append("Required data or documents are available for a prototype.")

    return category, reasons


def build_report(requirements=REQUIREMENTS, output_path=OUTPUT_CSV):
    """Classify all requirements and write the CSV report."""
    if not requirements:
        raise ValueError("Requirements list cannot be empty.")

    rows = []
    for req in requirements:
        decision, reasons = classify_requirement(req)
        rows.append({
            "id": req["id"],
            "requirement": req["name"],
            "decision": decision,
            "reasons": " ".join(reasons),
            "risks": req["risk"],
            "first_next_step": req["next_step"],
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "requirement", "decision", "reasons", "risks", "first_next_step"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_lifecycle():
    """Print the 5-stage lifecycle map for AI-suitable requirements."""
    print("\nLIFECYCLE MAP (AI-suitable requirements)")
    for name, stages in LIFECYCLE.items():
        print(f"\n{name}")
        for stage, task, role in stages:
            print(f"  {stage}: {task} [{role}]")


def main():
    rows = build_report()
    print("AI DECISION REPORT")
    print("-" * 60)
    for row in rows:
        print(f'{row["id"]:>2}. {row["requirement"]} -> {row["decision"]}')
        print(f'    Reason: {row["reasons"]}')
    print(f"\nSaved report to {OUTPUT_CSV.name}")
    print_lifecycle()


if __name__ == "__main__":
    main()
