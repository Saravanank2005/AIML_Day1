"""
AI Decision Helper
Classifies business requirements into No AI, Analytics/BI, Classical ML,
Deep Learning, or Generative AI using transparent requirement attributes.

AI category guide:
- AI: Broad field of systems performing tasks associated with human intelligence.
- ML: A subset of AI that learns patterns from data to make predictions/decisions.
- Deep Learning: A subset of ML using multi-layer neural networks; useful for
  complex data such as images, audio, and video.
- Generative AI: Models that generate new text, images, code, or other content.
  Retrieval-Augmented Generation (RAG) can ground answers in trusted documents.
"""

from pathlib import Path
import csv

OUTPUT_CSV = Path(__file__).with_name("decision_report.csv")

# Each attribute is a yes/no business characteristic.
REQUIREMENTS = [
    {
        "id": 1, "name": "Validate PAN number format",
        "description": "Check whether a PAN follows the required character pattern.",
        "fixed_rules_possible": True, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "A format check does not prove that the PAN is genuine or belongs to the user.",
        "next_step": "Implement a regex and add valid/invalid PAN test cases."
    },
    {
        "id": 2, "name": "Predict customers likely to cancel next month",
        "description": "Estimate customer churn risk from historical customer behaviour.",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": False,
        "risk": "Biased or stale data can produce misleading risk scores; predictions are not certainties.",
        "next_step": "Define churn and assemble labelled customer history with a time-based split."
    },
    {
        "id": 3, "name": "Summarise long support emails",
        "description": "Produce concise summaries of free-form support email conversations.",
        "fixed_rules_possible": False, "needs_learning": False,
        "unstructured_input": True, "input_type": "text",
        "needs_generated_content": True, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "The model may omit important details or invent facts; protect personal data.",
        "next_step": "Collect representative emails and define a summary format with human review."
    },
    {
        "id": 4, "name": "Detect defects in product photos",
        "description": "Identify visible scratches, dents, cracks, or other defects in images.",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": True, "input_type": "image",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": False,
        "risk": "Poor lighting, camera variation, and rare defects can cause false alarms or misses.",
        "next_step": "Create an expert-labelled image dataset covering defect types and normal products."
    },
    {
        "id": 5, "name": "Calculate GST on an invoice",
        "description": "Calculate tax using configured rates, taxable value, and invoice rules.",
        "fixed_rules_possible": True, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Incorrect tax rates or business rules can create compliance errors.",
        "next_step": "Confirm applicable tax rules and implement decimal-safe calculation with unit tests."
    },
    {
        "id": 6, "name": "Forecast weekly product demand",
        "description": "Predict future weekly demand from sales history and relevant drivers.",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": False, "input_type": "structured time series",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": True,
        "risk": "Promotions, stockouts, seasonality, and sudden market changes can distort forecasts.",
        "next_step": "Gather weekly sales and stockout data, then establish a seasonal-naive baseline."
    },
    {
        "id": 7, "name": "Answer employee questions from the HR policy PDF",
        "description": "Answer natural-language questions using the organisation's HR policy document.",
        "fixed_rules_possible": False, "needs_learning": False,
        "unstructured_input": True, "input_type": "text and PDF",
        "needs_generated_content": True, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Unsupported answers or outdated policy versions could mislead employees.",
        "next_step": "Build a permission-aware RAG proof of concept with citations to policy sections."
    },
    {
        "id": 8, "name": "Sort invoices by amount",
        "description": "Order invoice records by a numeric amount field.",
        "fixed_rules_possible": True, "needs_learning": False,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": False, "aggregation_or_trends": False,
        "risk": "Missing values, currency mismatches, or incorrect parsing can produce a wrong order.",
        "next_step": "Use a normal sort with an explicit currency and null-handling rule."
    },
    {
        "id": 9, "name": "Detect potentially fraudulent transactions",
        "description": "Flag unusual transactions using historical examples and behaviour patterns.",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": False, "input_type": "structured",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": False,
        "risk": "Fraud labels may be delayed or imbalanced; false positives can block genuine customers.",
        "next_step": "Work with fraud analysts to define labels, costs, and a review workflow."
    },
    {
        "id": 10, "name": "Group recurring customer complaint themes",
        "description": "Discover common themes across a large collection of free-text complaints.",
        "fixed_rules_possible": False, "needs_learning": True,
        "unstructured_input": True, "input_type": "text",
        "needs_generated_content": False, "data_available": True,
        "prediction_or_pattern_discovery": True, "aggregation_or_trends": True,
        "risk": "Themes may be hard to interpret and minority complaints may be hidden.",
        "next_step": "Sample and anonymise complaints, then test text embeddings and clustering with reviewers."
    },
]

# Lifecycle map: one practical task and accountable role per stage.
LIFECYCLE = {
    "Predict customers likely to cancel next month": [
        ("Problem framing", "Agree on the churn definition, prediction horizon, and intervention.", "Product Owner"),
        ("Data", "Build a dated customer-feature table with churn labels.", "Data Engineer"),
        ("Model", "Train and validate a calibrated classification model.", "Data Scientist"),
        ("Deployment", "Expose weekly churn scores to the retention workflow.", "ML Engineer"),
        ("Monitoring", "Track drift, precision/recall, and retention outcomes.", "ML Engineer"),
    ],
    "Summarise long support emails": [
        ("Problem framing", "Define summary length, required fields, and quality criteria.", "Product Owner"),
        ("Data", "Collect approved, redacted email examples and evaluation references.", "Data Analyst"),
        ("Model", "Select an LLM and test prompts against factuality criteria.", "AI Engineer"),
        ("Deployment", "Integrate summarisation into the support tool with agent approval.", "Software Engineer"),
        ("Monitoring", "Review omissions, hallucinations, latency, and cost.", "Support Operations Lead"),
    ],
    "Detect defects in product photos": [
        ("Problem framing", "Define defect classes and acceptable false-negative rate.", "Quality Manager"),
        ("Data", "Capture and label representative product images.", "Data Engineer"),
        ("Model", "Train and evaluate an image classifier or object detector.", "Data Scientist"),
        ("Deployment", "Run image inference in the inspection pipeline.", "ML Engineer"),
        ("Monitoring", "Sample predictions and track performance by camera and product type.", "Quality Engineer"),
    ],
    "Forecast weekly product demand": [
        ("Problem framing", "Define forecast horizon, product granularity, and error metric.", "Demand Planner"),
        ("Data", "Prepare weekly sales, inventory, promotion, and calendar data.", "Data Engineer"),
        ("Model", "Compare seasonal-naive, statistical, and ML forecasting baselines.", "Data Scientist"),
        ("Deployment", "Publish forecasts to the replenishment dashboard.", "ML Engineer"),
        ("Monitoring", "Track forecast error, bias, and stockout impact each week.", "Demand Planner"),
    ],
    "Answer employee questions from the HR policy PDF": [
        ("Problem framing", "Set supported topics, access rules, and citation requirements.", "HR Product Owner"),
        ("Data", "Version, extract, chunk, and permission-tag approved HR documents.", "Data Engineer"),
        ("Model", "Build and evaluate retrieval plus grounded answer generation (RAG).", "AI Engineer"),
        ("Deployment", "Deploy the assistant behind employee authentication.", "Software Engineer"),
        ("Monitoring", "Audit answer citations, unanswered questions, and policy changes.", "HR Operations Lead"),
    ],
    "Detect potentially fraudulent transactions": [
        ("Problem framing", "Define fraud types, decision thresholds, and review costs.", "Fraud Product Owner"),
        ("Data", "Join transaction history with confirmed investigation outcomes.", "Data Engineer"),
        ("Model", "Train an imbalanced-class classifier and evaluate precision-recall.", "Data Scientist"),
        ("Deployment", "Send risk scores to a human investigation queue.", "ML Engineer"),
        ("Monitoring", "Track false positives, fraud loss, and changing attack patterns.", "Fraud Operations Lead"),
    ],
    "Group recurring customer complaint themes": [
        ("Problem framing", "Agree how themes will guide service improvements.", "Customer Experience Lead"),
        ("Data", "Anonymise, deduplicate, and prepare complaint text.", "Data Engineer"),
        ("Model", "Generate text embeddings and cluster complaints for review.", "Data Scientist"),
        ("Deployment", "Publish reviewed themes in the service analytics dashboard.", "Analytics Engineer"),
        ("Monitoring", "Review cluster quality and emerging themes each month.", "Customer Experience Lead"),
    ],
}

def classify_requirement(req):
    """Return (category, reasons) using a simple, explainable decision order."""
    reasons = []


    if req["needs_generated_content"] and req["unstructured_input"]:
        category = "Generative AI"
        reasons.append("It must generate new content from unstructured input.")
        if req["name"] == "Answer employee questions from the HR policy PDF":
            reasons.append("RAG can retrieve relevant policy passages and ground answers with citations.")

    elif req["input_type"] == "image" and req["needs_learning"]:
        category = "Deep Learning"
        reasons.append("It must learn visual patterns from product images.")
        reasons.append("Deep neural networks are well suited to image classification or detection.")

    elif req["needs_learning"] and req["prediction_or_pattern_discovery"]:
        category = "Classical ML"
        reasons.append("It must learn patterns from historical examples rather than rely only on fixed rules.")
        reasons.append("The task is prediction or pattern discovery on structured/text-derived features.")

    elif req["aggregation_or_trends"]:
        category = "Analytics/BI"
        reasons.append("The main need is aggregation, trend analysis, or reporting.")
        reasons.append("A dashboard and descriptive statistics may solve it without a predictive model.")
    elif req["fixed_rules_possible"]:
        category = "No AI (rules/code)"
        reasons.append("The outcome is deterministic and can be expressed as explicit rules or code.")
        reasons.append("A conventional implementation is easier to test, explain, and maintain.")
    else:
        category = "Analytics/BI"
        reasons.append("Start with descriptive analysis before introducing a learned model.")

    if not req["data_available"] and category not in ("No AI (rules/code)", "Analytics/BI"):
        reasons.append("Suitable data is not yet available, so a data-collection step is required.")
    elif req["data_available"] and category in ("Classical ML", "Deep Learning", "Generative AI"):
        reasons.append("The requirement indicates data or documents are available for a proof of concept.")

    return category, reasons


def build_report(requirements=REQUIREMENTS, output_path=OUTPUT_CSV):
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
    print("\nLIFECYCLE MAP (AI-suitable requirements)")
    for name, stages in LIFECYCLE.items():
        print(f"\n{name}")
        for stage, task, role in stages:
            print(f"  {stage}: {task} [Role: {role}]")


def main():
    rows = build_report()
    print("AI DECISION REPORT")
    print("-" * 72)
    for row in rows:
        print(f'{row["id"]:>2}. {row["requirement"]}')
        print(f'    Decision: {row["decision"]}')
        print(f'    Reasons:  {row["reasons"]}')
    print(f"\nCSV report written to: {OUTPUT_CSV.name}")
    print_lifecycle()


if __name__ == "__main__":
    main()
