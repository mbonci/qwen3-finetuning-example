"""Generate synthetic statute pairs with ordinal similarity scores.

Outputs data/mock_pairs.jsonl — a self-contained mock dataset that mimics the
boundary-sampling mechanics of the real 8M-statute corpus without containing any
proprietary text. Run directly:

    python src/generate_mock_data.py

Each row is a (anchor, candidate) pair with:
  - Simulated statutory language drawn from topic-specific templates
  - A cosine similarity score sampled from the appropriate quantile bin
  - An ordinal LLM-judge score (1–8) consistent with the cosine bin
  - Metadata: topic, state prefixes, pair type, bin label
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TOPICS = [
    "Health",
    "Education",
    "Government Operations",
    "Environment",
    "Transportation",
    "Criminal Justice",
    "Housing",
    "Finance",
]

STATES = [
    "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
    "WA", "AZ", "MA", "TN", "CO", "MN", "OR", "WI", "MD", "VA",
]

TOPIC_TEMPLATES: dict[str, list[str]] = {
    "Health": [
        "Any person providing {service} services shall obtain a license from the Department of Health "
        "prior to commencing operations. Licenses shall be renewed annually upon payment of the "
        "prescribed fee and completion of {hours} hours of continuing education.",
        "The {board} Board shall establish minimum standards for {facility} facilities operating "
        "within this state, including staffing ratios, equipment requirements, and patient safety "
        "protocols. Facilities failing to meet these standards shall be subject to civil penalties.",
        "No {substance} shall be dispensed without a valid prescription issued by a licensed "
        "practitioner. Pharmacies shall maintain records of all dispensations for a period of "
        "not less than {years} years.",
        "Health insurers operating in this state shall cover {treatment} treatment without prior "
        "authorization requirements. Coverage shall include inpatient and outpatient services "
        "deemed medically necessary by the treating physician.",
    ],
    "Education": [
        "Each school district shall adopt a {subject} curriculum aligned with state academic "
        "standards. The curriculum shall be reviewed and updated every {years} years to reflect "
        "current research and best practices.",
        "Teachers employed in public schools shall hold a valid teaching certificate issued by "
        "the State Board of Education. Certificates shall specify the grade levels and subject "
        "areas for which the holder is qualified.",
        "Students with disabilities shall receive a free appropriate public education in the "
        "least restrictive environment. Individualized education programs shall be developed "
        "in collaboration with parents and relevant specialists.",
        "The {fund} Fund is hereby established to provide grants to school districts demonstrating "
        "financial hardship. Grants shall be awarded based on need as determined by the "
        "Department of Education.",
    ],
    "Government Operations": [
        "All contracts for {goods} exceeding {threshold} dollars shall be subject to competitive "
        "bidding. Contracts shall be awarded to the lowest responsible bidder meeting the "
        "specifications set forth in the solicitation.",
        "Public records shall be made available for inspection during regular business hours. "
        "Agencies shall respond to requests within {days} business days and may charge reasonable "
        "fees for reproduction.",
        "The {agency} Agency is authorized to promulgate rules and regulations necessary to carry "
        "out the provisions of this chapter. Rules shall be adopted in accordance with the "
        "Administrative Procedure Act.",
        "Annual audits of all state agencies shall be conducted by the State Auditor. Reports "
        "shall be submitted to the Governor and Legislature within {months} months of the close "
        "of each fiscal year.",
    ],
    "Environment": [
        "No person shall discharge {pollutant} into waters of the state without a permit issued "
        "by the Department of Environmental Quality. Permits shall specify discharge limits and "
        "monitoring requirements.",
        "Operators of {facility} facilities shall submit annual emissions reports to the Air "
        "Quality Division. Reports shall include measured concentrations of regulated pollutants "
        "and comparisons to applicable standards.",
        "The {program} Program is established to fund remediation of contaminated sites. Priority "
        "shall be given to sites posing imminent risks to public health or the environment.",
        "Solid waste disposal facilities shall comply with closure and post-closure care "
        "requirements for a minimum period of {years} years following cessation of operations.",
    ],
    "Transportation": [
        "Commercial vehicles operating on state highways shall comply with weight and dimension "
        "limits established by the Department of Transportation. Overweight permits may be "
        "issued for non-divisible loads.",
        "Operators of motor vehicles shall maintain liability insurance in amounts not less than "
        "{amount} dollars per occurrence. Proof of insurance shall be carried at all times.",
        "The {authority} Authority is authorized to issue revenue bonds for the purpose of "
        "financing construction and improvements to transportation infrastructure.",
        "Speed limits on state highways shall be established by the Department of Transportation "
        "based on engineering studies. Posted limits shall not exceed {limit} miles per hour "
        "in urban areas.",
    ],
    "Criminal Justice": [
        "Any person convicted of {offense} shall be sentenced to a term of imprisonment of not "
        "less than {years} years. The court may suspend the sentence upon completion of an "
        "approved treatment program.",
        "Law enforcement officers shall provide written notice of rights to persons taken into "
        "custody. Statements obtained without proper notice shall be inadmissible in any "
        "subsequent proceeding.",
        "The {board} Board of Parole shall review the case of each eligible offender no later "
        "than {months} months prior to the earliest release date. Release decisions shall be "
        "based on risk assessment and rehabilitation progress.",
        "Victims of {crime} crimes shall be notified of all court proceedings and shall have "
        "the right to submit impact statements prior to sentencing.",
    ],
    "Housing": [
        "Landlords shall maintain rental premises in a habitable condition, including adequate "
        "heating, plumbing, and structural integrity. Failure to remedy deficiencies within "
        "{days} days of notice shall entitle tenants to withhold rent.",
        "The {fund} Affordable Housing Fund is established to provide low-interest loans to "
        "developers constructing units affordable to households earning less than {percent} "
        "percent of area median income.",
        "No person shall be discriminated against in the sale or rental of housing on the basis "
        "of race, color, national origin, religion, sex, familial status, or disability. "
        "Complaints shall be filed within {days} days of the alleged violation.",
        "Zoning ordinances shall include provisions for accessory dwelling units in all "
        "residential zones. Permit applications shall be approved or denied within {days} days.",
    ],
    "Finance": [
        "Depository institutions operating in this state shall maintain capital ratios in "
        "accordance with standards established by the Commissioner of Financial Institutions. "
        "Institutions falling below minimum ratios shall submit restoration plans within {days} days.",
        "Money transmitters shall obtain a license from the Department of Financial Services "
        "prior to conducting business. Licensees shall maintain surety bonds in amounts "
        "proportional to transaction volume.",
        "The {board} Securities Board is authorized to investigate and prosecute violations of "
        "the Securities Act. Civil penalties shall not exceed {amount} dollars per violation.",
        "Consumer lenders shall disclose the annual percentage rate, total finance charge, and "
        "all fees associated with a loan prior to consummation. Disclosures shall be provided "
        "in writing and retained by the lender for {years} years.",
    ],
}

FILL_VALUES: dict[str, list[str]] = {
    "service": ["healthcare", "mental health", "substance abuse treatment", "home health"],
    "hours": ["10", "20", "30", "40"],
    "board": ["Medical", "Dental", "Nursing", "Pharmacy"],
    "facility": ["hospital", "nursing home", "assisted living", "outpatient clinic"],
    "substance": ["Schedule II controlled substance", "opioid analgesic", "benzodiazepine"],
    "treatment": ["mental health", "substance use disorder", "preventive care"],
    "subject": ["mathematics", "literacy", "science", "social studies"],
    "years": ["3", "5", "7", "10"],
    "fund": ["Education Excellence", "Student Success", "Innovation", "Equity"],
    "goods": ["information technology", "professional services", "construction", "supplies"],
    "threshold": ["25,000", "50,000", "100,000", "250,000"],
    "days": ["5", "10", "15", "30"],
    "agency": ["Environmental Protection", "Health Services", "Transportation", "Revenue"],
    "months": ["3", "6", "9", "12"],
    "pollutant": ["wastewater", "stormwater", "industrial effluent", "treated sewage"],
    "program": ["Brownfields Remediation", "Site Cleanup", "Environmental Restoration"],
    "amount": ["25,000", "50,000", "100,000", "500,000"],
    "authority": ["Transit", "Port", "Toll Road", "Airport"],
    "limit": ["25", "35", "45", "55"],
    "offense": ["aggravated assault", "felony theft", "drug trafficking", "fraud"],
    "crime": ["violent", "domestic violence", "sexual assault", "stalking"],
    "percent": ["30", "50", "60", "80"],
    "threshold_small": ["10,000", "15,000", "25,000"],
}


def _fill_template(template: str) -> str:
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def _cosine_from_bin(bin_label: str) -> float:
    ranges = {
        "q40_60": (0.40, 0.60),
        "q60_80": (0.60, 0.80),
        "q80_90": (0.80, 0.90),
        "q90_95": (0.90, 0.95),
        "q95_100": (0.95, 1.00),
    }
    lo, hi = ranges[bin_label]
    return round(random.uniform(lo, hi), 4)


def _score_from_cosine(cosine: float) -> int:
    if cosine >= 0.90:
        return random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    elif cosine >= 0.75:
        return random.choices([2, 3, 4], weights=[0.3, 0.45, 0.25])[0]
    elif cosine >= 0.60:
        return random.choices([3, 4, 5], weights=[0.2, 0.45, 0.35])[0]
    elif cosine >= 0.50:
        return random.choices([5, 6, 7], weights=[0.3, 0.45, 0.25])[0]
    else:
        return random.choices([6, 7], weights=[0.4, 0.6])[0]


def generate_pairs(n: int = 100, seed: int = 42) -> list[dict]:
    random.seed(seed)
    pairs = []
    bins = ["q40_60", "q60_80", "q80_90", "q90_95", "q95_100"]
    bin_weights = [0.10, 0.20, 0.30, 0.25, 0.15]

    for i in range(n):
        topic = random.choice(TOPICS)
        templates = TOPIC_TEMPLATES[topic]

        anchor_tmpl, candidate_tmpl = random.sample(templates, 2) if len(templates) >= 2 else (templates[0], templates[0])

        state_a = random.choice(STATES)
        remaining = [s for s in STATES if s != state_a]
        state_b = random.choice(remaining)

        bin_label = random.choices(bins, weights=bin_weights)[0]
        cosine = _cosine_from_bin(bin_label)
        score = _score_from_cosine(cosine)

        pairs.append({
            "anchor_id": f"{state_a}-{topic.lower().replace(' ', '_')}-{i:04d}",
            "candidate_id": f"{state_b}-{topic.lower().replace(' ', '_')}-{i:04d}",
            "text_a": _fill_template(anchor_tmpl),
            "text_b": _fill_template(candidate_tmpl),
            "similarity_score": score,
            "primary_topic_a": topic,
            "cosine_similarity": cosine,
            "bin": bin_label,
            "pair_type": "within",
            "state_a": state_a,
            "state_b": state_b,
        })

    return pairs


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "data" / "mock_pairs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = generate_pairs(n=100)
    with output_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    score_dist: dict[int, int] = {}
    for p in pairs:
        s = p["similarity_score"]
        score_dist[s] = score_dist.get(s, 0) + 1

    print(f"Wrote {len(pairs)} pairs to {output_path}")
    print(f"Score distribution: { {k: score_dist[k] for k in sorted(score_dist)} }")
    positives = sum(v for k, v in score_dist.items() if k <= 3)
    negatives = sum(v for k, v in score_dist.items() if 4 <= k <= 7)
    print(f"Positives (1-3): {positives}  |  Negatives (4-7): {negatives}")


if __name__ == "__main__":
    main()
