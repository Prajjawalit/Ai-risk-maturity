import re
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical AI Risk Maturity & Techno-Legal Suite | AIIMS, Rajkot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0B3C5D 0%, #1D65A6 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .main-header h2 { margin: 0; font-size: 1.5rem; color: #FFFFFF; }
    .main-header p { margin: 0.3rem 0 0 0; font-size: 0.92rem; color: #E0EEF8; }
    .risk-badge-red {
        background-color: #B71C1C; color: #FFFFFF; padding: 4px 10px;
        border-radius: 4px; font-weight: 700; font-size: 0.85rem;
    }
    .risk-badge-amber {
        background-color: #E65100; color: #FFFFFF; padding: 4px 10px;
        border-radius: 4px; font-weight: 700; font-size: 0.85rem;
    }
    .risk-badge-green {
        background-color: #1B5E20; color: #FFFFFF; padding: 4px 10px;
        border-radius: 4px; font-weight: 700; font-size: 0.85rem;
    }
    .audit-box {
        background-color: #F4F7FA; border-left: 4px solid #0B3C5D;
        padding: 0.85rem 1rem; border-radius: 4px; margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION (ISOLATED PER BROWSER SESSION FOR WORKSHOPS)
# -----------------------------------------------------------------------------
def init_state():
    if "inventory" not in st.session_state:
        st.session_state.inventory = pd.DataFrame(
            [
                {
                    "System ID": "AI-CLIN-001",
                    "Tool Name": "ChestXpert Triage Overlay",
                    "Department": "Radiology & Emergency",
                    "Provenance": "Procured Vendor SaMD",
                    "Risk Tier": "High-Risk (Clinical SaMD Class C)",
                    "Risk Appetite": "Very Low (Near-Zero Harm Tolerance)",
                    "AIBOM & Model Card": "Verified",
                    "Status": "Production (Monitored)",
                },
                {
                    "System ID": "AI-CLIN-002",
                    "Tool Name": "ICU Sepsis Early-Warning Score",
                    "Department": "Critical Care / ICU",
                    "Provenance": "Procured EHR Module",
                    "Risk Tier": "High-Risk (Clinical SaMD Class C)",
                    "Risk Appetite": "Very Low (Near-Zero Harm Tolerance)",
                    "AIBOM & Model Card": "Pending Local Bias Audit",
                    "Status": "Shadow Validation",
                },
                {
                    "System ID": "AI-SHAD-003",
                    "Tool Name": "Unvetted Public Chatbot + WhatsApp",
                    "Department": "Medicine / ICU Night Duty",
                    "Provenance": "Shadow AI (Personal Mobile)",
                    "Risk Tier": "High-Risk (Unregulated Clinical Use)",
                    "Risk Appetite": "Prohibited (DPDP & Safety Breach)",
                    "AIBOM & Model Card": "None",
                    "Status": "Blocked / Remediation Active",
                },
                {
                    "System ID": "AI-OPS-004",
                    "Tool Name": "OPD Queue & Slot Optimizer",
                    "Department": "Hospital Administration",
                    "Provenance": "In-House IT Build",
                    "Risk Tier": "Medium-Risk (Operational)",
                    "Risk Appetite": "Medium (Periodic SLA Review)",
                    "AIBOM & Model Card": "Verified",
                    "Status": "Production (Monitored)",
                },
            ]
        )

    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = [
            {
                "Timestamp": (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "User Role": "Senior Resident (ED)",
                "RAG Tier": "RED",
                "PHI Scrubbed": "Yes (3 entities)",
                "Clinician Action": "Overridden — Ordered Stat eFAST & CT",
                "Audit Hash": "9f82c41a7e",
            }
        ]

    if "incidents" not in st.session_state:
        st.session_state.incidents = [
            {
                "Incident ID": "INC-2026-014",
                "Detected At": (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
                "System": "ChestXpert Triage Overlay",
                "Severity": "Adverse Event / Near-Miss",
                "Statutory SLA": "< 12 Hours (Safety Committee) | CDSCO SaMD Vigilance",
                "Root Cause": "Automation bias on supine CXR; subtle widened mediastinum missed",
                "CAPA Status": "Mandatory Dual-Read Protocol Activated",
            }
        ]

    if "model_rolled_back" not in st.session_state:
        st.session_state.model_rolled_back = False


init_state()

# -----------------------------------------------------------------------------
# TECHNO-LEGAL CORE FUNCTIONS
# -----------------------------------------------------------------------------
def scrub_phi_dpdp(text: str):
    """Sanitizes clinical free-text for DPDP Act 2023 compliance before LLM inference."""
    redactions = []
    sanitized = text

    patterns = [
        (r"\b(?:Pt|Patient|Name)\s*[:\-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "[REDACTED_PATIENT_NAME]", "Patient Name"),
        (r"\b(?:IPD|OPD|UHID|MRN|Reg(?:\.|istration)?)\s*(?:No\.?|#|:)?\s*[A-Z0-9\-\/]{4,15}", "[REDACTED_HOSPITAL_ID]", "Hospital/IPD ID"),
        (r"\bBed\s*(?:No\.?|#|:)?\s*[A-Z0-9\-]{1,6}", "[REDACTED_BED_LOC]", "Bed Location"),
        (r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b", "[REDACTED_PHONE]", "Phone Number"),
        (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[REDACTED_GOV_ID]", "12-Digit Govt ID"),
    ]

    for pattern, token, label in patterns:
        matches = re.findall(pattern, sanitized, flags=re.IGNORECASE)
        if matches:
            redactions.extend([label] * len(matches))
            sanitized = re.sub(pattern, token, sanitized, flags=re.IGNORECASE)

    return sanitized, redactions


def classify_clinical_prompt(prompt: str, template_used: bool):
    """Assigns Red / Amber / Green risk level per PSA Techno-Legal & ELGM2 frameworks."""
    high_risk_terms = [
        "dose", "dosing", "mg", "mcg", "infusion", "triage", "yellow zone", "red zone",
        "rta", "gcs", "intubate", "sepsis", "icu", "ct head", "chest x-ray", "cxr",
        "diagnosis", "diagnose", "prescribe", "laparotomy", "thrombolysis", "antibiotic"
    ]
    medium_risk_terms = [
        "discharge summary", "soap note", "handover", "translate", "patient education",
        "icd-10", "coding", "referral letter", "literature"
    ]

    lower_p = prompt.lower()
    if any(term in lower_p for term in high_risk_terms):
        tier = "RED"
        rationale = (
            "Direct clinical triage, diagnostic interpretation, or therapeutic dosing detected. "
            "Requires mandatory Human-in-the-Loop verification and confidence calibration display."
        )
    elif any(term in lower_p for term in medium_risk_terms) or not template_used:
        tier = "AMBER"
        rationale = (
            "Clinical documentation or unstandardized free-text prompt detected. "
            "Requires clinician review for confabulation/hallucination before EHR entry."
        )
    else:
        tier = "GREEN"
        rationale = "Administrative or non-clinical operational query with zero direct patient harm exposure."

    return tier, rationale


def generate_guarded_clinical_response(sanitized_prompt: str):
    """Simulates a safety-aligned clinical LLM with uncertainty quantification & anti-sycophancy."""
    lower_p = sanitized_prompt.lower()

    if "gcs" in lower_p or "rta" in lower_p or "yellow zone" in lower_p:
        return {
            "response": (
                "⚠️ **CLINICAL SAFETY COUNTER-CHECK (ANTI-SYCOPHANCY GUARDRAIL ACTIVE):**\n\n"
                "1. **Triage Escalation Alert:** Mechanism of injury (unhelmeted high-speed RTA) + fluctuating GCS (13–14) + "
                "tachycardia (HR >100) + thoracoabdominal pain **contraindicates Yellow-Zone observation**.\n"
                "2. **Immediate ATLS Protocol:** Assign **RED (Resuscitation) Zone**, immobilize C-spine, initiate supplemental O₂, "
                "secure 2 large-bore IVs, and perform **stat eFAST** + trauma CT series.\n"
                "3. **Automation Bias Warning:** A negative AI chest X-ray overlay does **not** rule out occult hemopneumothorax, "
                "mediastinal injury, or splenic laceration."
            ),
            "ece_score": 0.14,
            "confidence": "68% (High Epistemic Uncertainty — Blunt Trauma Edge Case)",
            "ius_warning": "Off-Label Restriction: Not validated for standalone trauma triage or pediatric (<18 yrs) patients.",
        }
    elif "sepsis" in lower_p or "lactate" in lower_p or "icu" in lower_p:
        return {
            "response": (
                "⚠️ **DECISION-SUPPORT ADVISORY (NOT A CLINICAL ORDER):**\n\n"
                "1. **Sepsis Screening:** Elevated lactate and hemodynamic instability warrant immediate **qSOFA / SOFA** bedside reassessment "
                "and Surviving Sepsis 1-hour bundle verification by the attending intensivist.\n"
                "2. **Historical Failure Mode Notice:** Commercial sepsis predictors have demonstrated up to a **67% miss rate** in external "
                "validations; never delay blood cultures or empiric antimicrobials based on a low AI risk score.\n"
                "3. **Privacy Enforcement:** Patient identifiers were automatically stripped prior to inference under DPDP Act 2023 rules."
            ),
            "ece_score": 0.11,
            "confidence": "74% (Moderate Uncertainty — Requires Serial Biomarker Correlation)",
            "ius_warning": "Intended Use Statement: Adjunct screening only for adult medical ICU patients; exclude pregnancy/pediatrics.",
        }
    else:
        return {
            "response": (
                "ℹ️ **STRUCTURED CLINICAL GOVERNANCE OUTPUT:**\n\n"
                "Draft generated using institutional evidence-aligned guardrails. Verify all drug dosages, "
                "contraindications, and patient-specific comorbidities against primary clinical records before action."
            ),
            "ece_score": 0.06,
            "confidence": "86% (Standard Administrative/Documentation Confidence)",
            "ius_warning": "Intended Use Statement: Administrative/documentation drafting support; clinician sign-off mandatory.",
        }


# -----------------------------------------------------------------------------
# HEADER & SIDEBAR
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h2>🛡️ Clinical AI Risk Maturity & Techno-Legal Governance Suite</h2>
        <p>Dr Pradip Chauhan, Additional Professor | Institutional AI Governance Workshop — AIIMS, Rajkot</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("🏥 Workshop & Governance Console")
    team_tag = st.text_input("Participant / Team ID (Optional)", placeholder="e.g., Team 03 — Emergency Med")
    if team_tag:
        st.caption(f"Active Sandbox Session: **{team_tag}**")

    st.info(
        "**Regulatory & Policy Anchors:**\n"
        "- CDSCO Medical Device Software (SaMD) Guidance\n"
        "- Digital Personal Data Protection (DPDP) Act, 2023\n"
        "- IndiaAI Governance Guidelines (Seven Sutras)\n"
        "- CERT-In 6-Hour Cyber Incident Mandate"
    )
    st.metric("Registered AI Systems", len(st.session_state.inventory))
    st.metric("Logged Audit Events", len(st.session_state.audit_logs))
    st.metric("Active Safety Incidents", len(st.session_state.incidents))

    st.divider()
    if st.button("Reset My Session Sandbox", use_container_width=True):
        for key in ["inventory", "audit_logs", "incidents", "model_rolled_back", "last_prompt"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ AI System Inventory & Risk Tiering",
        "2️⃣ Responsible AI Firewall (RAG Gateway)",
        "3️⃣ Post-Deployment Drift & Fairness Monitor",
        "4️⃣ Adverse Event RCA & HAIRA Maturity Audit",
    ]
)

# =============================================================================
# TAB 1: AI SYSTEM INVENTORY & RISK CLASSIFICATION
# =============================================================================
with tab1:
    st.subheader("Institutional AI System Registry & Risk Appetite Mapping")
    st.caption(
        "Discovery anchors every governance dimension: no control can protect patients from an uninventoried or Shadow AI tool."
    )

    st.dataframe(st.session_state.inventory, use_container_width=True, hide_index=True)

    csv_inv = st.session_state.inventory.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Current AI System Inventory (CSV)",
        data=csv_inv,
        file_name="aiims_rajkot_ai_system_inventory.csv",
        mime="text/csv",
    )

    with st.expander("➕ Register / Onboard New AI System into Institutional Inventory", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_name = st.text_input("AI Tool / Model Name", placeholder="e.g., Diabetic Retinopathy Fundus AI")
            new_dept = st.selectbox(
                "Clinical / Operational Department",
                ["Ophthalmology", "Radiology", "Pathology", "Emergency Medicine", "Critical Care / ICU", "Hospital Admin / HR"],
            )
        with col2:
            new_prov = st.selectbox(
                "Provenance / Acquisition Channel",
                ["Procured Vendor SaMD", "In-House Academic Model", "Open-Source Fine-Tuned LLM", "Shadow AI (Unapproved Browser/App)"],
            )
            new_tier = st.selectbox(
                "NCAIC / CDSCO Risk Classification",
                [
                    "High-Risk (Clinical SaMD Class C/D)",
                    "High-Risk (Clinical SaMD Class B)",
                    "Medium-Risk (Operational / Admin)",
                    "Low-Risk (Productivity / Non-PHI)",
                    "Prohibited (Social Scoring / Emotion Inference)",
                ],
            )
        with col3:
            new_aibom = st.selectbox(
                "AIBOM & Model Card Status",
                ["Verified & Signed", "Pending Local Validation", "Incomplete / Missing"],
            )
            submitted = st.button("Add to Registry & Assign Controls", type="primary")

        if submitted and new_name.strip():
            if "Prohibited" in new_tier or "Shadow AI" in new_prov:
                appetite = "Zero Tolerance (Immediate Block / Quarantine)"
                status = "Blocked Pending AIREC Review"
            elif "High-Risk" in new_tier:
                appetite = "Very Low (Pre-Deployment RCT/Audit + Mandatory HITL)"
                status = "Shadow Validation Gate"
            elif "Medium-Risk" in new_tier:
                appetite = "Medium (Vendor SLA + Quarterly KRI Audit)"
                status = "Conditional Production"
            else:
                appetite = "High (Annual Review + No-PHI Policy)"
                status = "Approved Production"

            new_row = pd.DataFrame(
                [
                    {
                        "System ID": f"AI-REG-{len(st.session_state.inventory) + 1:03d}",
                        "Tool Name": new_name.strip(),
                        "Department": new_dept,
                        "Provenance": new_prov,
                        "Risk Tier": new_tier,
                        "Risk Appetite": appetite,
                        "AIBOM & Model Card": new_aibom,
                        "Status": status,
                    }
                ]
            )
            st.session_state.inventory = pd.concat([st.session_state.inventory, new_row], ignore_index=True)
            st.success(f"Registered '{new_name}' with automated risk appetite control: {appetite}")
            st.rerun()

# =============================================================================
# TAB 2: RESPONSIBLE AI FIREWALL (RED-AMBER-GREEN INFERENCE GATEWAY)
# =============================================================================
with tab2:
    st.subheader("Inline Responsible AI Firewall (Techno-Legal Runtime Enforcement)")
    st.caption(
        "Demonstrates DPDP Act 2023 automated PII redaction, Red–Amber–Green risk tagging, uncertainty calibration, and mandatory Human-in-the-Loop sign-off."
    )

    preset = st.radio(
        "Select Clinical Simulation Scenario or Enter Custom Red-Team Prompt:",
        [
            "Case 1: ICU WhatsApp Shadow-AI Note (DPDP PHI Breach + Sepsis Risk)",
            "Case 2: 02:05 AM ED Trauma RTA Query (Automation Bias & Triage Trap)",
            "Custom Live Red-Team Input",
        ],
        horizontal=True,
    )

    if preset.startswith("Case 1"):
        default_prompt = (
            "Patient: Ramesh Patel, IPD: AIIMS/2026/88412, Bed: ICU-04, Phone: 9876543210. "
            "62M post-op sepsis, BP 88/54, Lactate 4.6 mmol/L. Can we hold broad-spectrum escalation "
            "and observe overnight since the AI sepsis score is green?"
        )
        default_template = False
    elif preset.startswith("Case 2"):
        default_prompt = (
            "37M bike RTA no helmet, GCS 14 confused, BP 110/70, HR 108, left chest and mild belly pain. "
            "AI CXR overlay says clear. Should we keep in Yellow Zone and defer CT abdomen?"
        )
        default_template = False
    else:
        default_prompt = ""
        default_template = True

    c_left, c_right = st.columns([1.15, 1])

    with c_left:
        user_prompt = st.text_area("Clinician / Resident Query Input:", value=default_prompt, height=130)
        template_flag = st.checkbox("Using Approved Institutional Prompt Library Template", value=default_template)
        run_firewall = st.button("Execute Techno-Legal Gateway Check", type="primary")

    if run_firewall and user_prompt.strip():
        st.session_state.last_prompt = user_prompt

    if "last_prompt" in st.session_state and st.session_state.last_prompt.strip():
        sanitized_text, redacted_items = scrub_phi_dpdp(st.session_state.last_prompt)
        rag_tier, rag_reason = classify_clinical_prompt(sanitized_text, template_flag)
        ai_out = generate_guarded_clinical_response(sanitized_text)

        with c_right:
            st.markdown("#### 🔒 Gateway Telemetry & Compliance Gate")
            badge_class = (
                "risk-badge-red" if rag_tier == "RED" else ("risk-badge-amber" if rag_tier == "AMBER" else "risk-badge-green")
            )
            st.markdown(f"**Assigned Risk Tag:** <span class='{badge_class}'>{rag_tier} TIER</span>", unsafe_allow_html=True)
            st.write(f"**Policy Rationale:** {rag_reason}")
            if redacted_items:
                st.warning(f"**DPDP Act 2023 Scrubber Triggered:** Stripped {len(redacted_items)} PHI identifiers ({', '.join(set(redacted_items))}).")
            else:
                st.info("**DPDP Act 2023 Scrubber:** Zero direct patient identifiers detected in prompt.")
            st.text_area("Sanitized Payload Sent to Model:", value=sanitized_text, height=90, disabled=True)

        st.divider()
        out_col1, out_col2 = st.columns([1.3, 1])
        with out_col1:
            st.markdown("#### 🤖 Guarded Model Output & Calibration Metadata")
            st.warning(ai_out["response"])
            st.caption(f"**Calibration & Uncertainty:** {ai_out['confidence']} | **ECE:** {ai_out['ece_score']}")
            st.error(ai_out["ius_warning"])

        with out_col2:
            st.markdown("#### 👨‍⚕️ Mandatory Human-in-the-Loop (HITL) Sign-Off")
            st.markdown(
                "<div class='audit-box'><b>Accountability Rule:</b> AI output is advisory only. "
                "Clinicians must log verification or override rationale into the immutable audit trail.</div>",
                unsafe_allow_html=True,
            )
            clinician_role = st.selectbox("Signing Clinician Role", ["Consultant / Faculty", "Senior Resident", "Junior Resident"])
            decision = st.radio(
                "Clinical Disposition:",
                [
                    "Override AI — Escalate to Red Zone / Stat Diagnostics",
                    "Modify Plan — Apply Bedside Clinical Judgment",
                    "Verify & Accept — Matches Institutional Standard of Care",
                ],
            )
            override_note = st.text_input("Clinical Justification / Override Note:", value="Bedside ATLS & clinical exam prioritized over AI score.")
            if st.button("Commit Signed Decision to Immutable Audit Log"):
                raw_record = f"{datetime.now().isoformat()}|{clinician_role}|{rag_tier}|{decision}|{override_note}"
                audit_hash = hashlib.sha256(raw_record.encode()).hexdigest()[:10]
                st.session_state.audit_logs.insert(
                    0,
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "User Role": f"{clinician_role} ({team_tag})" if team_tag else clinician_role,
                        "RAG Tier": rag_tier,
                        "PHI Scrubbed": f"Yes ({len(redacted_items)})" if redacted_items else "Clean",
                        "Clinician Action": f"{decision} ({override_note})",
                        "Audit Hash": audit_hash,
                    },
                )
                st.success(f"Decision cryptographically logged with Audit Hash `{audit_hash}`.")

    st.markdown("#### 📜 Immutable Audit Trail (Recent Gateway Transactions)")
    st.dataframe(pd.DataFrame(st.session_state.audit_logs), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: POST-DEPLOYMENT SURVEILLANCE, FAIRNESS & DRIFT MONITOR
# =============================================================================
with tab3:
    st.subheader("Post-Deployment Surveillance: Drift Detection, Subgroup Equity & 15-Min Rollback")
    st.caption(
        "Simulates silent model degradation and demographic sensitivity disparities (e.g., regional/rural population shift or scanner hardware update)."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        drift_pct = st.slider("Simulated Population / Scanner Covariate Drift (%)", 0, 40, 22)
    with ctrl2:
        subgroup_penalty = st.slider("Underrepresented Subgroup Sensitivity Drop (%)", 0, 30, 16)
    with ctrl3:
        st.write("")
        if st.session_state.model_rolled_back:
            st.success("✅ **ROLLBACK ACTIVE:** Reverted to Validated Baseline v1.2")
            if st.button("Restore Live v2.0 Model (Test Mode)"):
                st.session_state.model_rolled_back = False
                st.rerun()
        else:
            if st.button("🚨 Execute 15-Min Automated Model Rollback", type="primary"):
                st.session_state.model_rolled_back = True
                st.session_state.incidents.insert(
                    0,
                    {
                        "Incident ID": f"INC-2026-{len(st.session_state.incidents) + 15:03d}",
                        "Detected At": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "System": "ChestXpert Triage Overlay v2.0",
                        "Severity": "KRI Breach — Demographic Sensitivity Drift",
                        "Statutory SLA": "< 12 Hours (Safety Committee) | Model Rolled Back in < 15 Mins",
                        "Root Cause": f"Covariate shift ({drift_pct}%) caused subgroup sensitivity drop",
                        "CAPA Status": "Reverted to v1.2 + Mandatory Radiologist Over-Read",
                    },
                )
                st.rerun()

    eff_drift = 0 if st.session_state.model_rolled_back else drift_pct
    eff_drop = 0 if st.session_state.model_rolled_back else subgroup_penalty

    ref_sensitivity = round(max(0.60, 0.94 - (eff_drift * 0.002)), 3)
    sub_sensitivity = round(max(0.50, ref_sensitivity - (eff_drop / 100.0)), 3)
    disparate_impact = round(sub_sensitivity / ref_sensitivity, 3)
    eop_diff = round(ref_sensitivity - sub_sensitivity, 3)
    ece_val = round(0.04 + (eff_drift * 0.004), 3)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reference Cohort Sensitivity (TPR)", f"{ref_sensitivity:.1%}")
    m2.metric("Subgroup Sensitivity (TPR)", f"{sub_sensitivity:.1%}", delta=f"-{eop_diff:.1%}" if eop_diff > 0 else "0.0%", delta_color="inverse")
    m3.metric("Disparate Impact Ratio (Target ≥ 0.80)", f"{disparate_impact:.2f}", delta="PASS" if disparate_impact >= 0.80 else "BREACH", delta_color="normal" if disparate_impact >= 0.80 else "inverse")
    m4.metric("Expected Calibration Error (ECE ≤ 0.08)", f"{ece_val:.3f}", delta="CALIBRATED" if ece_val <= 0.08 else "DRIFT ALERT", delta_color="normal" if ece_val <= 0.08 else "inverse")

    if disparate_impact < 0.80 or ece_val > 0.08:
        st.error(
            "🚨 **KRI THRESHOLD BREACH DETECTED:** Subgroup Disparate Impact < 0.80 or Calibration Drift > 0.08. "
            "Triggering automated rollback and notifying AI Risk Lead & Hospital Safety Committee (< 12 hrs)."
        )

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.markdown("**Subgroup Sensitivity Comparison (Equal Opportunity Check)**")
        fairness_df = pd.DataFrame(
            {
                "Cohort": ["Urban Reference Cohort", "Regional / Rural Subgroup"],
                "Sensitivity (TPR)": [ref_sensitivity, sub_sensitivity],
            }
        ).set_index("Cohort")
        st.bar_chart(fairness_df)

    with chart_c2:
        st.markdown("**6-Month Rolling Sensitivity Trajectory**")
        months = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        trend_ref = [0.94, 0.94, 0.93, 0.93, 0.92, ref_sensitivity]
        trend_sub = [0.92, 0.91, 0.89, 0.86, 0.82, sub_sensitivity]
        trend_df = pd.DataFrame(
            {"Reference Cohort": trend_ref, "Regional Subgroup": trend_sub},
            index=months,
        )
        st.line_chart(trend_df)

# =============================================================================
# TAB 4: ADVERSE EVENT PROTOCOL & HAIRA WEAKEST-LINK MATURITY SCORECARD
# =============================================================================
with tab4:
    st.subheader("Part A: 4-Stage AI Adverse Event Protocol (Detect → Report → Investigate → CAPA)")
    st.dataframe(pd.DataFrame(st.session_state.incidents), use_container_width=True, hide_index=True)

    with st.expander("📝 Log New AI Adverse Event / Near-Miss"):
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            inc_sys = st.selectbox("Affected AI System", st.session_state.inventory["Tool Name"].tolist())
            inc_type = st.selectbox(
                "Event Classification",
                [
                    "Near-Miss / No-Harm (< 24h to AI Risk Lead)",
                    "Clinical Adverse Event (< 12h to Safety Committee)",
                    "Sentinel AI Harm (ASAP < 12h + CDSCO SaMD Report)",
                    "Data Privacy / Cyber Breach (< 6h CERT-In + DPDP Board)",
                ],
            )
        with ic2:
            inc_rca = st.text_input("Root Cause (Systemic / Workflow / Drift)", value="Unannounced vendor threshold change + alert fatigue")
        with ic3:
            inc_capa = st.text_input("CAPA Action", value="Enforce vendor change-notification SLA + recalibrate alert threshold")
            if st.button("Submit Incident & Trigger Statutory SLA"):
                st.session_state.incidents.insert(
                    0,
                    {
                        "Incident ID": f"INC-2026-{len(st.session_state.incidents) + 15:03d}",
                        "Detected At": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "System": inc_sys,
                        "Severity": inc_type.split(" (")[0],
                        "Statutory SLA": inc_type.split("(")[1].rstrip(")"),
                        "Root Cause": inc_rca,
                        "CAPA Status": inc_capa,
                    },
                )
                st.success("Incident logged and statutory escalation timers initiated.")
                st.rerun()

    st.divider()
    st.subheader("Part B: Institutional AI Risk Maturity Audit (48-Point Checklist + HAIRA Weakest-Link Rule)")
    st.caption(
        "Score each governance domain (0 = Not in place, 1 = Partially in place, 2 = Fully implemented & evidenced). "
        "Per the HAIRA framework, overall maturity is capped by the lowest-scoring domain ('Weakest-Link Rule')."
    )

    domains = {
        "1. Governance Architecture (6 items, Max 12)": (6, 8),
        "2. Clinical Safety Governance (5 items, Max 10)": (5, 6),
        "3. Data Governance & DPDP Compliance (4 items, Max 8)": (4, 5),
        "4. Vendor & Third-Party Risk (4 items, Max 8)": (4, 4),
        "5. Workforce Literacy & Human Factors (3 items, Max 6)": (3, 4),
        "6. Regulatory Alignment & Reporting (2 items, Max 4)": (2, 2),
    }

    domain_scores = {}
    domain_pcts = {}
    cols = st.columns(3)
    for idx, (d_name, (n_items, default_val)) in enumerate(domains.items()):
        max_pts = n_items * 2
        with cols[idx % 3]:
            val = st.slider(d_name, min_value=0, max_value=max_pts, value=default_val)
            domain_scores[d_name] = val
            domain_pcts[d_name] = val / max_pts

    total_score = sum(domain_scores.values())
    weakest_domain = min(domain_pcts, key=domain_pcts.get)
    weakest_pct = domain_pcts[weakest_domain]

    if total_score <= 12:
        agg_band = "Band 1: Ad Hoc (0–12)"
    elif total_score <= 24:
        agg_band = "Band 2: Aware (13–24)"
    elif total_score <= 36:
        agg_band = "Band 3: Defined (25–36)"
    elif total_score <= 44:
        agg_band = "Band 4: Managed (37–44)"
    else:
        agg_band = "Band 5: Optimised (45–48)"

    if weakest_pct < 0.25:
        haira_level = "Level 1: Initial / Ad Hoc"
    elif weakest_pct < 0.50:
        haira_level = "Level 2: Defined (Basic Oversight)"
    elif weakest_pct < 0.75:
        haira_level = "Level 3: Established (Standardized Validation)"
    elif weakest_pct < 0.90:
        haira_level = "Level 4: Advanced (Proactive Real-Time Governance)"
    else:
        haira_level = "Level 5: Leading (Center of Excellence)"

    r1, r2, r3 = st.columns(3)
    r1.metric("Aggregate Checklist Score (Max 48)", f"{total_score} / 48", delta=agg_band)
    r2.metric("HAIRA True Maturity (Weakest-Link Cap)", haira_level)
    r3.metric("Bottleneck Domain", weakest_domain.split(" (")[0], delta=f"{weakest_pct:.0%} Maturity", delta_color="inverse")

    st.bar_chart(pd.Series({k.split(" (")[0]: round(v * 100, 1) for k, v in domain_pcts.items()}))
