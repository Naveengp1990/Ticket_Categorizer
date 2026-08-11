import re
import joblib
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.60

ROUTING_MAP = {
    "billing": "Billing Department",
    "technical": "Technical Support",
    "hr": "HR Department",
    "general": "General Support"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "ticket_categorizer_model.joblib"
    )


model = load_model()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PRIORITY DETECTION
# ============================================================

def detect_priority(text):

    urgent_keywords = [
        "urgent",
        "asap",
        "critical",
        "emergency",
        "down",
        "not working",
        "cannot access",
        "unable to access"
    ]

    text = text.lower()

    if any(
        keyword in text
        for keyword in urgent_keywords
    ):
        return "URGENT"

    return "NORMAL"


# ============================================================
# TICKET PROCESSING
# ============================================================

def process_ticket(subject, body):

    text = f"{subject} {body}"

    cleaned_text = clean_text(text)

    # Edge case
    if len(cleaned_text.split()) < 2:

        return {
            "category": "unknown",
            "confidence": 0.0,
            "priority": "NORMAL",
            "decision": "NEEDS HUMAN REVIEW",
            "route_to": "Manual Review Queue"
        }

    # Prediction
    prediction = model.predict(
        [cleaned_text]
    )[0]

    # Confidence
    probabilities = model.predict_proba(
        [cleaned_text]
    )[0]

    confidence = probabilities.max()

    # Priority
    priority = detect_priority(text)

    # Routing
    if confidence < CONFIDENCE_THRESHOLD:

        decision = "NEEDS HUMAN REVIEW"
        route = "Manual Review Queue"

    else:

        decision = "AUTO-ASSIGN"

        route = ROUTING_MAP.get(
            prediction,
            "Manual Review Queue"
        )

    return {
        "category": prediction,
        "confidence": confidence * 100,
        "priority": priority,
        "decision": decision,
        "route_to": route
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Auto Ticket Categorizer",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 Auto Email / Ticket Categorizer")

st.write(
    "Automatically classify incoming support tickets "
    "into Billing, Technical, HR, or General."
)

st.divider()

subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Payment charged twice"
)

body = st.text_area(
    "Ticket Body",
    placeholder="Describe the issue...",
    height=180
)

if st.button(
    "🔍 Categorize Ticket",
    type="primary"
):

    if not subject.strip() and not body.strip():

        st.warning(
            "Please enter a ticket subject or body."
        )

    else:

        result = process_ticket(
            subject,
            body
        )

        st.divider()

        st.subheader(
            "Classification Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Category",
                result["category"].upper()
            )

            st.metric(
                "Priority",
                result["priority"]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{result['confidence']:.2f}%"
            )

            st.metric(
                "Decision",
                result["decision"]
            )

        st.info(
            f"📍 Route to: {result['route_to']}"
        )

        if result["decision"] == "NEEDS HUMAN REVIEW":

            st.warning(
                "⚠️ Confidence is below the 60% "
                "threshold. This ticket should be "
                "reviewed manually."
            )
