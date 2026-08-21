import streamlit as st
import time


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Drone Crowd Monitor",
    page_icon="🚁",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚁 AI Drone Crowd Monitoring System")

st.caption(
    "Real-Time AI-Based Crowd Monitoring"
)


# ==========================================
# SETTINGS
# ==========================================

MAX_PEOPLE = 10


# ==========================================
# SESSION STATE
# ==========================================

if "people" not in st.session_state:
    st.session_state.people = 0

if "peak" not in st.session_state:
    st.session_state.peak = 0

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "mission_status" not in st.session_state:
    st.session_state.mission_status = "READY"

if "battery" not in st.session_state:
    st.session_state.battery = None

if "obstacle" not in st.session_state:
    st.session_state.obstacle = "N/A"


# ==========================================
# STATUS
# ==========================================

people = st.session_state.people

occupancy = min(
    (people / MAX_PEOPLE) * 100,
    100
)


if people >= MAX_PEOPLE:

    crowd_status = "CROWDED 🔴"

elif people >= 7:

    crowd_status = "HIGH 🟡"

else:

    crowd_status = "NORMAL 🟢"


# ==========================================
# TOP METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "👥 People",
        f"{people}/{MAX_PEOPLE}"
    )


with col2:

    st.metric(
        "📊 Occupancy",
        f"{occupancy:.0f}%"
    )


with col3:

    st.metric(
        "🚦 Status",
        crowd_status
    )


with col4:

    st.metric(
        "📈 Peak People",
        st.session_state.peak
    )


st.divider()


# ==========================================
# LIVE CAMERA AREA
# ==========================================

left, right = st.columns([2, 1])


with left:

    st.subheader("📷 Live Camera")

    st.info(
        "Tello camera will appear here "
        "when connected."
    )


with right:

    st.subheader("📋 Mission Information")

    st.write(
        f"🚁 **Mission:** "
        f"{st.session_state.mission_status}"
    )

    if st.session_state.battery is None:

        st.write("🔋 **Battery:** N/A")

    else:

        st.write(
            f"🔋 **Battery:** "
            f"{st.session_state.battery}%"
        )

    st.write(
        f"🎯 **Avg Confidence:** "
        f"{st.session_state.confidence:.1f}%"
    )

    st.write(
        f"⚠️ **Obstacle:** "
        f"{st.session_state.obstacle}"
    )


st.divider()


# ==========================================
# ALERT
# ==========================================

if people >= MAX_PEOPLE:

    st.error(
        "🚨 CROWD ALERT — Hall capacity reached!"
    )

elif people >= 7:

    st.warning(
        "⚠️ HIGH CROWD LEVEL"
    )

else:

    st.success(
        "🟢 Crowd level is normal"
    )


# ==========================================
# TEST CONTROLS
# ==========================================

st.subheader("🧪 Testing Controls")

col1, col2, col3 = st.columns(3)


with col1:

    if st.button("👤 Add Person"):

        st.session_state.people += 1

        if (
            st.session_state.people
            > st.session_state.peak
        ):

            st.session_state.peak = (
                st.session_state.people
            )

        st.rerun()


with col2:

    if st.button("➖ Remove Person"):

        if st.session_state.people > 0:

            st.session_state.people -= 1

        st.rerun()


with col3:

    if st.button("🔄 Reset"):

        st.session_state.people = 0
        st.session_state.peak = 0

        st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Drone Crowd Monitoring System | "
    "YOLO + ByteTrack"
)