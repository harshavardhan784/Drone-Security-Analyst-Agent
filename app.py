import streamlit as st
from my_agent import MyAgent
from config import gemini_config

# Set Streamlit page config
st.set_page_config(page_title="Drone Security Analyst", layout="wide")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "logs" not in st.session_state:
    st.session_state.logs = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "output_video" not in st.session_state:
    st.session_state.output_video = None

st.title("🚨 Drone Security Analyst Agent")

st.sidebar.header("⚙️ Configuration")

video_file = st.sidebar.text_input("Video Path", "combined.mp4")
query = st.sidebar.text_input("Detection Query", "railway track")
output_video_path = st.sidebar.text_input("Output Video", "output.mp4")
output_csv_path = st.sidebar.text_input("Output CSV", "detections.csv")

run_detection = st.sidebar.button("🔍 Run Detection")

# Reset button
if st.sidebar.button("🔄 Reset"):
    st.session_state.agent = None
    st.session_state.logs = None
    st.session_state.video_processed = False
    st.session_state.csv_path = None
    st.session_state.output_video = None
    st.success("Session reset!")

# Step 1: Run detection and logging
if run_detection:
    st.session_state.agent = MyAgent()
    st.session_state.agent.setup(gemini_config)

    with st.spinner(f"Processing video: {video_file}"):
        df, generated_logs, logs = st.session_state.agent.run(
            input_video_path=video_file,
            query=query,
            flag="detections",
            output_video_path=output_video_path,
            output_csv_path=output_csv_path
        )
        st.session_state.logs = logs
        st.session_state.video_processed = True
        st.session_state.csv_path = output_csv_path
        st.session_state.output_video = output_video_path
        st.success("✅ Detection completed and logs generated!")

    with st.expander("📜 Detection Summary"):
        st.write(generated_logs)

    with st.expander("📂 Detection CSV"):
        st.dataframe(df)

# Step 2: Ask follow-up questions
if st.session_state.video_processed:
    st.subheader("🤖 Ask a Question Based on the Logs")
    ask_query = st.text_input("Ask anything (e.g., how many vehicles?)")
    if st.button("🔎 Get Answer"):
        if ask_query and st.session_state.logs:
            _, answer, _ = st.session_state.agent.run(
                input_video_path=video_file,
                query=ask_query,
                flag="logs",
                Retrieved_logs=st.session_state.logs
            )
            st.success("✅ Answer generated:")
            st.write(answer)
        else:
            st.warning("❗ Please run detection first or ask a valid question.")

# Show output video (if available)
if st.session_state.output_video:
    st.subheader("📼 Output Video with Detections")
    try:
        with open(st.session_state.output_video, "rb") as f:
            st.video(f.read())
    except Exception as e:
        st.warning(f"Could not load video: {e}")
