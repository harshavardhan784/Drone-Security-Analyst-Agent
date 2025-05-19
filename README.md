
# 🚨 Drone Security Analyst Agent

A functional prototype of an intelligent agent that monitors fixed property using a docked drone. It simulates real-time video and telemetry analysis to detect suspicious events, generate alerts, and provide contextual insights through AI-based querying and summarization.

---

## 🔍 Key Features

- **Automated Security Monitoring**: Detects objects (e.g., people, vehicles) using computer vision and logs them with contextual metadata.
- **Real-Time Alerting**: Triggers alerts based on predefined rules (e.g., loitering near main gate at night).
- **Semantic Log Querying**: Uses vector-based retrieval ([ChromaDB](https://www.trychroma.com/) + [LangChain](https://www.langchain.com/)) and [Sentence Transformers](https://www.sbert.net/) for intelligent querying over event logs.
- **AI-Driven Analysis**: LLM generates human-readable summaries and insights based on raw tracking data.
- **Streamlit UI**: Web-based interactive interface for log display, alert monitoring, and AI-assisted queries.

---

## 📦 Tech Stack

| Component            | Tool/Library                        | Purpose                                         |
|----------------------|-------------------------------------|-------------------------------------------------|
| 🧠 Object Detection   | [YOLOv11s](https://docs.ultralytics.com/tasks/detect/) | Real-time object detection and tracking        |
| 🧾 Log Generation     | OpenAI GPT-4 / Gemini Pro Vision     | Convert tracked data into human-readable logs and alerts |
| 🔎 Semantic Search    | [ChromaDB](https://python.langchain.com/docs/integrations/vectorstores/chroma/), [LangChain](https://www.langchain.com/), [all-MiniLM-L6-v2](https://www.sbert.net/) | Vector DB & contextual search |
| 🧰 Frameworks         | Python, OpenCV, Streamlit           | Frontend & backend integration                  |

---

## 🧩 System Architecture

```
Simulated Drone Data
    ├──> YOLOv11s Object Detection
    ├──> Event Tracker (Dataframe Logger)
    ├──> LLM Log Formatter
    ├──> Alert Engine
    ├──> ChromaDB + Sentence Transformers (Embeddings)
    └──> LangChain + RetrievalQA 
    └──> Question Answering
```

---

## 🧪 How It Works

### Input
- Video
```

### Output Examples
- ✅ **Log**: "Blue Ford F150 spotted at gate, 12:00."
- 🚨 **Alert**: "Person loitering at midnight near main gate!"
- 🔍 **Query**: "Show all truck events" → Returns all relevant logs.
- 🔍 **Query**: "Give me summary out of the video" → Returns the summary.

---

## 📁 Folder Structure

```plaintext
├── tools/
│   └── GeminiApiTool.py
├── utils/
│   ├── utils.py
├── my_agent.py
├── run.py
├── streamlit_ui.py
├── README.md
├── requirements.txt

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Drone-Security-Analyst-Agent.git
cd Drone-Security-Analyst-Agent
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export GEMINI_API_KEY="your_google_api_key"
```

Or create a `.env` file and add:
```
GEMINI_API_KEY=your_google_api_key
```

### 4. Run the Streamlit App

Make sure to update `video_path` and sample queries in `run.py` or `streamlit_ui.py` before launching:

```bash
streamlit run app.py
```

---

## 🧠 Design Decisions

- **LLM Integration**: GPT/Gemini powers summarization and querying, enabling rich semantic understanding.
- **Modular Tools**: Tools are separated in `tools/` and `utils/` to maintain clean architecture.
- **Vector Search**: Semantic search enhances user interaction by allowing natural language queries on past events.
- **Streamlit UI**: Offers a quick, lightweight interface for demonstrations and operator interactions.

---

## ✅ Benefits of the Pipeline

- High scalability for adding new alert rules or detection models.
- Easy to adapt from simulated to real drone telemetry feeds.
- Fast debugging and prototyping through MODULAR design.

---

## 📝 Improvements (Planned)

- Incorporate real drone telemetry and live video stream.
- Support multiple drone feeds and geo-mapping integration.
- Add timeline navigation for past logs.

---

## 📜 License

This project is proprietary and shared for evaluation.
