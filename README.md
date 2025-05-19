# 🚨 Drone Security Analyst Agent

A functional prototype of an intelligent agent that monitors fixed property using a docked drone. It simulates real-time video and telemetry analysis to detect suspicious events, generate alerts, and provide contextual insights through AI-based querying and summarization.

---

## 🔍 Key Features

- **Automated Security Monitoring**: Detects objects (e.g., people, vehicles) using computer vision and logs them with contextual metadata.
- **Real-Time Alerting**: Triggers alerts based on predefined rules (e.g., loitering near main gate at night).
- **Semantic Log Querying**: Uses vector-based retrieval ([ChromaDB](https://www.trychroma.com/) + [LangChain](https://www.langchain.com/)) and [Sentence Transformers](https://www.sbert.net/) for intelligent querying over event logs.
- **AI-Driven Analysis**: LLM generates human-readable summaries and insights based on raw tracking data.

---

## 📦 Tech Stack

| Component            | Tool/Library                        | Purpose                                         |
|----------------------|-------------------------------------|-------------------------------------------------|
| 🧠 Object Detection   | [YOLOv11s](https://docs.ultralytics.com/tasks/detect/) | Real-time object detection and tracking        |
| 🧾 Log Generation     | OpenAI GPT-4 / Gemini Pro Vision     | Convert tracked data into human-readable logs and alerts |
| 🔎 Semantic Search    | [ChromaDB](https://www.trychroma.com/), [LangChain](https://www.langchain.com/), [all-MiniLM-L6-v2](https://www.sbert.net/) | Vector DB & contextual search |
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
    └──> LangChain + RetrievalQA → User Interface
```

---

## 🧪 How It Works

### Simulated Input
```python
frames = [
    { "frame": 1, "description": "Blue Ford F150 at gate", "time": "12:00", "location": "gate" },
    { "frame": 2, "description": "Person loitering at midnight", "time": "00:01", "location": "main gate" },
]
```

### Output Examples
- ✅ **Log**: "Blue Ford F150 spotted at gate, 12:00."
- 🚨 **Alert**: "Person loitering at midnight near main gate!"
- 🔍 **Query**: "Show all truck events" → Returns all relevant logs.

---

## 📁 Folder Structure

```
├── Agents/
│   └── AgentBase.py
├── tools/
│   └── GeminiTool.py
├── utils/
│   ├── utils.py
│   ├── utils_traceablity.py
│   └── utils_agent.py
├── my_agent.py
├── app.py
├── README.md
├── requirements.txt
└── tests/
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit UI

```bash
streamlit run app.py
```

---

## 🔐 Setup Environment Variables

```bash
export GOOGLE_API_KEY="your_google_api_key"
export OPENAI_API_KEY="your_openai_key"
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📌 Improvements

- Add video summarization
- Natural language Q&A from logs
- Real-time drone integration (e.g., MQTT, ROS)

---

## 📜 License

This project is proprietary and shared for evaluation only.
