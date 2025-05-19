#!/usr/bin/env python3
"""
Simple runner for drone security analysis
"""

from my_agent import MyAgent
from config import gemini_config

# Configuration
VIDEO_PATH = "combined.mp4"
QUERY_FOR_DETECTION = "railway track"
Ask_any_question = "how many vechiles passed through railway gate."
OUTPUT_VIDEO = "output.mp4"
OUTPUT_CSV = "detections.csv"

def main():
    # Create and setup agent
    agent = MyAgent()
    agent.setup(gemini_config)
    
    # Run the analysis
    print(f"Processing video: {VIDEO_PATH}")
    print(f"Using query: {QUERY_FOR_DETECTION}")
    
    df, generated_description_with_system, Retrieved_logs = agent.run(
        input_video_path=VIDEO_PATH,
        query=QUERY_FOR_DETECTION,
        flag = "detections",
        output_video_path=OUTPUT_VIDEO,
        output_csv_path=OUTPUT_CSV
    )

    print("Retrieved logs for the query:", Retrieved_logs)
    
    _, generated_description_with_system, _ = agent.run(
        input_video_path=VIDEO_PATH,
        query=Ask_any_question,
        flag = "logs",
        Retrieved_logs = Retrieved_logs
    )
    
    print(generated_description_with_system)
    return 0

if __name__ == "__main__":
    exit(main())
