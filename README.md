# worker-evaluation-profiling-tool
AI-powered workforce evaluation tool using NLP
# Worker Evaluation and Profiling Tool

## Overview
This project is an AI-powered Workforce Intelligence Platform developed for CSC 546: Applied NLP Systems. It analyzes employee work logs using Natural Language Processing to support fair, data-driven performance evaluation and workforce profiling.

## Motivation
Traditional employee evaluation methods are often manual, time-consuming, and subjective. This project was designed to automate parts of that process by extracting meaningful insights from textual work logs.

The system helps organizations:
- Identify key contributors and underperformers
- Detect critical skill dependencies
- Generate performance summaries
- Support balanced team composition and training recommendations

## NLP Components
The platform uses:
- Named Entity Recognition (NER)
- Summarization
- Text Generation

These techniques are used to extract skills, summarize employee contributions, and generate analytical reports for managers.

## Data Source
The system works with a synthetic dataset (`synthetic_work_logs.csv`) containing structured and textual employee work log information.

### Example dataset fields
- TicketID
- Employee
- Project
- Task Description
- Skills
- Notes / Work Logs

## Technologies Used
- Python
- Streamlit
- NLP
- pandas
- transformers
- PyTorch

## Project Structure
- `MiniProject.py` - main application script
- `app/` - application components
- `Data/` - project data
- `HYB_data/` - hybrid model/data resources
- `SYN_data/` - synthetic data resources
- `Technical_Document.md` - technical documentation

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
2. Run the application:
streamlit run MiniProject.py
Open the local URL shown in the terminal.


Outcome
This project demonstrates skills in Natural Language Processing, data analysis, text summarization, entity extraction, and building interactive AI applications for organizational decision support.
