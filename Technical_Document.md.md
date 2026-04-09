Worker Evaluation and Profiling Tool
Course: CSC 546- Applied NLP Systems
Team: C
Repository: https://github.com/connorsg-cua/NLP_CSC546_Project2
Submission Date: November 13, 2025
1. Problem Brief
Motivation
In modern organizations, analyzing employee productivity and identifying strengths or
improvement areas are crucial for maintaining a competitive advantage. Traditional evaluation
methods are often manual, time-consuming, and subjective, which can lead to bias and
inconsistency.
This project introduces an AI-powered Workforce Intelligence Platform that uses Natural
Language Processing (NLP) to automatically analyze textual work logs. By applying Named
Entity Recognition (NER), Summarization, and Text Generation, the system extracts skills,
summarizes employee contributions, and generates analytical reports for managers.
The ultimate goal is to create a data-driven, fair, and explainable performance evaluation
system, helping organizations to:
Identify key contributors and underperformers.
Detecting critical skill dependencies.
Suggest balanced team compositions and training recommendations.
Data Sources
The system works with a synthetic dataset (synthetic_work_logs.csv) containing structured and
textual information about employee work logs.
Dataset Columns:
Column Description
TicketID Unique identifier for each task
Employee Employee assigned to the task
Project Project name
TaskCategory Task type (e.g., Testing, Design, Backend)

Description Textual task description analyzed by NLP
Week Week of task
Status Task completion status
ProgressPct Task progress percentage
EstimatedHours Estimated work hours
StartDate Task start date
EndDate Task end date
ManagerNote Optional manager note or comment

Ethical Considerations:
All data is synthetic, ensuring no personal or confidential information is included. This complies
with ethical data-handling practices for AI research.
2. System Architecture
The application integrates three main NLP components — NER, Summarization, and (future)
Text Generation — within an interactive Streamlit web app.
Architecture Diagram
+-----------------+ +----------------------+ +-------------------------+
| Data Input |-----&gt;| NER Skill Extractor |-----&gt;| Skill-based Profiling |
| (CSV Work Logs) | | (dslim/bert-base-NER)| | (Employee Skill Matrix) |
+-----------------+ +----------------------+ +-------------------------+
|
v
+----------------------+ +---------------------+ +---------------------+
| Summarization Engine |-----&gt;| Employee Summaries |-----&gt;| Performance Reports |
| (sshleifer/distilbart-cnn-12-6)| | (Concise Profiles) | | (Manager Dashboards)|
+----------------------+ +---------------------+ +---------------------+
|
v
+----------------------+
| Text Generation |
| ( Qwen/Qwen2-1.5B-Instruct)|
+----------------------+

Pipeline Stages
Data Input:
CSV file uploaded by the user via Streamlit sidebar (file_uploader).
NER Skill Extractor:
Uses dslim/bert-base-NER with aggregation_strategy=&quot;simple&quot; to detect named entities and
filter relevant technical skills (e.g., Python, Docker, AWS).
Skill-based Profiling:
Constructs a Skill Matrix per employee, showing unique skills, total count, and frequency
across projects.
Summarization Engine:
Uses sshleifer/distilbart-cnn-12-6 to produce concise summaries of employee contributions by
concatenating their task descriptions.
Employee Summaries:
Each employee receives a summary paragraph describing main accomplishments.
Performance Reports:
Generates leaderboards, insights, and downloadable text reports that combine NER +
Summarization results.
Text Generation
Uses Qwen/Qwen2-1.5B-Instruct to generate synthetic employee logs.
3. Technical Documentation
3.1 Models Integrated
1. Named Entity Recognition (NER)
Model: dslim/bert-base-NER
Type: Encoder-based BERT model fine-tuned for entity classification.
Purpose: Extract skill and technology mentions from textual task descriptions.
Implementation Highlights:
ner_pipe = pipeline(&quot;ner&quot;, model=&quot;dslim/bert-base-NER&quot;, aggregation_strategy=&quot;simple&quot;)
entities = ner_pipe(text)
Code uses the dslim/bert-base-NER to identify employees and skills in the Enhanced NER
Analysis tab, which displays the confidence scores associated with each.
If the pipeline fails, the code defaults to a rules-based NER model, where results are post-
processed using a custom filtering mechanism that matches keywords (Python, Docker, Cloud,
ML, etc.).
Output Example:
[&quot;python&quot;, &quot;docker&quot;, &quot;aws&quot;, &quot;sql&quot;]
Limitations:

May miss domain-specific skill names or capture irrelevant entities. Keyword filtering improves
accuracy.
2. Text Summarization
Model: sshleifer/distilbart-cnn-12-6
Type: Encoder–decoder transformer (BART architecture).
Purpose: Generate concise summaries of employees’ overall contributions.
Implementation:
summarizer = pipeline(&quot;summarization&quot;, model=“sshleifer/distilbart-cnn-12-6”,
tokenizer=“sshleifer/distilbart-cnn-12-6”)
summary = summarizer(long_text, max_length=120, min_length=30,
do_sample=False)[0][&#39;summary_text&#39;]
Output Example:
“Alice contributed to backend API development, testing automation, and database optimization
across multiple projects.”
Strengths: Produces coherent summaries; handles moderately long input text.
Limitations: Generic outputs for sparse data; sensitive to verbosity of task descriptions.
3. Text Generation
Model: Qwen/Qwen2-1.5B-Instruct
Purpose: Generate synthetic work log entries for:
Data augmentation
Simulation of team productivity scenarios
Generative “what-if” analysis

### 3.2 Model Architecture Comparison: Encoder-Decoder vs. Decoder-Only

This project intentionally utilizes both an **encoder-decoder** model (`sshleifer/distilbart-cnn-12-6` for summarization) and a **decoder-only** model (`Qwen/Qwen2-1.5B-Instruct` for text generation) to align with the project requirements and leverage the distinct strengths of each architecture.

**Encoder-Decoder (BART/DistilBART):**

*   **Architecture:** Consists of two main parts: an encoder that processes the entire input text to create a rich contextual representation, and a decoder that generates the output sequence based on that representation.
*   **Strengths:** This architecture excels at tasks where the output is heavily conditioned on the full input, such as summarization or translation. The encoder ensures that the generated summary is factually grounded in the source text, reducing the risk of hallucination.
*   **Use Case in Project:** We chose DistilBART for summarization because it is crucial to produce concise summaries that accurately reflect the content of the original work logs. Its encoder-decoder structure is ideal for this "many-to-few" text transformation.
*   **Trade-offs:** While robust for summarization, these models are less flexible for open-ended generative or instruction-following tasks.

**Decoder-Only (Qwen2):**

*   **Architecture:** Consists of a single transformer block that predicts the next token in a sequence based on the preceding tokens. It is autoregressive by nature.
*   **Strengths:** This architecture is highly effective for tasks that require fluid, creative, or instruction-based text generation. Models like Qwen2 are trained to follow prompts and can generate diverse outputs, making them suitable for creative and conversational AI.
*   **Use Case in Project:** We selected Qwen2 for generating synthetic task logs. This task requires generating new, realistic-sounding text based on a simple prompt (employee name, task category). The decoder-only structure is perfect for this kind of open-ended generation.
*   **Trade-offs:** Without the strong grounding of an encoder, decoder-only models can be more prone to hallucinating facts if not carefully prompted. They are generally better suited for "few-to-many" text transformations (e.g., a short prompt resulting in a longer text).

**Performance and Trade-off Analysis:**

The selection of these two model types represents a key design decision. For **summarization**, the factual consistency offered by the encoder-decoder architecture of BART was prioritized. For **synthetic data generation**, the flexibility and instruction-following capability of the decoder-only Qwen2 model were more valuable. This division of labor allows the system to use the best architectural tool for each specific NLP task, balancing the need for factual accuracy in reporting with the need for creative generation in data augmentation.

### 3.3 Code Structure

Component Description
load_ner_model() Loads NER pipeline once and caches it with

 @st.cache_resource.

rule_based_ner_analysis() Entity recognition that uses regex patterns to identify
people and organizations. What code defaults to if NER
pipeline fails.

smart_ner_analysis() Entity recognition that uses dslim/bert-base-NER to identify

people, organizations, and skills.

calculate_employee_scores() Computes performance scores (completion, diversity,

workload).

summarize_text() Summarizes contributions per employee using

sshleifer/distilbart-cnn-12-6 model.

generate_synthetic_task_log() Creates a new description for an employee using
prompting on the Qwen/Qwen2-1.5B-Instruct model. If
model fails, defaults to template-based text generation.

generate_data_driven_insight
()

Creates insights for the Data-Driven Insights tab using
template approach. Four different template options; data is
fed dependent on the search type of the user.

generate_project_insight() Using generate_data_driven_insight, creates an insight on

the project.

generate_team_insight() Using generate_data_driven_insight, creates an insight on

the team.

generate_performance_insig
ht()

Using generate_data_driven_insight, creates an insight on
the employee’s performances / skills.

generate_skill_gap_insight() Using generate_data_driven_insight, creates an insight on
the skill gaps of all employees together as a team.
Streamlit Tabs UI components: Overview, Leaderboard, Skills, Reports,

Team Optimizer, NER Analysis, AI Insights.

### 3.4 Training Details

All models used in this project are pre-trained models from Hugging Face. **No fine-tuning or additional training was performed.** The models are used out-of-the-box to demonstrate their zero-shot capabilities on the given tasks. This approach ensures the system is lightweight and does not require extensive computational resources for training, making it easily reproducible.

4. Evaluation and Metrics
Qualitative Evaluation
The current app focuses on interactive qualitative analysis through:
Human inspection of NER outputs (skills detected).
Readability and coherence of summaries.
Manager insight via dashboards and leaderboards.
Quantitative Metrics
Task Metric Description
NER Precision / Recall / F1 Evaluate entity correctness and coverage.
Summarization ROUGE-1, ROUGE-2,

ROUGE-L

Compare generated summaries to gold-
standard texts.

Text Generation BLEU, Perplexity Assess fluency and content diversity.

Formal testing set creation for these metrics:
5. Error Analysis and Discussion
Observed Issues
NER Misclassification: Occasional detection of irrelevant entities (e.g., organization names).
Skill Coverage Bias: Some rare technical skills not recognized by pre-trained NER.
Summarization Redundancy: Repetitive phrasing when logs contain similar descriptions.
Data Sparsity: Employees with few tasks get weaker summaries and unreliable scores.
Mitigation Strategies
Keyword post-filtering for NER to improve precision.
Using longer context aggregation for BART to reduce redundancy.
Weight normalization in performance scoring to ensure fairness.
6. Future Work and Improvements
Model Expansion:
Integrate a decoder-only model (e.g., Mistral-7B-Instruct-Summarize-64k or SummLlama 3.2)
for longer and more human-like summaries.
Text Generation Module:
Implement generative models to create synthetic task logs and simulate team load scenarios.
Model Fine-Tuning:
Fine-tune NER and summarization models on a small, domain-specific dataset of annotated
work logs to improve contextual accuracy.
Quantitative Evaluation Pipeline:
Develop a test suite to calculate ROUGE, Precision, and F1 automatically.
Knowledge Graph Integration:
Build a graph of employee–skill relationships for visual exploration.
7. Demo App and User Interface

Platform:
Developed using Streamlit, with Plotly for visualization and Pandas/Numpy for data processing.
Tabs Overview
Tab Description
Overview Displays overall statistics and skill distribution bar chart.
Leaderboard Ranks employees by computed performance score.
Skills Shows per-employee skill matrix with visual skill tags.
Reports Generates individual downloadable text reports.
Team Optimizer Analyzes project team composition, skill gaps, and staffing balance.
NER Analysis Runs raw NER pipeline on a selected task to visualize entities.
AI Insights User can select for “project analysis”, “team optimization”,
“employee performance”, and “skill gap analysis” and receive the
associated generated analysis report, or select “synthetic data
generator” and get a set of synthetic report logs for testing.

Design Highlights
Custom CSS themes with gradient cards and skill tags.
Dynamic metrics cards (custom_metric() function).
AI processing spinner for UX feedback.
Downloadable employee reports in .txt format.
8. Conclusion
The Workforce Intelligence Platform successfully demonstrates how applied NLP
techniques—specifically NER, Summarization, and Generation—can extract structured insights
from unstructured employee work logs.
With these components, the system provides:
Automated skill detection and employee profiling,
Summarized performance insights for each worker,
AI-driven dashboards for workforce optimization.

9. References
Hugging Face Transformers Library: https://huggingface.co/transformers/
Model: dslim/bert-base-NER
Model: sshleifer/distilbart-cnn-12-6
Streamlit Documentation: https://docs.streamlit.io/
Plotly Express Documentation: https://plotly.com/python/plotly-express/

Status Summary

Component Status
NER Model Integration Completed
Summarization Model Integration Completed
Text Generation Model Completed
Backend Integration Completed