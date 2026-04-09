import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.express as px
import re
import time

# -----------------------------
# 🎯 APP CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Workforce Intelligence Platform", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .skill-tag {
        background: #00b894;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        margin: 4px;
        display: inline-block;
    }
    .person-tag {
        background: #6c5ce7;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        margin: 4px;
        display: inline-block;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ai-insight {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #fd79a8;
    }
    .insight-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #74b9ff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏢 Workforce Intelligence Platform</div>', unsafe_allow_html=True)
st.write("**Enhanced Version** - Data-Driven AI Insights")

# -----------------------------
# 🛠️ IMPROVED SKILL DETECTION WITH 
# -----------------------------

# Comprehensive skill list
TECH_SKILLS = {
    'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin', 'r', 'php', 'ruby'],
    'web': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel', 'node.js', 'html', 'css', 'bootstrap', 'sass', 'less'],
    'mobile': ['android', 'ios', 'react native', 'flutter', 'mobile development'],
    'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'database', 'nosql', 'dynamodb', 'cassandra'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd', 'devops', 'cloud', 'serverless', 'lambda'],
    'data_science': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'machine learning', 'ml', 'ai', 'data analysis', 'etl', 'tableau', 'powerbi', 'big data', 'analytics'],
    'tools': ['git', 'jira', 'confluence', 'figma', 'photoshop', 'slack', 'teams', 'vscode', 'intellij', 'eclipse', 'postman'],
    'methods': ['agile', 'scrum', 'kanban', 'waterfall', 'tdd', 'test driven development'],
    'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'creativity', 'collaboration', 'presentation', 'mentoring'],
    'testing': ['testing', 'unit tests', 'integration tests', 'test cases', 'qa', 'quality assurance']
}

# Flatten all skills for easy matching
ALL_SKILLS = [skill for sublist in TECH_SKILLS.values() for skill in sublist]

def extract_skills_simple(text):
    """
    SIMPLE BUT EFFECTIVE skill extraction using keyword matching
    """
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    found_skills = []
    
    # Look for each skill in the text
    for skill in ALL_SKILLS:
        # Use word boundaries to avoid partial matches
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
    
    # Special patterns for common skills
    patterns = {
        'api': r'\bapi\b',
        'ui/ux': r'\b(ui|ux|user interface|user experience)\b',
        'frontend': r'\bfront.end\b',
        'backend': r'\back.end\b',
        'fullstack': r'\bfull.stack\b',
        'data pipeline': r'\bdata pipeline\b',
        'endpoint': r'\bendpoint\b',
        'analytics': r'\banalytics\b',
        'onboarding': r'\bonboarding\b',
        'authentication': r'\bauthentication\b',
        'browser': r'\bbrowser\b',
        'security': r'\bsecurity\b',
        'testing': r'\b(testing|tests|test cases)\b'
    }
    
    for skill_name, pattern in patterns.items():
        if re.search(pattern, text_lower, re.IGNORECASE) and skill_name not in found_skills:
            found_skills.append(skill_name)
    
    return list(set(found_skills))

def rule_based_ner_analysis(text):
    """
    Rule-based NER for extracting PERSON and ORG entities using regex patterns.
    Falls back to this when transformer model is unavailable.
    """
    if not isinstance(text, str):
        return []
    
    entities = []
    text_lower = text.lower()
    
    # Common patterns for person names (capitalized words, typically 2-3 words)
    # Look for patterns like "John Smith", "Alice Johnson", etc.
    person_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
    person_matches = re.finditer(person_pattern, text)
    
    for match in person_matches:
        person_name = match.group(1)
        # Filter out common false positives (days, months, project names that look like names)
        false_positives = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                          'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                          'september', 'october', 'november', 'december', 'project', 'task', 'ticket']
        if person_name.lower() not in false_positives and len(person_name.split()) <= 3:
            entities.append({'entity': 'PERSON', 'word': person_name, 'score': 0.75})
    
    # Common patterns for organizations (company names, often ending with Inc, Corp, LLC, etc.)
    org_patterns = [
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Systems|Technologies|Solutions))?)\b',
        r'\b(Google|Microsoft|Amazon|Apple|Facebook|Meta|Twitter|LinkedIn|IBM|Oracle|Salesforce|Adobe|Intel|NVIDIA)\b'
    ]
    
    for pattern in org_patterns:
        org_matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in org_matches:
            org_name = match.group(1)
            # Avoid duplicates and common false positives
            if not any(e['word'].lower() == org_name.lower() for e in entities):
                # Filter out single common words that might match
                if len(org_name.split()) > 1 or org_name.lower() in ['google', 'microsoft', 'amazon', 'apple', 'facebook', 'meta', 'twitter', 'linkedin', 'ibm', 'oracle', 'salesforce', 'adobe', 'intel', 'nvidia']:
                    entities.append({'entity': 'ORG', 'word': org_name, 'score': 0.70})
    
    return entities

def smart_ner_analysis(text):
    """
    Enhanced NER that attempts to use a Hugging Face transformer model first,
    falls back to rule-based if not available, and integrates skill extraction.
    """
    ner_pipeline = load_ner_model()
    entities = []

    if ner_pipeline:
        try:
            hf_entities = ner_pipeline(text)
            for ent in hf_entities:
                entity_type = ent['entity_group']
                word = ent['word']
                score = ent['score']

                # Map HF entity types to application's types
                if entity_type == 'PER':
                    entities.append({'entity': 'PERSON', 'word': word, 'score': score})
                elif entity_type == 'ORG':
                    entities.append({'entity': 'ORG', 'word': word, 'score': score})
                # For other entity types, we might ignore them or map them if needed
            
            # If HF model found some entities, combine with skills
            if entities:
                # Detect skills in context using the simple extractor
                skills_in_text = extract_skills_simple(text)
                for skill in skills_in_text:
                    # Avoid adding skills that are already identified as PERSON/ORG by HF model
                    if not any(e['word'].lower() == skill.lower() for e in entities if e['entity'] in ['PERSON', 'ORG']):
                        entities.append({'entity': 'SKILL', 'word': skill, 'score': 0.90})
                return entities

        except Exception as e:
            st.warning(f"Hugging Face NER pipeline failed: {e}. Falling back to rule-based NER.")
            # Fallback to rule-based NER if HF pipeline fails
            pass # Continue to rule_based_ner_analysis

    # Fallback to rule-based NER if HF model not loaded or no entities found by HF model
    # or if HF pipeline failed
    st.info("Using rule-based NER for analysis.")
    # Detect skills in context using the simple extractor
    skills_in_text = extract_skills_simple(text)
    for skill in skills_in_text:
        entities.append({'entity': 'SKILL', 'word': skill, 'score': 0.90})
    
    # Add rule-based PERSON/ORG detection
    rule_based_entities = rule_based_ner_analysis(text)
    for ent in rule_based_entities:
        # Only add if not already covered by skills or HF entities (if any were found)
        if not any(e['word'].lower() == ent['word'].lower() for e in entities):
            entities.append(ent)

    return entities

@st.cache_resource
def load_ner_model():
    """Load the Hugging Face NER model"""
    try:
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    except Exception as e:
        st.error(f"Failed to load NER model: {e}. Falling back to rule-based NER.")
        return None

@st.cache_resource
def load_summarizer():
    """Load a fast summarization model (encoder-decoder)"""
    try:
        return pipeline("summarization", 
                       model="sshleifer/distilbart-cnn-12-6",
                       tokenizer="sshleifer/distilbart-cnn-12-6")
    except:
        return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    """Simple summarization using encoder-decoder model"""
    try:
        summarizer = load_summarizer()
        
        # Clean and prepare text
        if len(text) > 1500:
            text = text[:1500] + "..."
        
        summary = summarizer(text, 
                           max_length=100, 
                           min_length=30, 
                           do_sample=False,
                           truncation=True)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Summary: {text[:200]}..."  # Fallback to truncated text

@st.cache_resource
def load_text_generator():
    """Load the text generation model for synthetic task log generation"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        st.warning(f"Failed to load text generation model: {e}. Using template-based generation.")
        return None

def generate_synthetic_task_log(employee_name, task_category, project_name, generator=None):
    """
    Generate a synthetic task log description using transformer model.
    Falls back to template-based generation if model is unavailable.
    """
    if generator is None:
        generator = load_text_generator()
    
    if generator is None:
        # Fallback to template-based generation
        templates = [
            f"{employee_name} worked on {task_category} tasks for {project_name}.",
            f"{employee_name} completed {task_category} work related to {project_name}.",
            f"{employee_name} handled {task_category} issues in {project_name}."
        ]
        return np.random.choice(templates)
    
    try:
        model = generator["model"]
        tokenizer = generator["tokenizer"]
        
        # Create a prompt for task log generation
        prompt = f"""Generate a realistic work task description for an employee named {employee_name} working on a {task_category} task in {project_name}. 
The description should be professional, specific, and similar to real work logs. Keep it concise (1-2 sentences)."""
        
        # Format for Qwen2-Instruct
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates realistic work task descriptions."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
        
    except Exception as e:
        st.warning(f"Text generation failed: {e}. Using template-based generation.")
        # Fallback
        templates = [
            f"{employee_name} worked on {task_category} tasks for {project_name}.",
            f"{employee_name} completed {task_category} work related to {project_name}.",
            f"{employee_name} handled {task_category} issues in {project_name}."
        ]
        return np.random.choice(templates)

def generate_data_driven_insight(insight_type, data_context):
    """
    Generate data-driven insights without hallucinations
    Uses template-based approach with real data
    """
    
    insights_templates = {
        'project_description': [
            "Project {project_name} involves {team_size} team members with key skills in {top_skills}. The team has completed {completed_tasks} tasks with a {completion_rate}% success rate. Key focus areas include {key_areas}.",
            "{project_name} is managed by a team of {team_size} professionals skilled in {top_skills}. With {completed_tasks} completed tasks and {active_tasks} in progress, the project maintains a {completion_rate}% completion rate.",
            "The {project_name} initiative leverages expertise in {top_skills} across {team_size} team members. Current progress shows {completed_tasks} deliverables completed with ongoing work in {key_areas}."
        ],
        
        'team_recommendation': [
            "Based on skill distribution, consider cross-training in {missing_skills}. The team excels in {strong_skills} but could benefit from additional expertise in {gap_areas}.",
            "Team optimization suggestions: Focus on developing {skill_gaps} capabilities. Current strengths in {top_skills} provide a solid foundation for expanding into {growth_areas}.",
            "Recommendation: Balance skill distribution by training team members in {needed_skills}. The current expertise in {existing_skills} should be complemented with {complementary_skills}."
        ],
        
        'performance_analysis': [
            "{employee_name} demonstrates strong performance with a score of {score}/100 and {completion_rate}% task completion. Key skills include {skills} with notable strengths in {top_skills}.",
            "Performance review: {employee_name} maintains a {score}/100 rating with expertise in {skills}. Areas for growth include developing {growth_areas} while leveraging existing {strengths}.",
            "{employee_name} shows consistent performance ({score}/100) with specialization in {skills}. Completion rate of {completion_rate}% indicates reliable delivery on assigned tasks."
        ],
        
        'skill_gap_analysis': [
            "Organization shows strong capabilities in {top_skills} but has gaps in {missing_skills}. Recommended focus areas: {recommended_skills} to enhance overall team capabilities.",
            "Skill distribution analysis: Expertise concentrated in {common_skills}. Strategic gaps identified in {gap_skills}. Consider prioritizing {priority_skills} for development.",
            "Current skill landscape features {existing_skills} as core competencies. Development opportunities exist in {development_areas} to create more balanced team capabilities."
        ]
    }
    
    # Select appropriate template
    templates = insights_templates.get(insight_type, [])
    if not templates:
        return "Insight type not supported."
    
    template = np.random.choice(templates)
    
    # Fill template with actual data
    try:
        filled_insight = template.format(**data_context)
        return filled_insight
    except KeyError as e:
        return f"Data-driven insight: Analysis based on available metrics shows opportunities for optimization and improvement."

def generate_project_insight(project_name, df, employee_skills):
    """Generate project-specific insights"""
    project_data = df[df['Project'] == project_name]
    team_members = project_data['Employee'].unique()
    team_size = len(team_members)
    
    # Calculate project metrics
    completed_tasks = (project_data['Status'] == 'Done').sum()
    total_tasks = len(project_data)
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Get project skills
    project_skills = []
    for member in team_members:
        if member in employee_skills:
            project_skills.extend(employee_skills[member])
    top_skills = list(set(project_skills))[:3]
    
    # Identify key areas from task categories
    key_areas = project_data['TaskCategory'].value_counts().head(3).index.tolist()
    
    context = {
        'project_name': project_name,
        'team_size': team_size,
        'top_skills': ', '.join(top_skills) if top_skills else 'various technical skills',
        'completed_tasks': completed_tasks,
        'active_tasks': total_tasks - completed_tasks,
        'completion_rate': round(completion_rate, 1),
        'key_areas': ', '.join(key_areas) if key_areas else 'multiple domains'
    }
    
    return generate_data_driven_insight('project_description', context)

def generate_team_insight(df, employee_skills, all_detected_skills):
    """Generate team optimization insights"""
    # Analyze skill distribution
    skill_counts = pd.Series(all_detected_skills).value_counts()
    top_skills = skill_counts.head(5).index.tolist()
    
    # Find missing skills
    common_skills = ['python', 'javascript', 'sql', 'aws', 'react']  # Expected common skills
    missing_skills = [skill for skill in common_skills if skill not in all_detected_skills]
    
    # Identify skill gaps
    strong_skills = top_skills[:3] if len(top_skills) >= 3 else top_skills
    gap_areas = missing_skills[:3] if missing_skills else ['advanced cloud infrastructure', 'specialized data analysis', 'automated testing']
    
    context = {
        'missing_skills': ', '.join(missing_skills) if missing_skills else 'several emerging technologies',
        'strong_skills': ', '.join(strong_skills) if strong_skills else 'core technical capabilities',
        'gap_areas': ', '.join(gap_areas),
        'top_skills': ', '.join(top_skills[:3]) if top_skills else 'diverse technical skills',
        'skill_gaps': ', '.join(missing_skills[:2]) if missing_skills else 'specialized technical domains',
        'existing_skills': ', '.join(top_skills[:3]) if top_skills else 'current technical expertise',
        'complementary_skills': 'complementary technical specializations',
        'needed_skills': ', '.join(missing_skills[:2]) if missing_skills else 'specialized technical skills',
        'common_skills': ', '.join(top_skills[:3]) if top_skills else 'established technical capabilities',
        'growth_areas': ', '.join(gap_areas)
    }
    
    return generate_data_driven_insight('team_recommendation', context)

def generate_performance_insight(employee_name, skills, score, completion_rate):
    """Generate performance insights"""
    top_skills = skills[:3] if skills else ['various technical capabilities']
    growth_areas = ['advanced technical specializations', 'cross-functional collaboration', 'leadership development']
    
    context = {
        'employee_name': employee_name,
        'score': score,
        'completion_rate': completion_rate,
        'skills': ', '.join(skills) if skills else 'diverse capabilities',
        'top_skills': ', '.join(top_skills),
        'growth_areas': ', '.join(growth_areas[:2]),
        'strengths': ', '.join(top_skills) if top_skills else 'technical proficiency'
    }
    
    return generate_data_driven_insight('performance_analysis', context)

def generate_skill_gap_insight(all_detected_skills):
    """Generate skill gap analysis"""
    skill_counts = pd.Series(all_detected_skills).value_counts()
    top_skills = skill_counts.head(5).index.tolist()
    
    # Common skills that might be missing
    expected_skills = ['python', 'javascript', 'sql', 'aws', 'docker', 'react', 'node.js']
    missing_skills = [skill for skill in expected_skills if skill not in all_detected_skills]
    
    context = {
        'top_skills': ', '.join(top_skills[:3]) if top_skills else 'core technical skills',
        'missing_skills': ', '.join(missing_skills) if missing_skills else 'specialized technical domains',
        'recommended_skills': ', '.join(missing_skills[:3]) if missing_skills else 'emerging technology areas',
        'common_skills': ', '.join(top_skills[:3]) if top_skills else 'established capabilities',
        'gap_skills': ', '.join(missing_skills[:2]) if missing_skills else 'strategic technical areas',
        'priority_skills': ', '.join(missing_skills[:2]) if missing_skills else 'high-demand technical skills',
        'existing_skills': ', '.join(top_skills[:3]) if top_skills else 'current technical expertise',
        'development_areas': ', '.join(missing_skills[:3]) if missing_skills else 'expanded technical capabilities'
    }
    
    return generate_data_driven_insight('skill_gap_analysis', context)

def calculate_employee_scores(df, employee_data):
    """Calculate performance scores for employees"""
    score = 0
    # Completion rate (40% weight)
    completion_rate = (employee_data['Status'] == 'Done').mean()
    score += completion_rate * 40
    
    # Task diversity (20% weight)
    task_diversity = employee_data['TaskCategory'].nunique() / df['TaskCategory'].nunique()
    score += min(task_diversity * 20, 20)
    
    # Project diversity (20% weight)
    project_diversity = employee_data['Project'].nunique() / df['Project'].nunique()
    score += min(project_diversity * 20, 20)
    
    # Work volume (20% weight)
    volume_score = min(len(employee_data) / (len(df) / df['Employee'].nunique()) * 20, 20)
    score += volume_score
    
    return round(score, 1)

def custom_metric(title, value):
    st.markdown(f"""
        <div class="metric-card">
            <p style="font-weight: bold; font-size: 1.1em; margin: 0; opacity: 0.9;">{title}</p>
            <p style="font-size: 2.2em; font-weight: 700; margin: 5px 0 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">{value}</p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 📂 LOAD DATA
# -----------------------------
st.sidebar.header("📁 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Basic data validation
    st.success(f"✅ Data loaded: {len(df)} records, {df['Employee'].nunique()} employees")
    
    # PROCESS ALL DATA FIRST - BEFORE CREATING TABS
    with st.spinner("🔄 Processing data and generating insights..."):
        # Create leaderboard data
        leaderboard_data = []
        for employee in df['Employee'].unique():
            emp_data = df[df['Employee'] == employee]
            score = calculate_employee_scores(df, emp_data)
            
            leaderboard_data.append({
                'Employee': employee,
                'Score': score,
                'Tasks': len(emp_data),
                'Projects': emp_data['Project'].nunique(),
                'Completion Rate': f"{(emp_data['Status'] == 'Done').mean() * 100:.1f}%",
                'Task Categories': emp_data['TaskCategory'].nunique()
            })
        
        leaderboard_df = pd.DataFrame(leaderboard_data).sort_values('Score', ascending=False)
        
        # Process skills across organization
        all_detected_skills = []
        employee_skills = {}
        
        for employee in df['Employee'].unique():
            emp_data = df[df['Employee'] == employee]
            emp_skills = []
            
            for desc in emp_data['Description'].dropna():
                skills = extract_skills_simple(desc)
                emp_skills.extend(skills)
            
            unique_skills = list(set(emp_skills))
            employee_skills[employee] = unique_skills
            all_detected_skills.extend(unique_skills)
        
        # Analyze project teams for Team Optimizer
        project_teams = {}
        for project in df['Project'].unique():
            project_data = df[df['Project'] == project]
            team_members = project_data['Employee'].unique()
            
            # Analyze team composition
            team_skills = []
            for member in team_members:
                if member in employee_skills:
                    team_skills.extend(employee_skills[member])
            
            unique_skills = list(set(team_skills))
            
            project_teams[project] = {
                'team_size': len(team_members),
                'team_members': list(team_members),
                'skills_covered': unique_skills,
                'skill_gaps': [skill for skill in (set(all_detected_skills) if all_detected_skills else []) if skill not in unique_skills][:5]
            }
        
        # NOW CREATE ALL TABS 
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Overview", "🏆 Leaderboard", "🔧 Skills", "📋 Reports", "👥 Team Optimizer", "🤖 NER Analysis", "🧠 AI Insights"])
        
        with tab1:
            st.header("📊 Organizational Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                custom_metric("Total Employees", df['Employee'].nunique())
            with col2:
                custom_metric("Total Tasks", len(df))
            with col3:
                custom_metric("Active Projects", df['Project'].nunique())
            with col4:
                completion = (df['Status'] == 'Done').mean() * 100
                custom_metric("Overall Completion", f"{completion:.1f}%")
            
            # Display skill distribution
            st.subheader("🎯 Skill Distribution Across Organization")
            
            if all_detected_skills:
                skill_counts = pd.Series(all_detected_skills).value_counts()
                
                # Filter out numeric values and nonsense
                valid_skills = skill_counts[skill_counts.index.astype(str).str.isalpha()]
                
                if len(valid_skills) > 0:
                    fig = px.bar(valid_skills.head(10), 
                                title="Top 10 Skills Detected",
                                labels={'value': 'Frequency', 'index': 'Skills'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show skills as tags
                    st.write("**All Detected Skills:**")
                    for skill in valid_skills.index[:20]:
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.info("🤖 No valid skills detected. Try the NER Analysis tab for detailed text analysis.")
            else:
                st.info("📝 No skills detected yet. The system will analyze task descriptions to find skills.")
        
        with tab2:
            st.header("🏆 Employee Leaderboard")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Performance Ranking")
                styled_df = leaderboard_df.reset_index(drop=True).style.background_gradient(
                    subset=['Score'], cmap='YlOrBr'
                )
                st.dataframe(styled_df, use_container_width=True)
            
            with col2:
                st.subheader("🎖️ Top Performers")
                top_3 = leaderboard_df.head(3)
                for i, (_, employee) in enumerate(top_3.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div style="padding: 15px; background: linear-gradient(135deg, #ffeaa7, #fab1a0); 
                                border-radius: 10px; margin: 10px 0; text-align: center;">
                        <h4 style="margin: 0; color: #2d3436;">{medal} {employee['Employee']}</h4>
                        <h3 style="margin: 5px 0; color: #e17055;">{employee['Score']} pts</h3>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.header("🔧 Skills Dashboard")
            
            # Employee skill matrix
            st.subheader("Employee Skill Matrix")
            
            skill_data = []
            for employee, skills in employee_skills.items():
                skill_data.append({
                    'Employee': employee,
                    'Skill Count': len(skills),
                    'Skills': ', '.join(skills) if skills else 'None detected'
                })
            
            skill_df = pd.DataFrame(skill_data).sort_values('Skill Count', ascending=False)
            st.dataframe(skill_df, use_container_width=True)
            
            # Skill categories
            st.subheader("📚 Skills by Category")
            for category, skills in TECH_SKILLS.items():
                with st.expander(f"{category.title()} ({len(skills)} skills)"):
                    col1, col2, col3 = st.columns(3)
                    skills_per_col = len(skills) // 3 + 1
                    
                    for i, skill in enumerate(skills):
                        col = i // skills_per_col
                        if col == 0:
                            col1.write(f"• {skill}")
                        elif col == 1:
                            col2.write(f"• {skill}")
                        else:
                            col3.write(f"• {skill}")
        
        with tab4:
            st.header("📋 Employee Reports")
            
            selected_employee = st.selectbox("Select Employee", df['Employee'].unique(), key="reports_select")
            
            if st.button("Generate Report", type="primary", key="reports_btn"):
                emp_data = df[df['Employee'] == selected_employee]
                skills = employee_skills.get(selected_employee, [])
                
                st.subheader(f"Report for {selected_employee}")
                
                # Summary section
                st.write("### 📝 AI Summary")
                all_descriptions = " ".join(emp_data['Description'].dropna().astype(str))
                if all_descriptions.strip():
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(all_descriptions)
                    st.info(summary)
                else:
                    st.warning("No descriptions available for summary")
                
                # AI Performance Insight
                if skills:
                    st.write("### 🧠 AI Performance Insight")
                    completion_rate = (emp_data['Status'] == 'Done').mean() * 100
                    score = calculate_employee_scores(df, emp_data)
                    insight = generate_performance_insight(selected_employee, skills, score, f"{completion_rate:.1f}")
                    st.markdown(f'<div class="ai-insight">{insight}</div>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    score = calculate_employee_scores(df, emp_data)
                    custom_metric("Performance Score", f"{score:.1f}")
                with col2:
                    done_tasks = (emp_data['Status'] == 'Done').sum()
                    custom_metric("Tasks Completed", f"{done_tasks}/{len(emp_data)}")
                with col3:
                    custom_metric("Skills Detected", len(skills))
                with col4:
                    custom_metric("Projects", emp_data['Project'].nunique())
                
                # Skills section
                st.write("### 🛠️ Detected Skills")
                if skills:
                    for skill in skills:
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.info("No skills detected for this employee")
                
                # Recent tasks
                st.write("### 📋 Recent Tasks")
                st.dataframe(emp_data[['TicketID', 'Project', 'TaskCategory', 'Description', 'Status']].head(10))
        
        with tab5:
            st.header("👥 Team Optimizer")
            
            st.subheader("📋 Project Team Analysis")
            
            # Display project teams
            for project, data in project_teams.items():
                with st.expander(f"🚀 {project} (Team: {data['team_size']} members)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**👥 Team Members:**")
                        for member in data['team_members']:
                            st.write(f"• {member}")
                        
                        st.write("**✅ Skills Covered:**")
                        for skill in data['skills_covered'][:8]:
                            st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                    
                    with col2:
                        st.write("**💡 Recommendations:**")
                        if len(data['skill_gaps']) > 0:
                            st.write("**Consider adding:**")
                            for gap in data['skill_gaps'][:3]:
                                st.markdown(f"<div style='color: #e17055; font-weight: bold;'>⚡ {gap}</div>", unsafe_allow_html=True)
                        else:
                            st.success("✅ Well-rounded team composition!")
                        
                        # Team balance score
                        avg_tasks = len(df[df['Project'] == project]) / data['team_size']
                        if avg_tasks > 10:
                            st.warning("📊 Team may be understaffed")
                        elif avg_tasks < 3:
                            st.info("📊 Team may be overstaffed")
                        else:
                            st.success("📊 Good team size for workload")
            
            # Team optimization suggestions
            st.subheader("🔧 Optimization Suggestions")
            
            # Find employees with unique skills
            unique_skill_holders = {}
            if all_detected_skills:
                for skill in set(all_detected_skills):
                    holders = [emp for emp, skills in employee_skills.items() if skill in skills]
                    if len(holders) == 1:  # Only one person has this skill
                        unique_skill_holders[skill] = holders[0]
            
            if unique_skill_holders:
                st.error("🚨 Critical Skills Dependency:")
                for skill, holder in list(unique_skill_holders.items())[:3]:
                    st.write(f"• **{skill}** only known by **{holder}**")
            
            # Cross-training opportunities
            st.info("💡 Cross-Training Opportunities:")
            busy_employees = leaderboard_df.head(3)['Employee'].tolist()
            for emp in busy_employees[:2]:
                st.write(f"• Consider training backup for **{emp}**'s responsibilities")
        
        with tab6:
            st.header("🤖 Enhanced NER Analysis")
            
            st.write("**Analyze specific task descriptions for skills and entities**")
            
            # Sample tasks for analysis
            sample_tasks = df['Description'].dropna().head(10).tolist()
            selected_task = st.selectbox("Select task to analyze:", sample_tasks, key="ner_select")
            
            if st.button("Analyze Text", type="primary", key="ner_btn"):
                st.subheader("🔍 Analysis Results")
                
                # Skill detection
                skills_found = extract_skills_simple(selected_task)
                st.write("### 🎯 Skills & Technologies Detected")
                if skills_found:
                    for skill in skills_found:
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.info("No specific skills detected in this text")
                
                # Enhanced NER analysis
                st.write("### 📋 Entity Analysis")
                entities = smart_ner_analysis(selected_task)
                
                if entities:
                    # Convert to dataframe for nice display
                    entities_df = pd.DataFrame(entities)
                    st.dataframe(entities_df, use_container_width=True)
                    
                    # Show statistics
                    st.write("### 📊 Analysis Summary")
                    persons = [e for e in entities if e['entity'] == 'PERSON']
                    skills_ner = [e for e in entities if e['entity'] == 'SKILL']
                    orgs = [e for e in entities if e['entity'] == 'ORG']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Persons", len(persons))
                    col2.metric("Skills", len(skills_ner))
                    col3.metric("Organizations", len(orgs))
                    
                    # Show detected persons separately
                    if persons:
                        st.write("### 👥 Persons Detected")
                        for person in persons:
                            st.markdown(f'<span class="person-tag">{person["word"]}</span>', unsafe_allow_html=True)
                    
                else:
                    st.info("No entities detected in this text")
                
                # Show raw text
                st.write("### 📝 Original Text")
                st.text_area("Task Description", selected_task, height=100, key="ner_text")
        
        with tab7:
            st.header("🧠 Data-Driven AI Insights")
            
            st.write("**Generate reliable, data-driven insights based on your actual workforce data**")
            
            # Insight generation options
            insight_type = st.selectbox(
                "Choose insight type:",
                ["Project Analysis", "Team Optimization", "Employee Performance", "Skill Gap Analysis", "Synthetic Data Generator"],
                key="insights_select"
            )
            
            if insight_type == "Project Analysis":
                selected_project = st.selectbox("Select Project", df['Project'].unique(), key="project_select")
                if st.button("Generate Project Analysis", key="project_btn"):
                    with st.spinner("🔍 Analyzing project data..."):
                        insight = generate_project_insight(selected_project, df, employee_skills)
                        st.markdown(f'<div class="ai-insight">{insight}</div>', unsafe_allow_html=True)
            
            elif insight_type == "Team Optimization":
                if st.button("Generate Team Recommendations", key="team_btn"):
                    with st.spinner("🔍 Analyzing team composition..."):
                        insight = generate_team_insight(df, employee_skills, all_detected_skills)
                        st.markdown(f'<div class="ai-insight">{insight}</div>', unsafe_allow_html=True)
            
            elif insight_type == "Employee Performance":
                selected_employee = st.selectbox("Select Employee for Analysis", df['Employee'].unique(), key="perf_select")
                if st.button("Generate Performance Insight", key="perf_btn"):
                    with st.spinner("🔍 Analyzing performance data..."):
                        emp_data = df[df['Employee'] == selected_employee]
                        skills = employee_skills.get(selected_employee, [])
                        score = calculate_employee_scores(df, emp_data)
                        completion_rate = (emp_data['Status'] == 'Done').mean() * 100
                        
                        insight = generate_performance_insight(selected_employee, skills, score, f"{completion_rate:.1f}")
                        st.markdown(f'<div class="ai-insight">{insight}</div>', unsafe_allow_html=True)
            
            elif insight_type == "Skill Gap Analysis":
                if st.button("Generate Skill Gap Analysis", key="gap_btn"):
                    with st.spinner("🔍 Analyzing skill distribution..."):
                        insight = generate_skill_gap_insight(all_detected_skills)
                        st.markdown(f'<div class="ai-insight">{insight}</div>', unsafe_allow_html=True)
            
            elif insight_type == "Synthetic Data Generator":
                st.subheader("🤖 Generate Synthetic Task Logs")
                st.write("**Use AI to generate realistic task descriptions for testing and data augmentation**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    gen_employee = st.selectbox("Employee Name", df['Employee'].unique(), key="gen_emp")
                with col2:
                    gen_category = st.selectbox("Task Category", df['TaskCategory'].unique(), key="gen_cat")
                with col3:
                    gen_project = st.selectbox("Project", df['Project'].unique(), key="gen_proj")
                
                num_logs = st.number_input("Number of logs to generate", min_value=1, max_value=10, value=3, key="num_logs")
                
                if st.button("Generate Task Logs", type="primary", key="gen_btn"):
                    with st.spinner("🤖 Generating synthetic task logs using AI..."):
                        generator = load_text_generator()
                        generated_logs = []
                        
                        for i in range(num_logs):
                            log = generate_synthetic_task_log(gen_employee, gen_category, gen_project, generator)
                            generated_logs.append({
                                'Log #': i + 1,
                                'Description': log
                            })
                        
                        st.success(f"✅ Generated {num_logs} synthetic task logs!")
                        st.subheader("📋 Generated Task Logs")
                        
                        for i, log_data in enumerate(generated_logs):
                            st.markdown(f"**Log {log_data['Log #']}:**")
                            st.info(log_data['Description'])
                            st.markdown("---")
                        
                        # Option to download as CSV
                        logs_df = pd.DataFrame(generated_logs)
                        csv = logs_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Generated Logs as CSV",
                            data=csv,
                            file_name=f"synthetic_logs_{gen_employee}_{gen_category}.csv",
                            mime="text/csv",
                            key="download_logs"
                        )

else:
    st.info("👆 Please upload a CSV file to get started with the analysis")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Workforce Intelligence Platform • BuilD by TeamC • NLP miniproject</p>
</div>
""", unsafe_allow_html=True)
