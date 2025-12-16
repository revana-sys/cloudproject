# ============================================================================
# HOSPITAL PRIORITY SCHEDULING SYSTEM WITH ML PREDICTION
# ============================================================================

#@title Import Libraries
import os
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
import joblib

warnings.filterwarnings('ignore')

print("🏥 Setting up Hospital Priority Scheduling System...")

# ============================================================================
# PART 1: DATA GENERATION & ML MODEL
# ============================================================================

# Generate training data
np.random.seed(42)
num_samples = 5000

data = {
    'patient_id': np.arange(1, num_samples + 1),
    'arrival_time': np.random.exponential(60, num_samples),
    'treatment_time': np.random.gamma(30, 2, num_samples),
    'age': np.random.randint(1, 100, num_samples),
    'condition_severity': np.random.randint(1, 11, num_samples),
    'blood_pressure': np.random.normal(120, 20, num_samples),
    'heart_rate': np.random.normal(80, 15, num_samples),
    'temperature': np.random.normal(98.6, 1.5, num_samples),
    'pain_level': np.random.randint(0, 11, num_samples),
    'consciousness': np.random.binomial(1, 0.9, num_samples),
    'breathing_difficulty': np.random.binomial(1, 0.3, num_samples),
    'bleeding': np.random.binomial(1, 0.2, num_samples),
    'fracture': np.random.binomial(1, 0.25, num_samples),
    'allergies': np.random.binomial(1, 0.15, num_samples),
    'pre_existing_conditions': np.random.choice([0, 1, 2, 3], num_samples, p=[0.6, 0.25, 0.1, 0.05]),
    'priority_score': np.random.choice([1, 2, 3, 4, 5], num_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
}

df = pd.DataFrame(data)
MODEL_PATH = r"D:\cloud code\model.pkl"

# Create emergency target
df['emergency'] = np.where(
    ((df['condition_severity'] >= 7) &
     ((df['breathing_difficulty'] == 1) | (df['bleeding'] == 1) | (df['consciousness'] == 0))) |
    ((df['pain_level'] >= 7) & (df['age'] > 60)) |
    ((df['priority_score'] <= 2) & (df['heart_rate'] > 110)) |
    (df['condition_severity'] >= 9),
    1, 0
)

print(f"Emergency rate: {df['emergency'].mean():.2%}")

# Train model
feature_columns = [
    'age', 'condition_severity', 'pain_level', 'blood_pressure',
    'heart_rate', 'temperature', 'consciousness', 'breathing_difficulty',
    'bleeding', 'fracture', 'allergies', 'pre_existing_conditions', 'priority_score'
]

X = df[feature_columns]
y = df['emergency']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if os.path.exists(MODEL_PATH):
    print("📦 Loading saved ML model...")
    model = joblib.load(MODEL_PATH)
else:
    print("🧠 Training new ML model...")

    model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, MODEL_PATH)
print("💾 Model saved as model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ ML Model accuracy: {accuracy:.2%}")



# ============================================================================
# PART 2: PRIORITY SCHEDULING CLASSES
# ============================================================================

class Patient:
    def __init__(self, patient_id, name, arrival_time, treatment_time, age,
                 condition_severity, pain_level, emergency=0):
        self.patient_id = patient_id
        self.name = name
        self.arrival_time = arrival_time
        self.treatment_time = treatment_time
        self.age = age
        self.condition_severity = condition_severity
        self.pain_level = pain_level
        self.emergency = emergency
        self.waiting_time = 0
        self.treatment_start = 0
        self.treatment_end = 0
        self.total_time = 0
        self.assigned_doctor = None

class Doctor:
    def __init__(self, doctor_id, name):
        self.doctor_id = doctor_id
        self.name = name
        self.available_from = 0
        self.patients_treated = []
        self.total_busy_time = 0

def priority_schedule(patients, num_doctors=2):
    """Priority Scheduling: Emergencies jump queue"""
    if not patients:
        return [], [], []

    # Initialize doctors
    doctors = [Doctor(i+1, f"Dr. {chr(65+i)}") for i in range(num_doctors)]

    # Sort by arrival time initially
    patients_sorted = sorted(patients, key=lambda x: x.arrival_time)

    timeline = []
    scheduled = []
    current_time = 0
    waiting_emergency = []
    waiting_normal = []
    all_patients = patients_sorted.copy()

    while all_patients or waiting_emergency or waiting_normal:
        # Add arrived patients to appropriate waiting lists
        arrived = [p for p in all_patients if p.arrival_time <= current_time]
        for p in arrived:
            if p.emergency == 1:
                waiting_emergency.append(p)
            else:
                waiting_normal.append(p)
            all_patients.remove(p)

        # Sort emergency list by severity (higher severity first)
        waiting_emergency.sort(key=lambda x: (-x.condition_severity, x.arrival_time))

        # Sort normal list by arrival time
        waiting_normal.sort(key=lambda x: x.arrival_time)

        # Sort doctors by availability
        doctors.sort(key=lambda d: d.available_from)

        # Assign patients to available doctors
        for doctor in doctors:
            if doctor.available_from <= current_time:
                # ALWAYS prioritize emergency over normal
                if waiting_emergency:
                    patient = waiting_emergency.pop(0)
                elif waiting_normal:
                    patient = waiting_normal.pop(0)
                else:
                    continue  # No patients to assign

                # Calculate start time
                start_time = max(current_time, patient.arrival_time)
                patient.waiting_time = start_time - patient.arrival_time

                # Calculate end time
                end_time = start_time + patient.treatment_time
                doctor.available_from = end_time
                doctor.total_busy_time += patient.treatment_time
                doctor.patients_treated.append(patient)
                patient.assigned_doctor = doctor.name

                # Record patient times
                patient.treatment_start = start_time
                patient.treatment_end = end_time
                patient.total_time = end_time - patient.arrival_time

                # Add to timeline
                timeline.append((f'{patient.name} ({doctor.name})', start_time, end_time))
                scheduled.append(patient)

        # Advance time
        if not waiting_emergency and not waiting_normal and all_patients:
            # No patients waiting, advance to next arrival
            next_arrival = min(p.arrival_time for p in all_patients)
            if next_arrival > current_time:
                timeline.append(('Idle', current_time, next_arrival))
                current_time = next_arrival
            else:
                current_time += 1
        else:
            # Move time forward to next doctor availability
            next_doctor_time = min(d.available_from for d in doctors)
            if next_doctor_time > current_time:
                current_time = next_doctor_time
            else:
                current_time += 0.1

    return scheduled, timeline, doctors

def calculate_metrics(patients, timeline, doctors):
    """Calculate scheduling metrics"""
    if not patients:
        return {}

    total_time = timeline[-1][2] if timeline else 0

    # Separate emergency and normal patients
    emergency_patients = [p for p in patients if p.emergency == 1]
    normal_patients = [p for p in patients if p.emergency == 0]

    # Calculate average waiting times
    avg_waiting_emergency = np.mean([p.waiting_time for p in emergency_patients]) if emergency_patients else 0
    avg_waiting_normal = np.mean([p.waiting_time for p in normal_patients]) if normal_patients else 0

    metrics = {
        'total_patients': len(patients),
        'emergency_count': len(emergency_patients),
        'normal_count': len(normal_patients),
        'avg_waiting_time': np.mean([p.waiting_time for p in patients]),
        'avg_waiting_emergency': avg_waiting_emergency,
        'avg_waiting_normal': avg_waiting_normal,
        'avg_total_time': np.mean([p.total_time for p in patients]),
        'throughput': len(patients) / total_time if total_time > 0 else 0,
        'doctor_utilization': (sum(p.treatment_time for p in patients) / (len(doctors) * total_time) * 100) if total_time > 0 else 0,
        'total_time': total_time,
        'num_doctors': len(doctors),
        'avg_treatment_time': np.mean([p.treatment_time for p in patients])
    }
    return metrics

# ============================================================================
# PART 3: MANUAL DATA ENTRY FUNCTIONS
# ============================================================================

def create_manual_patient_form(patient_num, default_values=None):
    """Create form for a single patient"""
    if default_values is None:
        default_values = {
            'name': f'Patient {patient_num}',
            'arrival_time': (patient_num-1)*30,
            'treatment_time': 30,
            'age': 50,
            'condition_severity': 5,
            'pain_level': 5,
            'priority_score': 3,
            'blood_pressure': 120,
            'heart_rate': 80
        }

    form_html = f"""
    <div class="patient-form" style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 4px solid #3498db;">
        <h4 style="margin-top: 0; color: #2c3e50;">👤 Patient {patient_num}</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Name:</label>
                <input type="text" id="name_{patient_num}" name="name_{patient_num}" value="{default_values['name']}"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Arrival Time (min):</label>
                <input type="number" id="arrival_{patient_num}" name="arrival_{patient_num}" value="{default_values['arrival_time']}" min="0" step="5"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Treatment Time (min):</label>
                <input type="number" id="treatment_{patient_num}" name="treatment_{patient_num}" value="{default_values['treatment_time']}" min="5" step="5"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Age:</label>
                <input type="number" id="age_{patient_num}" name="age_{patient_num}" value="{default_values['age']}" min="0" max="120"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Condition Severity (1-10):</label>
                <select id="severity_{patient_num}" name="severity_{patient_num}"
                        style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
                    <option value="1" {'selected' if default_values['condition_severity'] == 1 else ''}>1 - Very Mild</option>
                    <option value="2" {'selected' if default_values['condition_severity'] == 2 else ''}>2 - Mild</option>
                    <option value="3" {'selected' if default_values['condition_severity'] == 3 else ''}>3</option>
                    <option value="4" {'selected' if default_values['condition_severity'] == 4 else ''}>4</option>
                    <option value="5" {'selected' if default_values['condition_severity'] == 5 else ''}>5 - Moderate</option>
                    <option value="6" {'selected' if default_values['condition_severity'] == 6 else ''}>6</option>
                    <option value="7" {'selected' if default_values['condition_severity'] == 7 else ''}>7</option>
                    <option value="8" {'selected' if default_values['condition_severity'] == 8 else ''}>8 - Severe</option>
                    <option value="9" {'selected' if default_values['condition_severity'] == 9 else ''}>9 - Critical</option>
                    <option value="10" {'selected' if default_values['condition_severity'] == 10 else ''}>10 - Life Threatening</option>
                </select>
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Pain Level (0-10):</label>
                <select id="pain_{patient_num}" name="pain_{patient_num}"
                        style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
                    <option value="0" {'selected' if default_values['pain_level'] == 0 else ''}>0 - No Pain</option>
                    <option value="1" {'selected' if default_values['pain_level'] == 1 else ''}>1</option>
                    <option value="2" {'selected' if default_values['pain_level'] == 2 else ''}>2</option>
                    <option value="3" {'selected' if default_values['pain_level'] == 3 else ''}>3</option>
                    <option value="4" {'selected' if default_values['pain_level'] == 4 else ''}>4</option>
                    <option value="5" {'selected' if default_values['pain_level'] == 5 else ''}>5 - Moderate Pain</option>
                    <option value="6" {'selected' if default_values['pain_level'] == 6 else ''}>6</option>
                    <option value="7" {'selected' if default_values['pain_level'] == 7 else ''}>7</option>
                    <option value="8" {'selected' if default_values['pain_level'] == 8 else ''}>8 - Severe Pain</option>
                    <option value="9" {'selected' if default_values['pain_level'] == 9 else ''}>9 - Very Severe</option>
                    <option value="10" {'selected' if default_values['pain_level'] == 10 else ''}>10 - Worst Possible</option>
                </select>
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Priority Score (1-5):</label>
                <select id="priority_{patient_num}" name="priority_{patient_num}"
                        style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
                    <option value="1" {'selected' if default_values['priority_score'] == 1 else ''}>1 - Immediate</option>
                    <option value="2" {'selected' if default_values['priority_score'] == 2 else ''}>2 - Very Urgent</option>
                    <option value="3" {'selected' if default_values['priority_score'] == 3 else ''}>3 - Urgent</option>
                    <option value="4" {'selected' if default_values['priority_score'] == 4 else ''}>4 - Standard</option>
                    <option value="5" {'selected' if default_values['priority_score'] == 5 else ''}>5 - Non-Urgent</option>
                </select>
            </div>
            <div>
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #2c3e50;">Blood Pressure:</label>
                <input type="number" id="bp_{patient_num}" name="bp_{patient_num}" value="{default_values['blood_pressure']}" min="60" max="200" step="1"
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
        </div>
    </div>
    """
    return form_html

def parse_manual_data(num_patients, *form_data):
    """Parse form data into Patient objects with ML prediction"""
    patients = []

    for i in range(num_patients):
        try:
            # Extract data (8 fields per patient)
            start_idx = i * 8
            name = str(form_data[start_idx] or f'Patient {i+1}')
            arrival_time = float(form_data[start_idx + 1] or (i * 30))
            treatment_time = float(form_data[start_idx + 2] or 30)
            age = int(float(form_data[start_idx + 3] or 50))
            condition_severity = int(float(form_data[start_idx + 4] or 5))
            pain_level = int(float(form_data[start_idx + 5] or 5))
            priority_score = int(float(form_data[start_idx + 6] or 3))
            blood_pressure = float(form_data[start_idx + 7] or 120)

            # Create features for ML prediction
            features = pd.DataFrame([{
                'age': age,
                'condition_severity': condition_severity,
                'pain_level': pain_level,
                'blood_pressure': blood_pressure,
                'heart_rate': 80,
                'temperature': 98.6,
                'consciousness': 1,
                'breathing_difficulty': 0,
                'bleeding': 0,
                'fracture': 0,
                'allergies': 0,
                'pre_existing_conditions': 0,
                'priority_score': priority_score,
            }])

            # Ensure all columns exist
            for col in feature_columns:
                if col not in features.columns:
                    features[col] = 0

            # Predict emergency using ML model
            emergency = model.predict(features[feature_columns])[0]

            # Create patient
            patient = Patient(
                patient_id=i+1,
                name=name,
                arrival_time=arrival_time,
                treatment_time=treatment_time,
                age=age,
                condition_severity=condition_severity,
                pain_level=pain_level,
                emergency=emergency
            )

            patients.append(patient)

        except Exception as e:
            print(f"Error creating patient {i+1}: {e}")
            # Create default patient
            patients.append(Patient(
                patient_id=i+1,
                name=f'Patient {i+1}',
                arrival_time=i*30,
                treatment_time=30,
                age=50,
                condition_severity=5,
                pain_level=5,
                emergency=0
            ))

    return patients

# ============================================================================
# PART 4: GRADIO INTERFACE - SIMPLIFIED
# ============================================================================

css = """
.gradio-container {
    max-width: 1400px !important;
}
.header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    text-align: center;
}
.header h1 {
    margin: 0;
    color: white !important;
    font-size: 2.2em;
}
.config-panel {
    background: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.manual-forms-container {
    background: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    max-height: 700px;
    overflow-y: auto;
}
.run-button {
    background: linear-gradient(135deg, #00c853 0%, #64dd17 100%) !important;
    color: white !important;
    font-weight: bold !important;
    padding: 15px 30px !important;
    border-radius: 10px !important;
    border: none !important;
    width: 100%;
}
.metrics-panel {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #3498db;
}
.emergency-highlight {
    background: linear-gradient(135deg, #ff5252 0%, #ff1744 100%);
    color: white;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    text-align: center;
    font-weight: bold;
}
.normal-highlight {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
    color: white;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    text-align: center;
    font-weight: bold;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML("""
    <div class="header">
        <h1>🏥 Hospital Priority Scheduling System</h1>
        <h3>ML-Powered Emergency Prediction & Priority Scheduling</h3>
    </div>
    """)

    with gr.Row():
        # ======================================================================
        # LEFT PANEL: CONFIGURATION
        # ======================================================================
        with gr.Column(scale=1):
            with gr.Group("⚙️ Configuration", elem_classes="config-panel"):
                gr.Markdown("### Patient Source")
                mode = gr.Radio(
                    choices=["auto", "manual"],
                    value="manual",
                    label="Data Source",
                    info="Enter patient data manually"
                )

                gr.Markdown("### Hospital Resources")
                num_patients = gr.Slider(
                    minimum=3, maximum=8, value=4, step=1,
                    label="Number of Patients",
                    info="How many patients to schedule"
                )

                num_doctors = gr.Slider(
                    minimum=1, maximum=3, value=2, step=1,
                    label="Number of Doctors",
                    info="Available medical staff"
                )

                gr.Markdown("### ML Settings")
                use_ml = gr.Checkbox(
                    value=True, label="Use ML for Emergency Prediction",
                    info="AI predicts if patient needs immediate attention"
                )

                random_seed = gr.Number(
                    value=42, label="Random Seed (Auto Mode)",
                    visible=False
                )

                run_btn = gr.Button("🚀 Run Priority Scheduling",
                                  variant="primary",
                                  elem_classes="run-button")

        # ======================================================================
        # MIDDLE PANEL: MANUAL DATA ENTRY
        # ======================================================================
        with gr.Column(scale=2):
            gr.Markdown("### 👨‍⚕️ Enter Patient Details")
            manual_forms = gr.HTML(
                label="Patient Entry Forms",
                elem_classes="manual-forms-container"
            )

            # Form inputs will be stored here
            form_inputs = []

        # ======================================================================
        # RIGHT PANEL: METRICS & ANALYSIS
        # ======================================================================
        with gr.Column(scale=1):
            with gr.Group("📊 Performance Metrics", elem_classes="metrics-panel"):
                metrics_display = gr.Markdown("""
                ## Performance Metrics

                **Waiting Times:**
                - Emergency Patients: --
                - Normal Patients: --
                - Average: --

                **Efficiency:**
                - Doctor Utilization: --
                - Throughput: --
                - Total Time: --

                **Results will appear here after scheduling**
                """)

    # ======================================================================
    # RESULTS TABS
    # ======================================================================
    with gr.Tabs():
        with gr.TabItem("📋 Patient List"):
            patient_table = gr.HTML(label="Patient Information")

        with gr.TabItem("📅 Schedule Timeline"):
            schedule_timeline = gr.Plot(label="Priority Schedule")

        with gr.TabItem("👨‍⚕️ Doctor Analysis"):
            doctor_analysis = gr.Plot(label="Doctor Workload")

    # ======================================================================
    # DYNAMIC FORM GENERATION
    # ======================================================================

    def update_forms(mode, num_patients):
        """Update forms based on mode"""
        if mode == "manual":
            forms_html = "<div style='padding: 10px;'>"

            # Create different default scenarios to show priority scheduling
            scenario_defaults = [
                {'name': 'John (Emergency)', 'arrival_time': 0, 'treatment_time': 25, 'age': 68,
                 'condition_severity': 9, 'pain_level': 8, 'priority_score': 1, 'blood_pressure': 180},
                {'name': 'Sarah (Normal)', 'arrival_time': 10, 'treatment_time': 35, 'age': 42,
                 'condition_severity': 4, 'pain_level': 3, 'priority_score': 4, 'blood_pressure': 120},
                {'name': 'Mike (Emergency)', 'arrival_time': 25, 'treatment_time': 30, 'age': 75,
                 'condition_severity': 8, 'pain_level': 7, 'priority_score': 2, 'blood_pressure': 165},
                {'name': 'Lisa (Normal)', 'arrival_time': 40, 'treatment_time': 40, 'age': 35,
                 'condition_severity': 5, 'pain_level': 4, 'priority_score': 3, 'blood_pressure': 115},
            ]

            for i in range(num_patients):
                if i < len(scenario_defaults):
                    defaults = scenario_defaults[i]
                else:
                    defaults = {
                        'name': f'Patient {i+1}',
                        'arrival_time': i*30,
                        'treatment_time': 30,
                        'age': 50,
                        'condition_severity': 5,
                        'pain_level': 5,
                        'priority_score': 3,
                        'blood_pressure': 120
                    }

                forms_html += create_manual_patient_form(i+1, defaults)

            forms_html += "</div>"

            # Create input components
            inputs = []
            for i in range(num_patients):
                if i < len(scenario_defaults):
                    defaults = scenario_defaults[i]
                else:
                    defaults = {
                        'name': f'Patient {i+1}',
                        'arrival_time': i*30,
                        'treatment_time': 30,
                        'age': 50,
                        'condition_severity': 5,
                        'pain_level': 5,
                        'priority_score': 3,
                        'blood_pressure': 120
                    }

                inputs.extend([
                    gr.Textbox(value=defaults['name'], visible=False),
                    gr.Number(value=defaults['arrival_time'], visible=False),
                    gr.Number(value=defaults['treatment_time'], visible=False),
                    gr.Number(value=defaults['age'], visible=False),
                    gr.Number(value=defaults['condition_severity'], visible=False),
                    gr.Number(value=defaults['pain_level'], visible=False),
                    gr.Number(value=defaults['priority_score'], visible=False),
                    gr.Number(value=defaults['blood_pressure'], visible=False),
                ])

            return [forms_html] + inputs

        else:
            # Auto mode
            forms_html = """
            <div style="text-align: center; padding: 50px; background: #f8f9fa; border-radius: 10px;">
                <h3>📊 Auto Mode Selected</h3>
                <p>Patients will be auto-generated when you run the scheduling.</p>
            </div>
            """

            # Create hidden inputs
            inputs = []
            for i in range(num_patients):
                inputs.extend([
                    gr.Textbox(visible=False),
                    gr.Number(visible=False),
                    gr.Number(visible=False),
                    gr.Number(visible=False),
                    gr.Number(visible=False),
                    gr.Number(visible=False),
                    gr.Number(visible=False),
                    gr.Number(visible=False),
                ])

            return [forms_html] + inputs

    # Connect updates
    mode.change(
        fn=update_forms,
        inputs=[mode, num_patients],
        outputs=[manual_forms] + form_inputs
    )

    num_patients.change(
        fn=update_forms,
        inputs=[mode, num_patients],
        outputs=[manual_forms] + form_inputs
    )

    # ======================================================================
    # AUTO-GENERATION FUNCTION
    # ======================================================================

    def generate_auto_patients(num_patients, random_seed):
        """Generate patients for auto mode"""
        np.random.seed(random_seed)
        patients = []
        current_time = 0

        patient_names = ["John Smith", "Maria Garcia", "David Chen", "Sarah Johnson",
                        "Robert Williams", "Lisa Brown", "Michael Davis", "Jennifer Miller"]

        for i in range(num_patients):
            name = patient_names[i % len(patient_names)]
            arrival_time = current_time + np.random.exponential(25)
            treatment_time = np.random.gamma(30, 2)
            age = np.random.randint(1, 100)
            condition_severity = np.random.randint(1, 11)
            pain_level = np.random.randint(0, 11)

            # Create features for ML prediction
            features = pd.DataFrame([{
                'age': age,
                'condition_severity': condition_severity,
                'pain_level': pain_level,
                'blood_pressure': np.random.normal(120, 25),
                'heart_rate': np.random.normal(85, 20),
                'temperature': np.random.normal(98.6, 2),
                'consciousness': np.random.binomial(1, 0.85),
                'breathing_difficulty': np.random.binomial(1, 0.35),
                'bleeding': np.random.binomial(1, 0.25),
                'fracture': np.random.binomial(1, 0.3),
                'allergies': np.random.binomial(1, 0.2),
                'pre_existing_conditions': np.random.choice([0, 1, 2, 3]),
                'priority_score': np.random.randint(1, 6),
            }])

            for col in feature_columns:
                if col not in features.columns:
                    features[col] = 0

            emergency = model.predict(features[feature_columns])[0]

            patients.append(Patient(
                patient_id=i+1,
                name=name,
                arrival_time=round(arrival_time, 1),
                treatment_time=round(treatment_time, 1),
                age=age,
                condition_severity=condition_severity,
                pain_level=pain_level,
                emergency=emergency
            ))

            current_time = arrival_time

        return patients

    # ======================================================================
    # MAIN SCHEDULING FUNCTION - PRIORITY ONLY
    # ======================================================================

    def run_priority_scheduling(mode, num_patients, num_doctors, use_ml, *form_data):
        """Run priority scheduling only"""

        # Get patients based on mode
        if mode == "manual":
            patients = parse_manual_data(num_patients, *form_data)
        else:
            patients = generate_auto_patients(num_patients, 42)

        # Create patient table
        patient_table_html = """
        <style>
            .patient-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-family: Arial, sans-serif;
            }
            .patient-table th {
                background: #2c3e50;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            .patient-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .patient-table tr:hover {
                background-color: #f5f5f5;
            }
            .emergency-row {
                background-color: #ffebee !important;
                color: #c62828;
                font-weight: bold;
            }
            .normal-row {
                background-color: #e8f5e9 !important;
                color: #2e7d32;
            }
            .doctor-cell {
                font-weight: bold;
                color: #1565c0;
            }
        </style>
        <table class="patient-table">
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Arrival</th>
                <th>Treatment</th>
                <th>Age</th>
                <th>Severity</th>
                <th>Status</th>
                <th>Doctor</th>
            </tr>
        """

        for p in patients:
            row_class = "emergency-row" if p.emergency == 1 else "normal-row"
            status_text = '🚨 EMERGENCY' if p.emergency == 1 else '📄 NORMAL'
            patient_table_html += f"""
            <tr class="{row_class}">
                <td>{p.patient_id}</td>
                <td><strong>{p.name}</strong></td>
                <td>{p.arrival_time:.1f} min</td>
                <td>{p.treatment_time:.1f} min</td>
                <td>{p.age}</td>
                <td>{p.condition_severity}/10</td>
                <td>{status_text}</td>
                <td class="doctor-cell">--</td>
            </tr>
            """
        patient_table_html += "</table>"

        # Run PRIORITY scheduling
        priority_patients = [Patient(p.patient_id, p.name, p.arrival_time, p.treatment_time,
                                    p.age, p.condition_severity, p.pain_level, p.emergency)
                            for p in patients]
        priority_scheduled, priority_timeline, priority_doctors = priority_schedule(priority_patients, num_doctors)
        priority_metrics = calculate_metrics(priority_scheduled, priority_timeline, priority_doctors)

        # Update patient table with doctor assignments
        patient_table_html = """
        <style>
            .patient-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-family: Arial, sans-serif;
            }
            .patient-table th {
                background: #2c3e50;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            .patient-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .patient-table tr:hover {
                background-color: #f5f5f5;
            }
            .emergency-row {
                background-color: #ffebee !important;
                color: #c62828;
                font-weight: bold;
            }
            .normal-row {
                background-color: #e8f5e9 !important;
                color: #2e7d32;
            }
            .doctor-cell {
                font-weight: bold;
                color: #1565c0;
            }
            .waiting-cell {
                font-weight: bold;
                color: #ff9800;
            }
        </style>
        <table class="patient-table">
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Arrival</th>
                <th>Treatment</th>
                <th>Wait</th>
                <th>Total</th>
                <th>Status</th>
                <th>Doctor</th>
            </tr>
        """

        for p in priority_scheduled:
            row_class = "emergency-row" if p.emergency == 1 else "normal-row"
            status_text = '🚨 EMERGENCY' if p.emergency == 1 else '📄 NORMAL'
            patient_table_html += f"""
            <tr class="{row_class}">
                <td>{p.patient_id}</td>
                <td><strong>{p.name}</strong></td>
                <td>{p.arrival_time:.1f} min</td>
                <td>{p.treatment_time:.1f} min</td>
                <td class="waiting-cell">{p.waiting_time:.1f} min</td>
                <td>{p.total_time:.1f} min</td>
                <td>{status_text}</td>
                <td class="doctor-cell">{p.assigned_doctor}</td>
            </tr>
            """
        patient_table_html += "</table>"

        # Create timeline visualization
        fig_timeline = go.Figure()

        for name, start, end in priority_timeline[:20]:  # Show first 20 entries
            if 'Idle' in name:
                color = '#cccccc'
                hover_text = f"Idle<br>{start:.0f}-{end:.0f} min"
            else:
                # Check if this is an emergency patient
                patient_name = name.split(' (')[0]
                patient = next((p for p in priority_scheduled if p.name == patient_name), None)
                if patient and patient.emergency == 1:
                    color = '#e74c3c'  # Red for emergencies
                else:
                    color = '#2ecc71'  # Green for normals

                hover_text = f"{name}<br>Start: {start:.0f} min<br>End: {end:.0f} min<br>Duration: {end-start:.0f} min"

            fig_timeline.add_trace(go.Bar(
                y=[name],
                x=[end - start],
                base=start,
                orientation='h',
                marker_color=color,
                text=hover_text.split('<br>')[0],
                textposition='inside',
                hoverinfo='text',
                hovertemplate=hover_text + "<extra></extra>"
            ))

        fig_timeline.update_layout(
            title=f"Priority Schedule Timeline ({num_doctors} Doctor{'s' if num_doctors > 1 else ''})",
            barmode='stack',
            height=max(400, len(priority_timeline) * 25),
            xaxis_title="Time (minutes)",
            yaxis_title="Patient (Doctor)",
            showlegend=False,
            template="plotly_white"
        )

        # Create doctor analysis
        fig_doctors = go.Figure()

        doctor_names = [d.name for d in priority_doctors]
        patients_per_doctor = [len(d.patients_treated) for d in priority_doctors]
        busy_time_per_doctor = [d.total_busy_time for d in priority_doctors]

        fig_doctors.add_trace(go.Bar(
            name='Patients Treated',
            x=doctor_names,
            y=patients_per_doctor,
            marker_color='#3498db',
            text=[f'{p} patients' for p in patients_per_doctor],
            textposition='auto'
        ))

        fig_doctors.add_trace(go.Bar(
            name='Busy Time (min)',
            x=doctor_names,
            y=busy_time_per_doctor,
            marker_color='#2ecc71',
            text=[f'{t:.0f} min' for t in busy_time_per_doctor],
            textposition='auto'
        ))

        fig_doctors.update_layout(
            title="Doctor Workload Analysis",
            barmode='group',
            height=400,
            xaxis_title="Doctor",
            yaxis_title="Count / Minutes",
            template="plotly_white"
        )

        # Update metrics display
        metrics_text = f"""
        ## 📊 Performance Metrics

        ### Patient Summary:
        - **Total Patients**: {priority_metrics['total_patients']}
        - **Emergency Cases**: {priority_metrics['emergency_count']}
        - **Normal Cases**: {priority_metrics['normal_count']}

        ### Waiting Times:
        <div class="emergency-highlight">
        🚨 Emergency Patients: {priority_metrics['avg_waiting_emergency']:.1f} minutes
        </div>

        <div class="normal-highlight">
        📄 Normal Patients: {priority_metrics['avg_waiting_normal']:.1f} minutes
        </div>

        **Average Waiting Time**: {priority_metrics['avg_waiting_time']:.1f} minutes

        ### Efficiency Metrics:
        - **Doctor Utilization**: {priority_metrics['doctor_utilization']:.1f}%
        - **Throughput**: {priority_metrics['throughput']:.3f} patients/minute
        - **Total Schedule Time**: {priority_metrics['total_time']:.1f} minutes
        - **Average Treatment Time**: {priority_metrics['avg_treatment_time']:.1f} minutes

        ### Key Insight:
        Priority scheduling ensures emergency patients wait **{priority_metrics['avg_waiting_emergency']:.1f} minutes** on average,
        while maintaining {priority_metrics['doctor_utilization']:.1f}% doctor utilization.

        ### ML Model Info:
        - **Accuracy**: {accuracy:.2%}
        - **Emergency Prediction**: {'Enabled' if use_ml else 'Disabled'}
        - **Emergency Rate**: {(priority_metrics['emergency_count']/priority_metrics['total_patients']*100):.1f}%
        """

        return [
            patient_table_html,      # Patient list
            fig_timeline,            # Schedule timeline
            fig_doctors,             # Doctor analysis
            metrics_text             # Metrics display
        ]

    # ======================================================================
    # CONNECT RUN BUTTON
    # ======================================================================

    # Add random_seed to inputs even if not used
    all_inputs = [mode, num_patients, num_doctors, use_ml, random_seed] + form_inputs

    run_btn.click(
        fn=run_priority_scheduling,
        inputs=all_inputs,
        outputs=[patient_table, schedule_timeline, doctor_analysis, metrics_display]
    )

    # Initial form setup
    demo.load(
        fn=lambda: update_forms("manual", 4),
        inputs=[],
        outputs=[manual_forms] + form_inputs
    )

print("🚀 Launching Hospital PRIORITY Scheduling System...")
print("📊 Focus: Priority Scheduling with ML Prediction")
demo.launch(debug=True, share=True)
