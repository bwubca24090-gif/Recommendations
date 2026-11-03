import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Personalized Learning Recommendation System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Personalized Learning Recommendation System")

st.write(
    "An ML-based system that recommends suitable future learning "
    "topics and educational pathways based on a student's profile."
)

# --------------------------------------------------
# 2. GENERATE A DEMONSTRATION DATASET
# --------------------------------------------------

random.seed(42)

learning_paths = {
    "Python Programming": [
        "Python Basics",
        "Conditional Statements",
        "Loops",
        "Functions",
        "Object-Oriented Programming",
        "File Handling",
        "Python Projects"
    ],
    "Database Management": [
        "Database Fundamentals",
        "SQL Basics",
        "SQL Joins",
        "Database Design",
        "Normalization",
        "Advanced SQL",
        "Database Projects"
    ],
    "Data Science and Analytics": [
        "Data Analysis Basics",
        "Python for Data Analysis",
        "Pandas and NumPy",
        "Data Visualization",
        "Statistics",
        "Machine Learning Basics",
        "Data Science Projects"
    ],
    "Web Development": [
        "HTML Basics",
        "CSS Basics",
        "JavaScript Basics",
        "DOM Manipulation",
        "Frontend Projects",
        "Backend Development",
        "Full-Stack Projects"
    ],
    "AI and Machine Learning": [
        "AI Fundamentals",
        "Python for AI",
        "Mathematics for ML",
        "Data Preprocessing",
        "Supervised Learning",
        "Model Evaluation",
        "ML Projects"
    ],
    "Cybersecurity and Computer Networks": [
        "Computer Network Basics",
        "Networking Protocols",
        "Operating System Security",
        "Cybersecurity Fundamentals",
        "Network Security",
        "Ethical Hacking Basics",
        "Security Projects"
    ]
}

interests = list(learning_paths.keys())
study_preferences = ["Videos", "Reading", "Practical Projects"]
qualifications = ["Class 12", "Diploma", "Bachelor's Degree"]

rows = []

for _ in range(500):
    interest = random.choice(interests)
    topics = learning_paths[interest]
    completed_index = random.randint(0, len(topics) - 2)

    completed_topic = topics[completed_index]
    recommended_topic = topics[completed_index + 1]

    rows.append({
        "qualification": random.choice(qualifications),
        "interest": interest,
        "completed_topic": completed_topic,
        "study_hours": random.randint(1, 8),
        "performance": random.randint(40, 100),
        "preference": random.choice(study_preferences),
        "recommended_topic": recommended_topic
    })

df = pd.DataFrame(rows)

# --------------------------------------------------
# 3. DATA CLEANING AND PREPROCESSING
# --------------------------------------------------

df = df.drop_duplicates().dropna().reset_index(drop=True)

categorical_features = [
    "qualification",
    "interest",
    "completed_topic",
    "preference"
]

numerical_features = [
    "study_hours",
    "performance"
]

features = categorical_features + numerical_features
target = "recommended_topic"

X = df[features]
y = df[target]

# --------------------------------------------------
# 4. TRAIN THE MACHINE LEARNING MODEL
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=12,
            random_state=42
        )
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Explore Subjects",
        "Higher Studies",
        "Dataset Overview",
        "Model Performance",
        "Visualizations"
    ]
)

# --------------------------------------------------
# 6. HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.header("Welcome to Your Learning Assistant")

    st.write(
        "This system uses a Decision Tree machine learning model "
        "to suggest what a student can study next."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", len(df))
    col2.metric("Learning Areas", len(learning_paths))
    col3.metric("Model Test Accuracy", f"{model_accuracy:.1%}")

    st.subheader("Main Features")

    st.markdown("""
    - **Explore Subjects:** Get a next-topic recommendation based on your profile.
    - **Higher Studies:** Explore bachelor's and master's degree options.
    - **Dataset Overview:** Inspect the learning dataset.
    - **Model Performance:** View the evaluation results.
    - **Visualizations:** Explore charts about student learning data.
    """)

    st.info(
        "Start with Explore Subjects to receive a learning recommendation."
    )

    st.caption(
        "This demonstration uses a synthetic dataset, not real student records."
    )

# --------------------------------------------------
# 7. EXPLORE SUBJECTS PAGE
# --------------------------------------------------

elif page == "Explore Subjects":

    st.header("📖 Explore Subjects and Learning Topics")

    st.write(
        "Enter your learning details to receive a suggested next topic."
    )

    with st.form("learning_form"):

        student_name = st.text_input(
            "Student Name (optional)"
        )

        qualification = st.selectbox(
            "Current Education Level",
            qualifications
        )

        interest = st.selectbox(
            "Choose Your Subject",
            interests
        )

        topics = learning_paths[interest]

        completed_topic = st.selectbox(
            "What topic have you completed?",
            topics[:-1]
        )

        study_hours = st.slider(
            "Hours you study per day",
            min_value=1,
            max_value=12,
            value=3
        )

        performance = st.slider(
            "Your recent performance (%)",
            min_value=0,
            max_value=100,
            value=75
        )

        preference = st.selectbox(
            "Preferred Learning Method",
            study_preferences
        )

        submitted = st.form_submit_button(
            "Get My Recommendation"
        )

    if submitted:

        completed_index = topics.index(completed_topic)

        # Build the student's input for the ML model.
        student_data = pd.DataFrame([{
            "qualification": qualification,
            "interest": interest,
            "completed_topic": completed_topic,
            "study_hours": study_hours,
            "performance": performance,
            "preference": preference
        }])

        prediction = model.predict(student_data)[0]

        # Keep the recommendation consistent with the selected
        # subject's actual learning sequence.
        next_topic = topics[completed_index + 1]

        st.success("Recommendation generated successfully!")

        if student_name.strip():
            st.write(f"Hello, **{student_name.strip()}**!")

        st.subheader("Your Next Learning Topic")

        st.markdown(f"### 🎯 {next_topic}")

        st.write(
            f"Your selected subject is **{interest}** and your "
            f"completed topic is **{completed_topic}**."
        )

        st.write(
            f"Preferred learning method: **{preference}**"
        )

        if performance < 50:
            st.warning(
                "Consider revising your previous topic before "
                "moving forward."
            )
        elif performance < 75:
            st.info(
                "Revise important concepts and practise exercises "
                "alongside your next topic."
            )
        else:
            st.success(
                "Your reported performance looks encouraging. "
                "Continue with practice and projects."
            )

        st.subheader("Your Learning Roadmap")

        for index, topic in enumerate(topics):
            if index < completed_index:
                st.write(f"✅ {topic}")
            elif index == completed_index:
                st.write(f"📍 {topic} — Completed topic")
            elif topic == next_topic:
                st.write(f"⭐ {topic} — Recommended next")
            else:
                st.write(f"⬜ {topic}")

        with st.expander("View ML model output"):
            st.write(
                "The model's predicted topic for your input is:"
            )
            st.write(f"**{prediction}**")

            st.caption(
                "The roadmap recommendation follows the selected "
                "subject's predefined topic sequence. The displayed "
                "model prediction is shown separately."
            )

# --------------------------------------------------
# 8. HIGHER STUDIES PAGE
# --------------------------------------------------

elif page == "Higher Studies":

    st.header("🎓 Higher Study and Degree Recommendations")

    st.write(
        "First, tell us whether you have completed a bachelor's "
        "degree. The system will suggest the next educational step."
    )

    education_status = st.radio(
        "Have you completed a bachelor's degree?",
        [
            "No, I have not completed a bachelor's degree",
            "Yes, I have completed a bachelor's degree"
        ]
    )

    interest = st.selectbox(
        "Choose your area of interest",
        [
            "Software Development",
            "Data Science and Analytics",
            "Artificial Intelligence",
            "Cybersecurity and Networking",
            "Business and Management",
            "General Computer Science"
        ]
    )

    # CASE 1: STUDENT HAS NOT COMPLETED A BACHELOR'S DEGREE

    if education_status == "No, I have not completed a bachelor's degree":

        st.subheader("Recommended Bachelor's Degrees")

        bachelor_courses = {
            "Software Development": [
                ("BCA", "Bachelor of Computer Applications"),
                ("B.Tech CSE", "Computer Science and Engineering"),
                ("B.Sc. Computer Science", "Computer Science degree")
            ],
            "Data Science and Analytics": [
                ("B.Sc. Data Science", "Data Science and analytics"),
                ("BCA", "Computer Applications"),
                ("B.Tech CSE", "Computer Science and Engineering")
            ],
            "Artificial Intelligence": [
                ("B.Tech AI/ML", "Artificial Intelligence and Machine Learning"),
                ("B.Sc. Computer Science", "Computer Science degree"),
                ("BCA", "Computer Applications")
            ],
            "Cybersecurity and Networking": [
                ("B.Sc. Cybersecurity", "Cybersecurity-focused degree"),
                ("B.Tech CSE", "Computer Science and Engineering"),
                ("BCA", "Computer Applications")
            ],
            "Business and Management": [
                ("BBA", "Bachelor of Business Administration"),
                ("B.Com.", "Bachelor of Commerce"),
                ("BCA", "Computer Applications")
            ],
            "General Computer Science": [
                ("BCA", "Bachelor of Computer Applications"),
                ("B.Sc. Computer Science", "Computer Science degree"),
                ("B.Tech CSE", "Computer Science and Engineering")
            ]
        }

        st.info(
            "Your first step is to explore a suitable bachelor's degree. "
            "Master's degree suggestions will be relevant after you "
            "complete a bachelor's degree and meet the entry requirements."
        )

        for course, description in bachelor_courses[interest]:

            st.markdown(f"### {course}")
            st.write(description)

        st.subheader("Your Suggested Education Path")

        st.write("1. Complete an eligible bachelor's degree.")
        st.write("2. Build subject knowledge through projects and practice.")
        st.write("3. Check master's degree eligibility after graduation.")

        st.caption(
            "Check each college's official website for Class 12 "
            "requirements, entrance exams, fees, and course availability."
        )

    # CASE 2: STUDENT HAS COMPLETED A BACHELOR'S DEGREE

    else:

        bachelor_degree = st.selectbox(
            "Which bachelor's degree have you completed?",
            [
                "BCA",
                "B.Sc. Computer Science / IT",
                "B.Tech / B.E.",
                "B.Sc. Data Science / Mathematics / Statistics",
                "BBA / B.Com.",
                "BA",
                "Other bachelor's degree"
            ]
        )

        st.subheader("Recommended Master's Degrees")

        master_courses = {
            "Software Development": [
                ("MCA", "Master of Computer Applications"),
                ("M.Sc. Computer Science", "Advanced computer science"),
                ("M.Sc. Software Engineering", "Software design and development")
            ],
            "Data Science and Analytics": [
                ("M.Sc. Data Science", "Data analysis and machine learning"),
                ("M.Sc. Statistics", "Statistics and analytical methods"),
                ("MCA", "Computer applications and software")
            ],
            "Artificial Intelligence": [
                ("M.Sc. Artificial Intelligence", "AI and intelligent systems"),
                ("M.Sc. Data Science", "Data analysis and machine learning"),
                ("MCA", "Computer applications")
            ],
            "Cybersecurity and Networking": [
                ("M.Sc. Cybersecurity", "Security and cyber defence"),
                ("M.Sc. Computer Science", "Advanced computing and networks"),
                ("MCA", "Computer applications")
            ],
            "Business and Management": [
                ("MBA", "Master of Business Administration"),
                ("MBA Business Analytics", "Business decisions using data"),
                ("MCA", "Computer applications, if eligible")
            ],
            "General Computer Science": [
                ("MCA", "Master of Computer Applications"),
                ("M.Sc. Computer Science", "Advanced computer science"),
                ("M.Sc. Information Technology", "Advanced IT")
            ]
        }

        if bachelor_degree in ["BBA / B.Com.", "BA", "Other bachelor's degree"]:

            st.warning(
                "Eligibility for technical master's courses varies by "
                "university. Some programmes require mathematics or "
                "specific undergraduate subjects."
            )

        elif bachelor_degree == "B.Tech / B.E.":

            st.write(
                "Explore postgraduate programmes related to your "
                "engineering subject and the university's requirements."
            )

        elif bachelor_degree == "B.Sc. Data Science / Mathematics / Statistics":

            st.write(
                "Consider data science, statistics, AI, and related "
                "postgraduate programmes, subject to eligibility."
            )

        elif bachelor_degree == "BCA":

            st.write(
                "You can explore MCA and related postgraduate programmes. "
                "Check the eligibility requirements of each university."
            )

        elif bachelor_degree == "B.Sc. Computer Science / IT":

            st.write(
                "Explore MCA, M.Sc. Computer Science, and related "
                "postgraduate programmes, subject to eligibility."
            )

        for course, description in master_courses[interest]:

            st.markdown(f"### {course}")
            st.write(description)

        st.subheader("Your Suggested Education Path")

        st.write(f"1. Completed degree: **{bachelor_degree}**")
        st.write("2. Compare eligible postgraduate courses.")
        st.write("3. Check entrance exams, fees, and admission dates.")

        st.caption(
            "These are general suggestions, not admission guarantees. "
            "Verify required subjects, marks, entrance exams, and "
            "eligibility on each university's official website."
        )

# --------------------------------------------------
# 9. DATASET OVERVIEW PAGE
# --------------------------------------------------

elif page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    st.write(
        "This demonstration dataset contains student learning profiles "
        "and the next topic in a predefined learning sequence."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(features))
    col3.metric("Target Column", "Recommended Topic")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    dataset_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[column].dtype) for column in df.columns
        ],
        "Missing Values": [
            int(df[column].isnull().sum()) for column in df.columns
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df[numerical_features].describe(),
        use_container_width=True
    )

    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="learning_recommendation_dataset.csv",
        mime="text/csv"
    )

    st.warning(
        "The dataset is synthetically generated for project demonstration. "
        "It is not collected from real students or Kaggle."
    )

# --------------------------------------------------
# 10. MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    st.write(
        "The system uses a Decision Tree Classifier to predict "
        "a recommended next learning topic."
    )

    st.metric(
        "Test Accuracy",
        f"{model_accuracy:.2%}"
    )

    st.write(f"Training records: {len(X_train)}")
    st.write(f"Testing records: {len(X_test)}")

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.subheader("How the Model Works")

    st.markdown("""
    1. **Data collection:** Student profiles are represented in a dataset.
    2. **Preprocessing:** Categorical inputs are converted using one-hot encoding.
    3. **Training:** A Decision Tree learns patterns from the training data.
    4. **Testing:** The model is evaluated on a held-out test dataset.
    5. **Recommendation:** The model predicts a next learning topic.
    """)

    st.warning(
        "Because the dataset is synthetic and its target follows predefined "
        "topic sequences, test accuracy does not establish real-world "
        "recommendation quality. Evaluate the system with authentic, "
        "representative student data before drawing real-world conclusions."
    )

# --------------------------------------------------
# 11. VISUALIZATIONS PAGE
# --------------------------------------------------

elif page == "Visualizations":

    st.header("📈 Data Visualizations")

    # Chart 1: Students by interest

    st.subheader("1. Student Records by Interest")

    interest_counts = df["interest"].value_counts()

    fig1, ax1 = plt.subplots()

    interest_counts.plot(
        kind="bar",
        ax=ax1
    )

    ax1.set_xlabel("Learning Interest")
    ax1.set_ylabel("Number of Records")
    ax1.set_title("Records by Learning Interest")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    st.pyplot(fig1)
    plt.close(fig1)

    # Chart 2: Performance distribution

    st.subheader("2. Student Performance Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(
        df["performance"],
        bins=10,
        edgecolor="black"
    )

    ax2.set_xlabel("Performance (%)")
    ax2.set_ylabel("Number of Records")
    ax2.set_title("Distribution of Student Performance")

    st.pyplot(fig2)
    plt.close(fig2)

    # Chart 3: Study hours

    st.subheader("3. Daily Study Hours")

    study_counts = df["study_hours"].value_counts().sort_index()

    fig3, ax3 = plt.subplots()

    study_counts.plot(
        kind="bar",
        ax=ax3
    )

    ax3.set_xlabel("Study Hours per Day")
    ax3.set_ylabel("Number of Records")
    ax3.set_title("Daily Study Hours Distribution")

    st.pyplot(fig3)
    plt.close(fig3)

    # Chart 4: Most common recommendations

    st.subheader("4. Most Common Recommended Topics")

    recommendation_counts = (
        df["recommended_topic"].value_counts().head(10)
    )

    fig4, ax4 = plt.subplots()

    recommendation_counts.plot(
        kind="barh",
        ax=ax4
    )

    ax4.set_xlabel("Number of Records")
    ax4.set_ylabel("Recommended Topic")
    ax4.set_title("Top 10 Recommended Topics")
    ax4.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.caption(
        "These charts describe the synthetic demonstration dataset."
    )

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Personalized Learning Recommendation System"
)

st.sidebar.caption(
    "BCA Machine Learning Project"
)