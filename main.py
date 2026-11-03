import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# 1. LOAD DATASET
# =====================================================

print("\n" + "=" * 55)
print(" PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
print("=" * 55)

FILE_NAME = "student_learning.csv"

if not os.path.exists(FILE_NAME):
    print(f"ERROR: {FILE_NAME} was not found.")
    print("Keep student_learning.csv in the same folder as main.py.")
    raise SystemExit

df = pd.read_csv(FILE_NAME)

print("\nDataset loaded successfully!")
print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)


# =====================================================
# 2. DATA CLEANING
# =====================================================

required_columns = [
    "subject",
    "interest",
    "learning_style",
    "completed_topic",
    "marks",
    "study_hours",
    "attendance",
    "recommended_topic"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:", missing_columns)
    raise SystemExit

df = df.drop_duplicates()
df = df.dropna(subset=required_columns)

for col in ["marks", "study_hours", "attendance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["marks", "study_hours", "attendance"]
)

df = df[
    (df["marks"].between(0, 100))
    & (df["attendance"].between(0, 100))
    & (df["study_hours"] >= 0)
].copy()

if len(df) < 4:
    print("\nERROR: Not enough valid records in the dataset.")
    raise SystemExit

print("\nData cleaning completed!")
print("Valid records:", len(df))


# =====================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================

os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# Graph 1: Average marks by subject
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="subject",
    y="marks",
    errorbar=None
)

plt.title("Average Student Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Marks")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("graphs/marks_by_subject.png")
plt.close()

# Graph 2: Study hours versus marks
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="subject",
    s=100
)

plt.title("Study Hours vs Student Marks")
plt.xlabel("Daily Study Hours")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("graphs/study_hours_vs_marks.png")
plt.close()

print("\nEDA completed!")
print("Graphs saved inside the graphs folder.")


# =====================================================
# 4. FEATURE SELECTION AND PREPROCESSING
# =====================================================

categorical_columns = [
    "subject",
    "interest",
    "learning_style",
    "completed_topic"
]

numeric_columns = [
    "marks",
    "study_hours",
    "attendance"
]

feature_columns = categorical_columns + numeric_columns

X = df[feature_columns].copy()
y = df["recommended_topic"].astype(str)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X[categorical_columns] = encoder.fit_transform(
    X[categorical_columns].astype(str)
)

print("\nFeature preprocessing completed!")


# =====================================================
# 5. TRAIN AND TEST THE ML MODEL
# =====================================================

# FIX: No stratify, so the small dataset does not cause
# the test-size-versus-number-of-classes error.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print(" MACHINE LEARNING MODEL EVALUATION")
print("=" * 55)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =====================================================
# 6. FEATURE IMPORTANCE GRAPH
# =====================================================

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    errorbar=None
)

plt.title("Feature Importance in Learning Recommendations")
plt.tight_layout()
plt.savefig("graphs/feature_importance.png")
plt.close()

print("\nFeature importance graph saved!")


# =====================================================
# 7. RECOMMENDATION FUNCTION
# =====================================================

def recommend_topics(
    subject,
    interest,
    learning_style,
    completed_topic,
    marks,
    study_hours,
    attendance
):
    student_data = pd.DataFrame([{
        "subject": subject,
        "interest": interest,
        "learning_style": learning_style,
        "completed_topic": completed_topic,
        "marks": marks,
        "study_hours": study_hours,
        "attendance": attendance
    }])

    student_data[categorical_columns] = encoder.transform(
        student_data[categorical_columns].astype(str)
    )

    student_data = student_data[feature_columns]

    predicted_topic = model.predict(student_data)[0]

    # Recommend topics found in the dataset for this subject.
    subject_rows = df[
        df["subject"].str.lower() == subject.lower()
    ]

    topic_counts = (
        subject_rows["recommended_topic"]
        .value_counts()
        .index.tolist()
    )

    recommendations = []

    if predicted_topic.lower() != completed_topic.lower():
        recommendations.append(predicted_topic)

    for topic in topic_counts:
        if (
            topic.lower() != completed_topic.lower()
            and topic not in recommendations
        ):
            recommendations.append(topic)

        if len(recommendations) == 3:
            break

    if not recommendations:
        recommendations.append(predicted_topic)

    return predicted_topic, recommendations


# =====================================================
# 8. TAKE STUDENT INPUT
# =====================================================

print("\n" + "=" * 55)
print(" ENTER STUDENT DETAILS")
print("=" * 55)

available_subjects = sorted(df["subject"].astype(str).unique())

print("\nAvailable subjects:", ", ".join(available_subjects))

subject = input("Enter subject: ").strip()
interest = input(
    "Enter interest (e.g. Programming, Technology, ProblemSolving): "
).strip()
learning_style = input(
    "Enter learning style (Video, Reading, Practice): "
).strip()
completed_topic = input("Enter the topic already completed: ").strip()

try:
    marks = float(input("Enter previous marks (0-100): "))
    study_hours = float(input("Enter daily study hours: "))
    attendance = float(input("Enter attendance percentage (0-100): "))

except ValueError:
    print("\nERROR: Enter numbers for marks, study hours and attendance.")
    raise SystemExit

if not 0 <= marks <= 100:
    print("\nERROR: Marks must be between 0 and 100.")
    raise SystemExit

if not 0 <= attendance <= 100:
    print("\nERROR: Attendance must be between 0 and 100.")
    raise SystemExit

if study_hours < 0:
    print("\nERROR: Study hours cannot be negative.")
    raise SystemExit

try:
    predicted_topic, recommendations = recommend_topics(
        subject,
        interest,
        learning_style,
        completed_topic,
        marks,
        study_hours,
        attendance
    )

except ValueError:
    print(
        "\nERROR: The entered details could not be processed. "
        "Check the spelling of your subject and other categories."
    )
    raise SystemExit


# =====================================================
# 9. DISPLAY RECOMMENDATIONS
# =====================================================

print("\n" + "=" * 55)
print(" PERSONALIZED LEARNING RECOMMENDATIONS")
print("=" * 55)

print("\nStudent subject:", subject)
print("Completed topic:", completed_topic)
print("Predicted next topic:", predicted_topic)

print("\nRecommended topics to study next:")

for number, topic in enumerate(recommendations, start=1):
    print(f"{number}. {topic}")

if marks < 60:
    print("\nSuggestion: Revise the basics and practise more questions.")
elif study_hours < 2:
    print("\nSuggestion: Try following a regular daily study schedule.")
else:
    print("\nSuggestion: Continue your study routine and practise the next topic.")

print("\nGraphs available in the 'graphs' folder:")
print("1. marks_by_subject.png")
print("2. study_hours_vs_marks.png")
print("3. feature_importance.png")

print("\nRecommendation process completed!")
print("=" * 55)