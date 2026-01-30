import re
import joblib
import gradio as gr
from huggingface_hub import hf_hub_download


# ----------------------------
# Load model artifacts
# ----------------------------
tfidf_path = hf_hub_download(
    repo_id="harishsahadev/toxic-comment-detector-classical",
    filename="tfidf_vectorizer.joblib"
)

model_path = hf_hub_download(
    repo_id="harishsahadev/toxic-comment-detector-classical",
    filename="toxic_classifier.joblib"
)

tfidf = joblib.load(tfidf_path)
model = joblib.load(model_path)


# ----------------------------
# Text preprocessing
# ----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------
# Prediction logic
# ----------------------------
def predict(text: str):
    if not text or len(text.strip()) < 3:
        return {
            "label": "Invalid input",
            "toxicity_probability": 0.0,
            "note": "Please enter a longer comment."
        }

    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    prob = float(model.predict_proba(vec)[0][1])

    # Multi-level classification
    if prob < 0.45:
        label = "Non-Toxic"
    elif prob < 0.60:
        label = "Mildly Toxic"
    else:
        label = "Toxic"

    return {
        "label": label,
        "toxicity_probability": round(prob, 4),
    }


# ----------------------------
# Gradio UI
# ----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter a comment to analyze toxicity"
    ),
    outputs="json",
    title="Toxic Comment Detection",
    description=(
        "Classical ML toxicity detection using TF-IDF and Logistic Regression. "
        "Outputs three levels: Non-Toxic, Mildly Toxic, and Toxic."
    ),
    examples=[
        ["I completely disagree with you, but let's keep this civil."],
        ["That was a pretty stupid thing to say."],
        ["You are an idiot and nobody likes you."],
        ["I don't think you understand the topic at all."],
        ["I love watching these videos and how you explain what you are doing. Definitely one of my favorite channels to watch."]
    ],
    cache_examples=False
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )