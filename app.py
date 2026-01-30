import re
import joblib
import gradio as gr
from huggingface_hub import hf_hub_download

# Download model artifacts from HF Model Hub
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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(text):
    vec = tfidf.transform([clean_text(text)])
    prob = model.predict_proba(vec)[0][1]
    return {
        "label": "Toxic" if prob >= 0.5 else "Non-Toxic",
        "toxicity_probability": round(float(prob), 4),
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4),
    outputs="json",
    title="Toxic Comment Detection",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
