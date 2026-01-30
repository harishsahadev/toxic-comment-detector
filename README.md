---
title: Toxic Comment Detector
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Toxic Comment Detection

A lightweight **toxic comment detection system** built using **classical machine learning**
(TF-IDF + Logistic Regression) and deployed with **Gradio**.

The live application can be tested here:  
https://huggingface.co/spaces/harishsahadev/toxic-comment-detector

---

## What the model does

Given a text comment, the model predicts:
- **Non-Toxic**
- **Mildly Toxic**
- **Toxic**

along with a toxicity probability score.

**Model repository:**  
[Hugging Face – Toxic Comment Detector (Classical ML)](https://huggingface.co/harishsahadev/toxic-comment-detector-classical)

**Training notebook:**  
[Google Colab – Model training and experiments](https://colab.research.google.com/drive/1qcG1KK6hr946W4IWT_VJoQmdgjBA0H_G?usp=sharing)

---

## Toxicity thresholds

Predictions are mapped to labels using the following thresholds:

- **Non-Toxic**: probability `< 0.45`
- **Mildly Toxic**: `0.45 – 0.60`
- **Toxic**: `≥ 0.60`

These thresholds are chosen to **reduce false positives** while still flagging borderline content.

---

## Dataset

- Google Civil Comments Toxicity dataset
- Continuous toxicity scores converted to labels
- English-only comments

---

## Model & deployment

- TF-IDF word n-grams + Logistic Regression (scikit-learn)
- Class-weighted training to handle imbalance
- CPU-only inference
- Model artifacts loaded at runtime from Hugging Face Model Hub
- No pretrained deep learning models used

---

## Notes

This project is intended for **educational and demonstration purposes** and should not be used
as a standalone moderation system.

---

## License

MIT
