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

This Space demonstrates a **toxic comment detection system** built using
**classical machine learning techniques** and deployed with **Gradio**.

---

## Overview

Given a text comment, the model predicts whether it is:
- Toxic
- Non-toxic

along with a confidence score.

This project focuses on a **clean ML pipeline and deployment workflow** rather than large pretrained models.

---

## Dataset

- Google Civil Comments Toxicity dataset
- Continuous toxicity scores converted into binary labels
- Subsampled for efficient training

---

## Model

- TF-IDF features (word n-grams)
- Logistic Regression (scikit-learn)
- Class-weighted to handle imbalance
- CPU-only inference

---

## Deployment

- Gradio-based user interface
- Hosted on Hugging Face Spaces
- Model artifacts loaded at runtime
- No GPU required

---

## Notes

This demo is intended for **educational purposes** and should not be used as a standalone moderation system.

---

## License

MIT