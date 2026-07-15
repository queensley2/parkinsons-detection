import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_model():
    if not os.path.exists("combined_model.pth"):
        url = "https://drive.google.com/file/d/1j8f1ayqi0SY_iPk1v8Bnsdlh0yv159Yu/view?usp=sharing"
        gdown.download(url, "combined_model.pth", quiet=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(torch.load("combined_model.pth",
                          map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ── Predict Function ──────────────────────────────────────────
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ["Healthy", "Parkinson's"]
    score = confidence.item() * 100

    if score < 75:
        return "Uncertain", score
    else:
        label = classes[predicted.item()]
        return label, score

# ── UI ────────────────────────────────────────────────────────
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a handwriting sample (spiral or wave) to detect signs of Parkinson's Disease.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analysing..."):
        label, score = predict(image)

    st.markdown("---")

  if label == "Healthy":
        st.success(f"✅ Prediction: {label} — {score:.2f}%")
    elif label == "Parkinson's":
        st.warning(f"⚠️ Prediction: {label} — {score:.2f}%")
    else:
        st.info(f"❓ Prediction: Uncertain — {score:.2f}%")
        st.write("Confidence too low — recommend clinical review")

    st.metric(label="Confidence", value=f"{score:.2f}%")
