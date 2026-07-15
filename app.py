import gdown
import os

@st.cache_resource
def load_model():
    # Download model from Google Drive if not already present
    if not os.path.exists("combined_model.pth"):
        url = "https://drive.google.com/file/d/1j8f1ayqi0SY_iPk1v8Bnsdlh0yv159Yu/view?usp=sharing"
        gdown.download(url, "combined_model.pth", quiet=False)

    def build_model(num_classes=2):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, num_classes)
        )
        return model

    model = build_model()
    model.load_state_dict(torch.load("combined_model.pth",
                          map_location=torch.device("cpu")))
    model.eval()
    return model
