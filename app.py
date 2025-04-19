import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from streamlit_drawable_canvas import st_canvas
import io
from utils.generator import UNetGenerator

IMAGE_SIZE = 256
NUM_CHANNELS = 3
ANOMALY_TYPES_FOR_TRAINING = ["crack", "glue_strip"] # MUST match training
DEFECT_MAP = {name: i for i, name in enumerate(ANOMALY_TYPES_FOR_TRAINING)}
NUM_CLASSES = len(DEFECT_MAP)
EMBED_SIZE = 16
NGF = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_CHECKPOINT_PATH = "./checkpoints_cgan_tile/cgan_generator_epoch_150.pth"


@st.cache_resource 
def load_model(path, device):
    try:
        model = UNetGenerator(num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES, embed_size=EMBED_SIZE, ngf=NGF)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.to(device)
        model.eval()
        print("Generator model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Generator checkpoint not found at {path}. Please ensure the path is correct.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

generator = load_model(GENERATOR_CHECKPOINT_PATH, DEVICE)

transform_image = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transform_mask = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

st.title("Tile Defect Generation Demo")

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Good Tile Image:", type=["png", "jpg", "jpeg", "tif"])
defect_type = st.sidebar.selectbox("Select Defect Type:", options=ANOMALY_TYPES_FOR_TRAINING)

drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "rect", "circle", "line", "transform"))
stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 5)
stroke_color = "#FFFFFF"
bg_color = "#000000"

image_pil = None
if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(image_pil, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.sidebar.error(f"Error opening image file: {e}")
        image_pil = None

st.subheader("Draw Mask on Image")
st.write("Draw the area where you want the defect to appear (in white).")

canvas_result = None
if image_pil:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image_pil.resize((IMAGE_SIZE, IMAGE_SIZE)),
        update_streamlit=True,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        drawing_mode=drawing_mode,
        key="canvas",
    )
else:
    st.warning("Please upload an image first.")

st.subheader("4. Generate Defective Image")
if st.button("Generate Anomaly"):
    if generator is None:
         st.error("Model not loaded. Cannot generate.")
    elif image_pil is None:
        st.warning("Please upload a valid image.")
    elif canvas_result is None or canvas_result.image_data is None:
        st.warning("Please draw a mask on the uploaded image.")
    else:
        with st.spinner("Generating..."):
            try:
                
                input_image_tensor = transform_image(image_pil).unsqueeze(0).to(DEVICE) 

                drawn_mask_np = canvas_result.image_data[:, :, 3] > 0
                mask_tensor = torch.from_numpy(drawn_mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask_tensor = mask_tensor.to(DEVICE)

                condition_label = DEFECT_MAP[defect_type]
                condition_tensor = torch.tensor([condition_label]).long().to(DEVICE)

                masked_input = input_image_tensor * (1.0 - mask_tensor) + (-1.0 * mask_tensor)

                with torch.no_grad():
                    generated_image = generator(masked_input, mask_tensor, condition_tensor)

                output_image = generated_image.squeeze(0).cpu()
                output_image_denorm = (output_image * 0.5) + 0.5
                output_image_denorm = torch.clamp(output_image_denorm, 0, 1)
                st.image(output_image_denorm.permute(1, 2, 0).numpy(), caption=f"Generated '{defect_type}'", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                import traceback
                st.text(traceback.format_exc())