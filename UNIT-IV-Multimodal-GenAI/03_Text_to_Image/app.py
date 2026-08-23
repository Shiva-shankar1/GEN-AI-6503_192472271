import streamlit as st
from diffusers import DiffusionPipeline
import torch

st.set_page_config(
    page_title="Engineering Text to Image",
    page_icon="🏗️"
)

st.title("🏗️ Engineering Text-to-Image Generator")

prompt = st.text_area(
    "Enter engineering image prompt",
    "A modern suspension bridge over a river, realistic engineering design"
)

if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:

        with st.spinner("Loading AI model and generating image..."):

            try:

                pipe = DiffusionPipeline.from_pretrained(
                    "segmind/tiny-sd",
                    torch_dtype=torch.float32
                )

                pipe = pipe.to("cpu")

                image = pipe(
                    prompt,
                    num_inference_steps=20
                ).images[0]

                st.success("Image generated!")

                st.image(
                    image,
                    caption=prompt,
                    use_container_width=True
                )

            except Exception as e:

                st.error("Image generation failed.")

                st.code(str(e))