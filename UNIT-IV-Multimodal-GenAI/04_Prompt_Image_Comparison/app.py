import streamlit as st
from diffusers import DiffusionPipeline
import torch

st.set_page_config(
    page_title="Prompt Image Comparison",
    page_icon="🖼️"
)

st.title("🖼️ Prompt-Based Image Comparison")

st.write(
    "Generate images using different prompts "
    "and compare how prompt changes affect the result."
)


@st.cache_resource
def load_model():

    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd",
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe


prompt1 = st.text_area(
    "Prompt 1",
    "A suspension bridge over a river"
)

prompt2 = st.text_area(
    "Prompt 2",
    "A futuristic suspension bridge over a river at sunset"
)

prompt3 = st.text_area(
    "Prompt 3",
    "A futuristic suspension bridge over a river at sunset, surrounded by skyscrapers, realistic engineering design"
)


if st.button("🎨 Generate and Compare"):

    with st.spinner(
        "Loading model and generating images..."
    ):

        try:

            pipe = load_model()

            image1 = pipe(
                prompt1,
                num_inference_steps=15
            ).images[0]

            image2 = pipe(
                prompt2,
                num_inference_steps=15
            ).images[0]

            image3 = pipe(
                prompt3,
                num_inference_steps=15
            ).images[0]


            st.success(
                "All three images generated!"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.subheader("Prompt 1")

                st.image(
                    image1,
                    caption=prompt1,
                    use_container_width=True
                )


            with col2:

                st.subheader("Prompt 2")

                st.image(
                    image2,
                    caption=prompt2,
                    use_container_width=True
                )


            with col3:

                st.subheader("Prompt 3")

                st.image(
                    image3,
                    caption=prompt3,
                    use_container_width=True
                )


            st.subheader(
                "📊 Prompt Comparison"
            )

            st.write(
                "**Prompt 1:** Basic description"
            )

            st.write(
                "**Prompt 2:** Added futuristic "
                "and sunset details"
            )

            st.write(
                "**Prompt 3:** Added environment, "
                "buildings, realism, and engineering details"
            )


        except Exception as e:

            st.error(
                "Image generation failed."
            )

            st.code(str(e))