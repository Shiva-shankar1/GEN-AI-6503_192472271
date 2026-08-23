import streamlit as st
from transformers import pipeline


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🔬 AI Research Assistance System")

st.write(
    "Enter a research topic and generate relevant "
    "information, keywords, summary and research ideas "
    "using a pre-trained AI model."
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    return generator


# =====================================================
# RESEARCH TOPIC
# =====================================================

topic = st.text_input(
    "🔎 Enter Research Topic",
    placeholder=(
        "Example: Artificial Intelligence in "
        "Healthcare"
    )
)


# =====================================================
# RESEARCH AREA
# =====================================================

research_area = st.selectbox(
    "Select Research Area",
    [
        "Computer Science",
        "Artificial Intelligence",
        "Machine Learning",
        "Data Science",
        "Cybersecurity",
        "Embedded Systems",
        "Electrical Engineering",
        "Mechanical Engineering",
        "Civil Engineering"
    ]
)


# =====================================================
# GENERATE RESEARCH INFORMATION
# =====================================================

if st.button("🔬 Generate Research Assistance"):

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        with st.spinner(
            "Generating research information..."
        ):

            try:

                generator = load_model()


                # -------------------------------------
                # Information prompt
                # -------------------------------------

                information_prompt = (
                    "Research topic: "
                    + topic
                    + "\nResearch area: "
                    + research_area
                    + "\n\nProvide a concise explanation "
                    "of this research topic:"
                )


                information_result = generator(
                    information_prompt,
                    max_new_tokens=120,
                    do_sample=False
                )


                information = (
                    information_result[0]
                    ["generated_text"]
                )


                # -------------------------------------
                # Keywords
                # -------------------------------------

                keyword_prompt = (
                    "Research topic: "
                    + topic
                    + "\n\nList important research "
                    "keywords related to this topic:"
                )


                keyword_result = generator(
                    keyword_prompt,
                    max_new_tokens=60,
                    do_sample=False
                )


                keywords = (
                    keyword_result[0]
                    ["generated_text"]
                )


                # -------------------------------------
                # Summary
                # -------------------------------------

                summary_prompt = (
                    "Research topic: "
                    + topic
                    + "\n\nWrite a short research "
                    "summary about this topic:"
                )


                summary_result = generator(
                    summary_prompt,
                    max_new_tokens=100,
                    do_sample=False
                )


                summary = (
                    summary_result[0]
                    ["generated_text"]
                )


                # -------------------------------------
                # Research directions
                # -------------------------------------

                direction_prompt = (
                    "Research topic: "
                    + topic
                    + "\n\nSuggest possible future "
                    "research directions:"
                )


                direction_result = generator(
                    direction_prompt,
                    max_new_tokens=100,
                    do_sample=False
                )


                directions = (
                    direction_result[0]
                    ["generated_text"]
                )


                # =====================================
                # DISPLAY RESULTS
                # =====================================

                st.success(
                    "Research assistance generated!"
                )


                # -------------------------------------
                # Information
                # -------------------------------------

                st.subheader(
                    "📚 Relevant Information"
                )

                st.write(
                    information
                )


                # -------------------------------------
                # Keywords
                # -------------------------------------

                st.subheader(
                    "🔑 Research Keywords"
                )

                st.write(
                    keywords
                )


                # -------------------------------------
                # Summary
                # -------------------------------------

                st.subheader(
                    "📝 Concise Summary"
                )

                st.write(
                    summary
                )


                # -------------------------------------
                # Future Research
                # -------------------------------------

                st.subheader(
                    "🚀 Possible Research Directions"
                )

                st.write(
                    directions
                )


            except Exception as e:

                st.error(
                    "Research generation failed."
                )

                st.code(
                    str(e)
                )


# =====================================================
# EXAMPLE TOPICS
# =====================================================

st.divider()

st.subheader(
    "💡 Example Research Topics"
)

st.write(
    "• Artificial Intelligence in Healthcare"
)

st.write(
    "• Machine Learning for Cybersecurity"
)

st.write(
    "• IoT-Based Smart Agriculture"
)

st.write(
    "• Autonomous Drone Systems"
)

st.write(
    "• Generative AI in Education"
)

st.write(
    "• Blockchain-Based Security"
)


# =====================================================
# INFORMATION
# =====================================================

st.divider()

st.info(
    "Note: AI-generated research information should "
    "be verified using reliable academic sources before "
    "being used in a research paper."
)