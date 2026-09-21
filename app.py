import io
import re

import numpy as np
import soundfile as sf
import streamlit as st
from kokoro import KPipeline
from PIL import Image
from transformers import pipeline


# -------------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="📚",
    layout="wide",
)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"

MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100

VOICE_OPTIONS = {
    "🧚 Bella — Warm American": "af_bella",
    "💖 Heart — Friendly American": "af_heart",
    "🇬🇧 Emma — British": "bf_emma",
    "🧙 Michael — American Male": "am_michael",
}


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    """Load and cache the Hugging Face image-captioning pipeline."""
    return pipeline(
        task="image-to-text",
        model=CAPTION_MODEL,
        device=-1,  # CPU, suitable for Streamlit Cloud
    )


@st.cache_resource(show_spinner=False)
def load_story_model():
    """Load and cache the Hugging Face story-generation pipeline."""
    return pipeline(
        task="text2text-generation",
        model=STORY_MODEL,
        device=-1,  # CPU, suitable for Streamlit Cloud
    )


@st.cache_resource(show_spinner=False)
def load_tts_model():
    """Load and cache the Kokoro text-to-speech pipeline."""
    return KPipeline(lang_code="a")  # American English phonemization


# -------------------------------------------------------------------
# Core application functions
# -------------------------------------------------------------------
def generate_caption(image: Image.Image) -> str:
    """
    Generate a short description of the uploaded image.

    Args:
        image: PIL image uploaded by the user.

    Returns:
        A text caption describing the image.
    """
    captioner = load_caption_model()
    image = image.convert("RGB")

    result = captioner(
        image,
        max_new_tokens=50,
    )

    if not result:
        raise RuntimeError("The image-captioning model returned no result.")

    return result[0]["generated_text"].strip()


def clean_text(text: str) -> str:
    """Normalize spacing and remove unnecessary wrapping quotation marks."""
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()

    return text


def count_words(text: str) -> int:
    """Return the number of words in a string."""
    return len(text.split())


def build_story_prompt(caption: str) -> str:
    """Create the prompt used by the Hugging Face story-generation model."""
    return f"""
Write one complete children's story inspired by this image description:

"{caption}"

Requirements:
- The audience is children aged 3 to 10.
- Write between 60 and 85 words.
- Use simple, warm, easy-to-understand English.
- Give the story a clear beginning, middle, and happy ending.
- Keep the story playful, positive, gentle, and age-appropriate.
- Avoid violence, frightening details, unsafe behavior, adult themes, or inappropriate language.
- Do not mention these instructions.
- Return only the story.

Story:
""".strip()


def rewrite_story_to_length(story: str, caption: str) -> str:
    """Rewrite a draft if it falls outside the required 50-100 word range."""
    story_generator = load_story_model()

    prompt = f"""
Rewrite the children's story below so that it is between 60 and 85 words.

Image description:
"{caption}"

Draft story:
"{story}"

Keep it suitable for children aged 3 to 10.
Use simple English, a positive tone, and a happy ending.
Return only the rewritten story.
""".strip()

    result = story_generator(
        prompt,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.08,
    )

    if not result:
        raise RuntimeError("The story-generation model returned no rewrite.")

    return clean_text(result[0]["generated_text"])


def shorten_to_word_limit(story: str, max_words: int = 100) -> str:
    """Shorten a story while trying to end on a complete sentence."""
    words = story.split()

    if len(words) <= max_words:
        return story

    shortened = " ".join(words[:max_words])

    last_ending = max(
        shortened.rfind("."),
        shortened.rfind("!"),
        shortened.rfind("?"),
    )

    if last_ending >= 40:
        shortened = shortened[: last_ending + 1]
    else:
        shortened = shortened.rstrip(" ,;:-") + "."

    return shortened


def generate_story(caption: str) -> str:
    """Expand the image caption into a 50-100 word children's story."""
    story_generator = load_story_model()
    prompt = build_story_prompt(caption)

    result = story_generator(
        prompt,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.85,
        top_p=0.95,
        repetition_penalty=1.08,
    )

    if not result:
        raise RuntimeError("The story-generation model returned no result.")

    story = clean_text(result[0]["generated_text"])

    # Give the model up to two extra attempts to satisfy the required length.
    for _ in range(2):
        word_count = count_words(story)

        if MIN_STORY_WORDS <= word_count <= MAX_STORY_WORDS:
            break

        story = rewrite_story_to_length(story, caption)

    if count_words(story) > MAX_STORY_WORDS:
        story = shorten_to_word_limit(story, MAX_STORY_WORDS)

    return clean_text(story)


def generate_audio(
    story: str,
    voice: str = "af_bella",
    speed: float = 0.95,
) -> io.BytesIO:
    """
    Convert the generated story into natural speech using Kokoro-82M.

    Args:
        story: Story text to narrate.
        voice: Kokoro voice ID.
        speed: Narration speed.

    Returns:
        WAV audio stored in memory.
    """
    tts_pipeline = load_tts_model()

    generator = tts_pipeline(
        story,
        voice=voice,
        speed=speed,
    )

    audio_chunks = []

    for _, _, audio in generator:
        audio_chunks.append(np.asarray(audio))

    if not audio_chunks:
        raise RuntimeError("The text-to-speech model returned no audio.")

    combined_audio = np.concatenate(audio_chunks)

    audio_buffer = io.BytesIO()

    sf.write(
        audio_buffer,
        combined_audio,
        24000,
        format="WAV",
    )

    audio_buffer.seek(0)
    return audio_buffer


def reset_story_state():
    """Clear previous generated results when the uploaded image changes."""
    for key in ("caption", "story", "audio"):
        st.session_state.pop(key, None)


# -------------------------------------------------------------------
# Streamlit user interface
# -------------------------------------------------------------------
def main():
    """Run the Streamlit storytelling application."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1250px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            text-align: center;
            padding: 0.6rem 1rem 1.3rem 1rem;
        }

        .hero h1 {
            margin-bottom: 0.3rem;
        }

        .panel-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .small-note {
            opacity: 0.75;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>📚 Magic Story Maker</h1>
            <p>Upload a picture and turn it into a short, friendly story with natural audio. ✨</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a picture",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image.",
        on_change=reset_story_state,
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to begin.")
        return

    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error(
            "I could not read that image. "
            "Please upload a valid JPG, JPEG, or PNG file."
        )
        return

    # ---------------------------------------------------------------
    # Two-column application layout:
    # Image on the left, generated content on the right.
    # ---------------------------------------------------------------
    image_column, story_column = st.columns(
        [1, 1],
        gap="large",
        vertical_alignment="top",
    )

    with image_column:
        st.markdown('<div class="panel-title">🖼 Your Picture</div>', unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True,
        )

        selected_voice_name = st.selectbox(
            "🎙 Choose your storyteller",
            options=list(VOICE_OPTIONS.keys()),
            index=0,
        )

        selected_voice = VOICE_OPTIONS[selected_voice_name]

        narration_speed = st.slider(
            "Narration speed",
            min_value=0.80,
            max_value=1.10,
            value=0.95,
            step=0.05,
            help="A slightly slower speed is easier for younger children to follow.",
        )

        create_story = st.button(
            "✨ Create My Story",
            type="primary",
            use_container_width=True,
        )

    with story_column:
        st.markdown('<div class="panel-title">✨ Your Story</div>', unsafe_allow_html=True)

        if create_story:
            try:
                with st.spinner("🔎 Looking at your picture..."):
                    caption = generate_caption(image)

                with st.spinner("✍️ Writing your story..."):
                    story = generate_story(caption)

                with st.spinner("🔊 Creating the narration..."):
                    audio = generate_audio(
                        story,
                        voice=selected_voice,
                        speed=narration_speed,
                    )

                st.session_state.caption = caption
                st.session_state.story = story
                st.session_state.audio = audio.getvalue()

            except Exception as error:
                st.error(
                    "Something went wrong while creating the story. "
                    "Please try again."
                )
                st.caption(f"Technical detail: {error}")

        if "story" not in st.session_state:
            with st.container(border=True):
                st.markdown("### 🌟 Your story will appear here")
                st.write(
                    "Choose a storyteller on the left, then click "
                    "**Create My Story**."
                )
        else:
            with st.expander("🔎 What the AI saw in the picture"):
                st.write(st.session_state.caption)

            with st.container(border=True):
                st.write(st.session_state.story)

            word_count = count_words(st.session_state.story)

            if MIN_STORY_WORDS <= word_count <= MAX_STORY_WORDS:
                st.caption(f"✅ Story length: {word_count} words")
            else:
                st.caption(
                    f"⚠️ Story length: {word_count} words "
                    f"(target: {MIN_STORY_WORDS}-{MAX_STORY_WORDS})"
                )

            st.markdown("#### 🔊 Listen to Your Story")

            st.audio(
                st.session_state.audio,
                format="audio/wav",
            )

            st.download_button(
                label="💾 Save the story as a text file",
                data=st.session_state.story,
                file_name="my_magic_story.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    st.markdown(
        """
        <p class="small-note">
        The app uses Hugging Face models for image captioning and story generation,
        followed by Kokoro text-to-speech for natural narration.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
