import io
import re

import streamlit as st
from PIL import Image
from gtts import gTTS
from transformers import pipeline


# -------------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="📚",
    layout="centered",
)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"

MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    """Load and cache the Hugging Face image-captioning pipeline."""
    return pipeline(
        task="image-to-text",
        model=CAPTION_MODEL,
        device=-1,  # CPU; suitable for Streamlit Cloud
    )


@st.cache_resource(show_spinner=False)
def load_story_model():
    """Load and cache the Hugging Face text-generation pipeline."""
    return pipeline(
        task="text2text-generation",
        model=STORY_MODEL,
        device=-1,  # CPU; suitable for Streamlit Cloud
    )


# -------------------------------------------------------------------
# Core application functions
# -------------------------------------------------------------------
def generate_caption(image: Image.Image) -> str:
    """
    Generate a short caption that describes the uploaded image.

    Args:
        image: A PIL image uploaded by the user.

    Returns:
        A text caption describing the image.
    """
    captioner = load_caption_model()

    # BLIP expects a standard RGB image.
    image = image.convert("RGB")

    result = captioner(image, max_new_tokens=50)

    if not result:
        raise RuntimeError("The image-captioning model returned no result.")

    return result[0]["generated_text"].strip()


def clean_text(text: str) -> str:
    """Remove unnecessary spaces and normalize generated text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Remove quotation marks when the whole answer is wrapped in quotes.
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
    """
    Ask the model to rewrite a draft when it falls outside the required
    50-100 word range.
    """
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

    return clean_text(result[0]["generated_text"])


def shorten_to_word_limit(story: str, max_words: int = 100) -> str:
    """
    Safely shorten a story if the model still returns more than 100 words.

    The function prefers ending at a sentence boundary when possible.
    """
    words = story.split()

    if len(words) <= max_words:
        return story

    shortened = " ".join(words[:max_words])

    # Prefer the final full sentence inside the first max_words words.
    sentence_endings = [
        shortened.rfind("."),
        shortened.rfind("!"),
        shortened.rfind("?"),
    ]
    last_ending = max(sentence_endings)

    if last_ending >= 40:
        shortened = shortened[: last_ending + 1]
    else:
        shortened = shortened.rstrip(" ,;:-") + "."

    return shortened


def generate_story(caption: str) -> str:
    """
    Expand an image caption into a 50-100 word children's story.

    The Hugging Face model is prompted for 60-85 words to provide a buffer
    inside the assignment's required 50-100 word range.
    """
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

    # Give the model another opportunity to meet the required length.
    for _ in range(2):
        word_count = count_words(story)
        if MIN_STORY_WORDS <= word_count <= MAX_STORY_WORDS:
            break
        story = rewrite_story_to_length(story, caption)

    # Final guard for overly long output.
    if count_words(story) > MAX_STORY_WORDS:
        story = shorten_to_word_limit(story, MAX_STORY_WORDS)

    return clean_text(story)


def generate_audio(story: str) -> io.BytesIO:
    """
    Convert the generated story to MP3 audio using Google Text-to-Speech.

    Returns:
        A BytesIO object that Streamlit can play directly.
    """
    audio_buffer = io.BytesIO()

    speech = gTTS(
        text=story,
        lang="en",
        slow=False,
    )
    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer


def reset_story_state():
    """Clear generated outputs when a new image is uploaded."""
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
            max-width: 820px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            text-align: center;
            padding: 1.2rem 1rem 1.5rem 1rem;
        }

        .hero h1 {
            margin-bottom: 0.35rem;
        }

        .story-box {
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 0.75rem;
            margin-bottom: 1rem;
            font-size: 1.08rem;
            line-height: 1.65;
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
            <p>Upload a picture and turn it into a short, friendly story with audio. ✨</p>
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
        st.error("I could not read that image. Please upload a valid JPG, JPEG, or PNG file.")
        return

    st.image(
        image,
        caption="Your picture",
        use_container_width=True,
    )

    if st.button(
        "✨ Create My Story",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("🔎 Looking at your picture..."):
                caption = generate_caption(image)

            with st.spinner("✍️ Writing your story..."):
                story = generate_story(caption)

            with st.spinner("🔊 Creating the audio..."):
                audio = generate_audio(story)

            st.session_state.caption = caption
            st.session_state.story = story
            st.session_state.audio = audio.getvalue()

        except Exception as error:
            st.error(
                "Something went wrong while creating the story. "
                "Please try again in a moment."
            )
            # Useful while developing without exposing a large traceback to users.
            st.caption(f"Technical detail: {error}")
            return

    if "story" in st.session_state:
        with st.expander("🔎 What the AI saw in the picture"):
            st.write(st.session_state.caption)

        st.subheader("✨ Your Story")

        st.markdown(
            f'<div class="story-box">{st.session_state.story}</div>',
            unsafe_allow_html=True,
        )

        word_count = count_words(st.session_state.story)

        if MIN_STORY_WORDS <= word_count <= MAX_STORY_WORDS:
            st.caption(f"✅ Story length: {word_count} words")
        else:
            st.caption(
                f"⚠️ Story length: {word_count} words "
                f"(target: {MIN_STORY_WORDS}-{MAX_STORY_WORDS})"
            )

        st.subheader("🔊 Listen to Your Story")
        st.audio(
            st.session_state.audio,
            format="audio/mp3",
        )

        st.download_button(
            label="💾 Save the story as a text file",
            data=st.session_state.story,
            file_name="my_magic_story.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown(
            """
            <p class="small-note">
            This app uses Hugging Face models for image captioning and story generation,
            followed by text-to-speech for audio playback.
            </p>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

