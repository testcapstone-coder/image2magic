"""ISOM5240: picture-to-story application with Kokoro narration."""
import hashlib
import io

import numpy as np
import soundfile as sf
import spacy
import streamlit as st
from PIL import Image, ImageOps
from kokoro import KPipeline
from transformers import pipeline

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"
NARRATION_SPEED = 0.95
VOICE_OPTIONS = {
    "🧚 Bella — Warm American": "af_bella",
    "💖 Heart — Friendly American": "af_heart",
    "🇬🇧 Emma — British": "bf_emma",
    "🧙 Michael — American Male": "am_michael",
}


@st.cache_resource(show_spinner=False)
def load_caption_model():
    return pipeline("image-to-text", model=CAPTION_MODEL, device=-1)


@st.cache_resource(show_spinner=False)
def load_story_model():
    return pipeline("text2text-generation", model=STORY_MODEL, device=-1)


@st.cache_resource(show_spinner=False)
def load_tts_model(lang_code: str):
    # Prevent Misaki from attempting a forbidden runtime package installation.
    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "The English speech dependency en_core_web_sm is missing. "
            "Deploy the updated requirements.txt and wait for installation to finish."
        )
    # Match British voices with British pronunciation, American voices with American.
    return KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")


def generate_caption(image: Image.Image) -> str:
    caption = load_caption_model()(image, max_new_tokens=60)[0]["generated_text"].strip()
    if not caption:
        raise RuntimeError("The image model returned an empty description.")
    return caption


def generate_story(caption: str) -> str:
    """Ask FLAN-T5 for a short story, retrying if its length is outside the target."""
    prompt = (
        "Write a complete, cheerful story of 50 to 100 words for children aged 3 to 10. "
        "Use simple words, a beginning, a small adventure, and a happy ending. "
        "Avoid scary or inappropriate content. Return only the story. "
        f"The story should be inspired by this picture: {caption}"
    )
    model = load_story_model()
    for _ in range(3):
        story = model(
            prompt, max_new_tokens=180, min_new_tokens=65,
            do_sample=True, temperature=0.8, top_p=0.9,
            repetition_penalty=1.15,
        )[0]["generated_text"].strip()
        if 50 <= len(story.split()) <= 100:
            return story
    raise RuntimeError("The story model could not produce 50–100 words. Please try again.")


def generate_audio(story: str, voice: str = "af_bella") -> bytes:
    """Generate a 24 kHz WAV at the fixed internal narration speed."""
    tts = load_tts_model("b" if voice.startswith("b") else "a")
    chunks = []
    for _, _, audio in tts(story, voice=voice, speed=NARRATION_SPEED):
        if audio is not None:
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
            if chunk.size:
                chunks.append(chunk)
    if not chunks:
        raise RuntimeError("The narrator did not produce audio. Please try again.")
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), 24000, format="WAV")
    return buffer.getvalue()


def create_story(image: Image.Image, voice: str, progress) -> dict:
    """Advance progress only when the preceding stage has finished."""
    progress.progress(0, text="1 of 3 · Looking at your picture…")
    caption = generate_caption(image)
    progress.progress(33, text="2 of 3 · Writing your story…")
    story = generate_story(caption)
    progress.progress(66, text="3 of 3 · Recording your story…")
    audio = generate_audio(story, voice)
    progress.progress(100, text="100% · Your story and narration are ready!")
    return {"caption": caption, "story": story, "audio": audio, "voice": voice}


def main():
    st.set_page_config(page_title="Magic Story Maker", page_icon="📚", layout="wide")
    st.title("📚 Magic Story Maker")
    st.write("Turn your picture into a little adventure you can read and listen to!")
    left, right = st.columns([1, 1.3], gap="large")
    image = None
    with left:
        st.subheader("🖼 Your Picture")
        uploaded = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
        image_id = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None
        if st.session_state.get("image_id") != image_id:
            st.session_state["image_id"] = image_id
            st.session_state.pop("result", None)
        if uploaded is not None:
            try:
                with Image.open(io.BytesIO(uploaded.getvalue())) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                st.image(image, width="stretch")
            except (OSError, ValueError, Image.DecompressionBombError):
                st.error("This picture could not be opened. Please upload another JPG or PNG.")
        create_clicked = st.button("✨ Create My Story", type="primary", disabled=image is None)

    with right:
        selected_name = st.selectbox("🎙 Choose your storyteller", list(VOICE_OPTIONS))
        st.subheader("✨ Your Story")
        selected_voice = VOICE_OPTIONS[selected_name]
        if create_clicked and image is not None:
            st.session_state.pop("result", None)
            progress = st.progress(0, text="Getting ready…")
            try:
                with st.spinner("Creating your story… The first run may take longer while models load."):
                    st.session_state["result"] = create_story(image, selected_voice, progress)
            except Exception as exc:
                progress.empty()
                st.error("We couldn't finish your story. Please try again.")
                with st.expander("Technical details"):
                    st.text(str(exc))
        result = st.session_state.get("result")
        if result:
            st.write(result["story"])
            st.caption(f"{len(result['story'].split())} words")
            with st.expander("What was in your picture?"):
                st.write(result["caption"])
            st.subheader("🔊 Listen to your story")
            narrator = next(name for name, voice in VOICE_OPTIONS.items() if voice == result["voice"])
            st.caption(f"Narrated by {narrator}")
            if selected_voice != result["voice"]:
                st.info("Click Create My Story to make a new story with your chosen storyteller.")
            st.audio(result["audio"], format="audio/wav")
            st.download_button("⬇ Download narration", result["audio"], "my-story.wav", "audio/wav")
            st.download_button("⬇ Download story", result["story"], "my-story.txt", "text/plain")
        else:
            st.info("Choose a picture and a storyteller, then click Create My Story.")


if __name__ == "__main__":
    main()
