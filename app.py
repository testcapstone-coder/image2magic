"""ISOM5240: picture-to-story application with Kokoro narration."""
import hashlib
import io
import re

import numpy as np
import soundfile as sf
import spacy
import streamlit as st
from PIL import Image, ImageOps
from kokoro import KPipeline
from transformers import pipeline

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-base"
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


def story_issue(story: str) -> str:
    """Check basic output quality; this is not a semantic or safety evaluator."""
    words = story.split()
    if not 50 <= len(words) <= 100:
        return f"Use 50–100 words; the previous attempt had {len(words)}."
    if not story.rstrip().endswith((".", "!", "?", '"', "”")):
        return "Finish with a complete sentence and a happy ending."
    tokens = re.findall(r"\b\w+\b", story.lower())
    phrases = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    if len(phrases) - len(set(phrases)) > 2:
        return "Avoid repeating phrases; make each sentence advance the adventure."
    return ""


def generate_story(caption: str) -> str:
    """Revise drafts toward the target; preserve a usable draft if checks fail."""
    prompt = (
        "Write a cheerful children's story inspired by the picture description below. "
        "Aim for 75 words, between 50 and 100 words total. Use simple English for ages 3–10. "
        "Give the main character a name and a small goal. Describe a playful problem, "
        "then show how the character solves it with kindness or curiosity. "
        "End happily. Keep the main subject and setting from the picture. "
        "Invent events, not a list of things in the picture. "
        "No danger, violence, scary events, title, instructions, or moral label. "
        "Return only one paragraph of story text.\n"
        f"Picture description: {caption}\nStory:"
    )
    model = load_story_model()
    feedback = ""
    candidates = []
    request = prompt
    for _ in range(3):
        story = model(
            request, max_new_tokens=220,
            do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
            repetition_penalty=1.1, no_repeat_ngram_size=4,
        )[0]["generated_text"].strip()
        if not story:
            request = prompt
            continue
        candidates.append(story)
        feedback = story_issue(story)
        if not feedback:
            return story
        count = len(story.split())
        action = "Expand" if count < 50 else "Shorten" if count > 100 else "Revise"
        request = (
            f"{action} this children's story into one complete paragraph of 50–100 words. "
            "Aim for 75 words. Keep the same characters and setting, use simple English, "
            "and finish happily. Add concrete events when expanding. "
            "Return only the rewritten story.\n"
            f"Picture: {caption}\nDraft: {story}\nCorrection: {feedback}\nRewritten story:"
        )
    if not candidates:
        raise RuntimeError("The story model returned no text. Please try again.")
    # Prefer the draft closest to the assignment length. Never silently pad or cut it.
    return min(candidates, key=lambda draft: (
        max(50 - len(draft.split()), len(draft.split()) - 100, 0),
        bool(story_issue(draft)),
        abs(len(draft.split()) - 75),
    ))


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
            except (OSError, ValueError, Image.DecompressionBombError):
                st.error("This picture could not be opened. Please upload another JPG or PNG.")
        create_clicked = st.button("✨ Create My Story", type="primary", disabled=image is None)
        if image is not None:
            st.image(image, width="stretch")

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
            issue = story_issue(result["story"])
            if issue:
                st.warning(
                    "This is the best of three drafts, but it still needs revision. "
                    + issue + " You can create another story."
                )
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
