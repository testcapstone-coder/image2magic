# Magic Story Maker

**Magic Story Maker** is a Streamlit storytelling application developed for the ISOM5240 individual assignment. A user uploads an image, the application analyzes what is visible, turns those details into a short child-friendly story, and generates spoken narration.

The application is designed for children aged **3–10** and produces stories between **50 and 100 words**. It uses pretrained Hugging Face models for image understanding and story generation, plus Kokoro for text-to-speech narration.

## Features

- **Image upload** — accepts JPG, JPEG, and PNG images up to 20 MB.
- **Detailed image understanding** — extracts the important subjects, setting, objects, colors, and visual details from the uploaded picture.
- **Child-friendly story generation** — creates a short first-person story grounded in the image description.
- **Automatic validation and retry** — checks story length and completeness, then asks the language model to revise the story when needed.
- **Text-to-speech narration** — converts the final story to WAV audio.
- **Multiple narrator voices** — supports American and British Kokoro voices.
- **Voice-only regeneration** — changing the narrator regenerates the audio without rewriting the story.
- **Interactive Streamlit UI** — provides image preview, progress feedback, story display, narration playback, and downloadable story/audio files.
- **CPU-oriented deployment** — models are loaded stage-by-stage and released where possible to reduce memory pressure on Streamlit Cloud.

## End-to-End Workflow

The application separates the task into three AI stages: **image description**, **story generation**, and **speech generation**.

```text
+------------------+
| 1. Upload Image  |
| JPG / JPEG / PNG |
+--------+---------+
         |
         v
+---------------------------+
| Validate and prepare image|
| - Size check              |
| - EXIF orientation        |
| - Convert to RGB          |
+-------------+-------------+
              |
              v
+--------------------------------------+
| 2. Image Understanding               |
| microsoft/Florence-2-base            |
| Task: <MORE_DETAILED_CAPTION>        |
+------------------+-------------------+
                   |
                   v
        Detailed image description
                   |
                   v
+--------------------------------------+
| 3. Story Generation                  |
| HuggingFaceTB/SmolLM2-360M-Instruct  |
| Hugging Face text-generation pipeline|
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| Validate story                       |
| - 50-100 words                       |
| - Complete ending                    |
| - No AI/instruction references       |
+------------------+-------------------+
          | valid              | invalid
          |                    |
          |                    +----> revise and retry
          v
+--------------------------------------+
| 4. Text-to-Speech                    |
| hexgrad/Kokoro-82M                   |
| + spaCy English text processing      |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| 5. Streamlit Output                  |
| Story + WAV narration + downloads    |
+--------------------------------------+
```

## Models Used

| Stage | Model / Component | Purpose | How it is used |
|---|---|---|---|
| Image understanding | `microsoft/Florence-2-base` | Produces a detailed factual description of the uploaded image | Loaded with `AutoProcessor` and `AutoModelForCausalLM`; the application uses Florence's `<MORE_DETAILED_CAPTION>` task token |
| Story generation | `HuggingFaceTB/SmolLM2-360M-Instruct` | Converts the factual image description into a warm children's story | Used through the Hugging Face `text-generation` pipeline with a system prompt, one short example, and validation/revision logic |
| Speech generation | `hexgrad/Kokoro-82M` | Converts the validated story into natural speech | Used through `KModel` and `KPipeline` to generate audio fragments that are combined into one WAV file |
| TTS text processing | `en_core_web_sm` | Processes English text required by Kokoro | Installed as a deployment dependency and loaded by the Kokoro pipeline |

### Why three separate models?

Each model performs one specialized task:

```text
Picture                 Text grounding              Creative text              Audio
  |                           |                          |                       |
  v                           v                          v                       v
Florence-2  ---------->  Image description  ---->  SmolLM2 story  ---->  Kokoro narration
```

Keeping image understanding separate from story generation makes the story easier to ground in visible image details. The generated story is then validated before it is passed to the speech model.

## Story Generation Rules

The application instructs SmolLM2 to produce a story that:

- is written in simple English for children aged 3–10;
- contains **50–100 words**;
- uses five short sentences where possible;
- has a beginning, a gentle adventure, and a happy ending;
- uses the main visible subject as the first-person narrator;
- preserves important image details such as setting, objects, colors, and scale;
- avoids adding people that are not present in the image description;
- avoids frightening, violent, unsafe, adult, or otherwise inappropriate content.

After generation, the code validates the result. If the story is too short, too long, incomplete, or contains unwanted AI/instruction references, it provides a targeted correction prompt and retries up to three times.

## Application Architecture

```text
                         +----------------------+
                         |     Streamlit UI     |
                         +----------+-----------+
                                    |
                     upload / voice | selection
                                    v
+----------------+       +----------------------+       +----------------+
| Session State  |<----->|  Application Logic   |<----->| Inference Lock |
| image_id       |       | validation / stages  |       | one model at a |
| result         |       +----------+-----------+       | time on CPU    |
+----------------+                  |                   +----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
     +-------------+         +--------------+         +-------------+
     | Florence-2  |         |   SmolLM2    |         |   Kokoro    |
     | description |         | story text   |         | narration   |
     +-------------+         +--------------+         +-------------+
```

The `result` object in Streamlit session state is updated progressively. If narration fails after the story has already been generated, the completed description and story remain available to the user.

## Main Processing Steps in `app.py`

1. **Configure models and constraints** — defines model IDs, story length limits, audio settings, supported image types, and voices.
2. **Load and prepare the uploaded image** — validates file size, fixes EXIF orientation, and converts the image to RGB.
3. **Generate a detailed image description** — Florence-2 analyzes the image using the `<MORE_DETAILED_CAPTION>` task.
4. **Generate the children's story** — SmolLM2 receives the description together with the story-writing rules.
5. **Validate and revise the story** — checks word count and output quality, then retries if necessary.
6. **Generate narration** — Kokoro produces one or more audio chunks, which are combined and written to an in-memory WAV file.
7. **Render the result** — Streamlit displays the story, word count, narration, detailed image description, and download actions.
8. **Handle voice changes** — if the user selects another narrator, only the audio stage runs again.

## Project Files

```text
.
├── app.py            # Streamlit application and AI processing logic
├── requirements.txt  # Python dependencies used by Streamlit Cloud
├── packages.txt      # Linux system packages required by audio/TTS libraries
├── README.md         # Project documentation
└── LICENSE           # GNU GPL v3.0 license text, if included in the repository
```

## Local Installation

Python **3.11 or 3.12** is recommended to match the deployment configuration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux, the system packages listed in `packages.txt` are also required:

```text
espeak-ng
libsndfile1
```

Run the application with:

```bash
streamlit run app.py
```

The first execution can take longer because the pretrained model files need to be downloaded and loaded.

## Streamlit Cloud Deployment

1. Push `app.py`, `requirements.txt`, `packages.txt`, `README.md`, and the license file to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Select `app.py` as the application entry point.
4. Use Python 3.11 or 3.12 if the deployment environment allows version selection.
5. Deploy the application and wait while Streamlit installs both Python and Linux dependencies.
6. Test the complete flow with several images: upload, image description, story creation, narration, voice change, and downloads.

### Deployment considerations

The application is configured for CPU execution. `run_stage()` performs garbage collection after each inference stage and attempts to return unused allocator memory to the operating system on Linux. An inference lock also prevents several large model stages from running simultaneously on the same small host.

## Key Dependencies

- `streamlit` — web interface and session state
- `transformers` — Florence-2 loading and Hugging Face text-generation pipeline
- `torch` / `torchvision` — model inference
- `kokoro` — text-to-speech pipeline
- `spacy` + `en_core_web_sm` — Kokoro English text processing
- `Pillow` — image loading and EXIF orientation handling
- `numpy` — audio array processing
- `soundfile` — WAV encoding
- `sentencepiece`, `timm`, `einops` — supporting model dependencies

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You are free to use, modify, and redistribute this software under the terms of the GNU GPL v3.0. Any redistributed or modified version must remain under a compatible open-source license and make the corresponding source code available.

See the `LICENSE` file for the complete license text.

## Third-Party Models

The pretrained models remain subject to their own licenses:

- `microsoft/Florence-2-base` — MIT License
- `HuggingFaceTB/SmolLM2-360M-Instruct` — Apache License 2.0
- `hexgrad/Kokoro-82M` — Apache License 2.0

These third-party models are not relicensed by this project's GPL-3.0 license. Their original licensing terms continue to apply.
