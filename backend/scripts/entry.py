import os
from pyannote.audio import Model, Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics

if __name__ == "__main__":
    # disable metrics globally
    set_telemetry_metrics(False, save_choice_as_default=True)

    # Get HF Token
    HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    print(HF_TOKEN)
    if not HF_TOKEN:
        raise RuntimeError("HUGGING_FACE_TOKEN environment variable is required.")

    print("Loading models from Hugging Face...")

    # Load Speaker Diarization Pipeline
    print("  → Loading pyannote/speaker-diarization-3.1...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

    # Load Segmentation Model
    print("  → Loading pyannote/segmentation-3.0...")
    segmentation_model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        token=HF_TOKEN
    )

    # Load Embedding Model
    print("  → Loading pyannote/embedding...")
    embedding_model = Model.from_pretrained(
        "pyannote/embedding",
        token=HF_TOKEN
    )

    print("All models cached in ~/.cache/huggingface!")
