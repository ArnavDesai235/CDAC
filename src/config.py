from pathlib import Path

# Base data directory
BASE_DIR = Path(__file__).resolve().parent.parent / "data"

# Input videos folder
VIDEOS_FOLDER = BASE_DIR / "videos"

# Transcripts folder
TRANSCRIPTS_FOLDER = BASE_DIR / "transcripts"

# Temporary WAV audio storage
WAV_FOLDER = BASE_DIR / "wav"

# Ensure folders exist
for folder in [VIDEOS_FOLDER, TRANSCRIPTS_FOLDER, WAV_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)
