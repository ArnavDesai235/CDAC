# **Video RAG QA System 🎥🔍**

A **Retrieval-Augmented Generation (RAG)** powered system for **semantic search** and **question answering** over video content.  

This project:
1. **Extracts audio** from videos
2. **Transcribes speech** to text using [Whisper](https://github.com/openai/whisper) via Hugging Face
3. **Embeds transcript segments** with [Sentence-BERT](https://www.sbert.net/)
4. Stores them in a **FAISS** index for fast retrieval
5. Provides a **Streamlit** interface to ask questions and jump to relevant timestamps.

---

## **📦 Features**
- 🎥 **Custom Video Player** with timestamp seeking
- 🗣 **Automatic Speech Recognition** using Whisper
- 🔍 **Semantic Search** over video transcripts
- ⚡ **Fast Retrieval** with FAISS
- 🖥 **Interactive Frontend** via Streamlit

---

## **📂 Project Structure**
```
video-rag-qa/
│
├── src/                         # Source code
│   ├── config.py                # Central path config
│   ├── RAG.py                   # Embedding + FAISS search
│   ├── VideoProcess.py          # Video → WAV extraction
│   ├── Transcription.py         # Whisper transcription
│   ├── Frontend.py              # Streamlit frontend
│   ├── main.py                  # Batch transcription runner
│
├── data/                        # Data storage (auto-created)
│   ├── videos/                  # Input videos
│   ├── transcripts/             # Transcripts output
│   ├── wav/                     # Temporary audio files
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## **🛠 Installation**
### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/video-rag-qa.git
cd video-rag-qa
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Install GPU FAISS
If you have a CUDA-capable GPU:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 4️⃣ Download NLTK resources
```bash
python -m nltk.downloader punkt stopwords wordnet
```

---

## **📸 Usage**

### 1. Add videos
Place your video files in:
```
data/videos/
```
Supported formats:
- `.mp4`
- `.avi`
- `.mov`

---

### 2. Extract audio + generate transcripts
Run:
```bash
python src/main.py
```
This will:
- Extract audio as `.wav` to `data/wav/`
- Transcribe with Whisper
- Save transcripts in `data/transcripts/` as `.csv`

---

### 3. Launch the frontend
```bash
streamlit run src/Frontend.py
```
This will open the app in your browser:
```
http://localhost:8501
```

From there you can:
- Select a video
- Load its transcript
- Ask questions
- Jump to relevant timestamps in the video

---

## **⚙ Configuration**
All file paths are defined in:
```
src/config.py
```
Default:
```python
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
VIDEOS_FOLDER = BASE_DIR / "videos"
TRANSCRIPTS_FOLDER = BASE_DIR / "transcripts"
WAV_FOLDER = BASE_DIR / "wav"
```

---

## **📈 How it works**
1. **Video Processing**  
   `VideoProcess.py` extracts audio from videos.

2. **Speech-to-Text**  
   `Transcription.py` uses Whisper for transcription.

3. **Embedding & Indexing**  
   `RAG.py` uses Sentence-BERT to embed transcript segments and FAISS for storage.

4. **Semantic Search**  
   When a user asks a question, we:
   - Encode the query
   - Retrieve top matching segments from FAISS
   - Display them in the frontend with timestamps

---

## **📋 Example Workflow**
```bash
# Step 1: Put "example.mp4" in data/videos/

# Step 2: Generate transcript
python src/main.py

# Step 3: Start app
streamlit run src/Frontend.py

# Step 4: Select "example.mp4" in sidebar, load transcript, ask:
"Who is speaking about the invention of the steam engine?"
```
The app will return relevant segments and allow you to jump directly to them.

---

## **🧩 Dependencies**
- `pandas` / `numpy`
- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-cpu` or `faiss-gpu`
- `nltk`
- `moviepy`
- `pydub`
- `librosa`
- `streamlit`

---

## **💡 Tips**
- Change the Whisper model in `main.py` for speed/accuracy tradeoff:
  ```python
  transcriber = AudioTranscriber(model_name="openai/whisper-small")
  ```
  Options: `"whisper-tiny"`, `"whisper-small"`, `"whisper-medium"`, `"whisper-large-v2"`

- For **faster search** on large datasets, enable GPU FAISS:
  ```python
  embeddings, faiss_index, metadata, model_name = RAG.process_transcript_data(..., use_gpu=True)
  ```

---

## **📜 License**
MIT License — free to use, modify, and distribute.
