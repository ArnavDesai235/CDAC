import streamlit as st
import os
from pathlib import Path
import pandas as pd
import RAG
from config import VIDEOS_FOLDER, TRANSCRIPTS_FOLDER

def custom_video_player(video_path, width=720, height=405, key="video_player", start_time=0):
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return None

    player_key = f"{key}_{Path(video_path).stem}"
    st.video(video_path, start_time=start_time)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        time_input = st.number_input(
            "Seek to time (seconds):",
            min_value=0.0,
            value=float(start_time),
            step=1.0,
            key=f"time_input_{player_key}"
        )
    with col2:
        if st.button("Seek", key=f"seek_button_{player_key}"):
            return time_input
    with col3:
        if st.button("Reload Video", key=f"reload_{player_key}"):
            st.rerun()

    return None

def main():
    st.title("Video Player with RAG Question Answering")
    st.sidebar.header("Video Selection")

    video_files = list(VIDEOS_FOLDER.glob("*.mp4")) + list(VIDEOS_FOLDER.glob("*.avi")) + list(VIDEOS_FOLDER.glob("*.mov"))
    if not video_files:
        st.warning(f"No video files found in {VIDEOS_FOLDER}. Please add some videos.")
        return

    video_options = [file.name for file in video_files]
    selected_video = st.sidebar.selectbox("Select a video to play", video_options)
    video_base_name = Path(selected_video).stem
    video_path = str(VIDEOS_FOLDER / selected_video)

    if 'current_video_time' not in st.session_state:
        st.session_state.current_video_time = 0

    seek_result = custom_video_player(
        video_path,
        width=720,
        height=405,
        key="main_player",
        start_time=st.session_state.current_video_time
    )
    if seek_result is not None:
        st.session_state.current_video_time = seek_result
        st.rerun()

    transcript_path = TRANSCRIPTS_FOLDER / f"{video_base_name}.csv"

    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
        st.session_state.metadata = None
        st.session_state.model_name = None

    if transcript_path.is_file():
        if st.sidebar.button("Load Transcript Data"):
            with st.spinner("Loading transcript data and building search index..."):
                try:
                    prefix = str(transcript_path).replace('.csv', '')
                    index_path = f"{prefix}_index.faiss"
                    if Path(index_path).exists():
                        st.sidebar.info("Loading pre-saved index...")
                        st.session_state.faiss_index, st.session_state.metadata, st.session_state.model_name = RAG.load_data(prefix)
                    else:
                        st.sidebar.info("Building new index from transcript...")
                        embeddings, faiss_index, metadata, model_name = RAG.process_transcript_data(
                            str(transcript_path),
                            model_name='all-MiniLM-L6-v2',
                            use_gpu=False
                        )
                        RAG.save_data(faiss_index, metadata, embeddings, model_name, prefix=prefix)
                        st.session_state.faiss_index = faiss_index
                        st.session_state.metadata = metadata
                        st.session_state.model_name = model_name
                    st.sidebar.success("Transcript data loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error loading transcript data: {str(e)}")
    else:
        st.sidebar.warning(f"No transcript found for {selected_video}.")

    st.subheader("Ask a Question About the Video")
    with st.form(key="question_form"):
        question = st.text_area("Type your question about the video:")
        submit_button = st.form_submit_button(label="Submit Question")

    if submit_button:
        if question.strip():
            if st.session_state.faiss_index is not None:
                with st.spinner("Searching for relevant parts of the video..."):
                    try:
                        results = RAG.semantic_search(
                            question,
                            st.session_state.faiss_index,
                            st.session_state.metadata,
                            st.session_state.model_name,
                            top_k=3
                        )
                        st.subheader(f"Results for: {question}")
                        for i, result in enumerate(results):
                            score = 1 - result['distance']/2
                            with st.expander(f"Result {i+1} (Relevance: {score:.2f})"):
                                st.markdown(f"**Text:** {result['text']}")
                                st.markdown(f"**Timestamp:** {result['start']:.2f}s - {result['end']:.2f}s")
                                if st.button(f"Jump to {result['start']:.2f}s", key=f"jump_{i}"):
                                    st.session_state.current_video_time = result['start']
                                    st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
            else:
                st.warning("Please load the transcript data first.")
        else:
            st.error("Please enter a question before submitting.")

if __name__ == "__main__":
    main()
