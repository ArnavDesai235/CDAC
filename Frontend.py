import streamlit as st
import os
from pathlib import Path
import pandas as pd
import RAG  # Import your RAG module
import streamlit.components.v1 as components

# Define paths (these should be relative or configurable in production)
transcripts_folder = Path("venv/CDAC transcripts")
videos_folder = Path("venv/classic-documentaries")

def custom_video_player(video_path, width=720, height=405, key="video_player", start_time=0):
    """
    Create a custom video player that allows seeking to specific timestamps
    using Streamlit's native video component but with custom controls.
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return None

    # Create a unique key for this instance
    player_key = f"{key}_{Path(video_path).stem}"

    # Display the video using Streamlit's built-in video component
    st.video(video_path, start_time=start_time)

    # Create a container for custom controls using HTML/JS
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
            # Return the timestamp to jump to
            return time_input

    with col3:
        if st.button("Reload Video", key=f"reload_{player_key}"):
            st.rerun()  # Updated from experimental_rerun to rerun

    return None  # No seek requested

def main():
    st.title("Video Player with RAG Question Answering")

    # Sidebar for video selection
    st.sidebar.header("Video Selection")

    # Use a more configurable approach for the videos folder
    if not videos_folder.exists():
        # Try to create the folder if it doesn't exist
        try:
            os.makedirs(videos_folder)
            st.sidebar.success(f"Created videos folder at {videos_folder}")
        except Exception as e:
            st.sidebar.error(f"The folder {videos_folder} does not exist and could not be created: {e}")
            st.sidebar.info("Please specify a valid path for your videos folder in the code.")
            return

    # Get list of video files from the videos folder
    video_files = list(videos_folder.glob("*.mp4")) + list(videos_folder.glob("*.avi")) + list(videos_folder.glob("*.mov"))

    if not video_files:
        st.warning(f"No video files found in {videos_folder}. Please add some videos.")
        return

    # Create a dropdown to select videos
    video_options = [file.name for file in video_files]
    selected_video = st.sidebar.selectbox("Select a video to play", video_options)

    # Get video path without extension for finding transcript
    video_base_name = Path(selected_video).stem

    # Display the selected video
    st.subheader("Video Player")
    video_path = str(videos_folder / selected_video)

    # Initialize session state for video time
    if 'current_video_time' not in st.session_state:
        st.session_state.current_video_time = 0

    # Display custom video player
    seek_result = custom_video_player(
        video_path,
        width=720,
        height=405,
        key="main_player",
        start_time=st.session_state.current_video_time
    )

    # Update the current time if seek was requested
    if seek_result is not None:
        st.session_state.current_video_time = seek_result
        st.rerun()  # Updated from experimental_rerun to rerun

    # Check for transcript files - using a more flexible approach
    transcript_path = os.path.join(transcripts_folder, f"{video_base_name.split('.')[0]}.csv")

    # Initialize session state for FAISS index and metadata
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
        st.session_state.metadata = None
        st.session_state.model_name = None

    # Load transcript data if available
    if os.path.isfile(transcript_path):
        if st.sidebar.button("Load Transcript Data"):
            with st.spinner("Loading transcript data and building search index..."):
                try:
                    # Check if pre-saved index exists
                    prefix = str(transcript_path).replace('.csv', '')
                    index_path = f"{prefix}_index.faiss"

                    if Path(index_path).exists():
                        # Load pre-saved data
                        st.sidebar.info("Loading pre-saved index...")
                        st.session_state.faiss_index, st.session_state.metadata, st.session_state.model_name = RAG.load_data(prefix)
                    else:
                        # Process transcript data for the first time
                        st.sidebar.info("Building new index from transcript...")
                        embeddings, faiss_index, metadata, model_name = RAG.process_transcript_data(
                            str(transcript_path),
                            model_name='all-MiniLM-L6-v2',
                            use_gpu=False
                        )
                        # Save the processed data
                        RAG.save_data(faiss_index, metadata, embeddings, model_name, prefix=prefix)

                        # Store in session state
                        st.session_state.faiss_index = faiss_index
                        st.session_state.metadata = metadata
                        st.session_state.model_name = model_name

                    st.sidebar.success("Transcript data loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error loading transcript data: {str(e)}")
    else:
        st.sidebar.warning(f"No transcript found for {selected_video}.")
        st.sidebar.warning(f"Expected path: {transcript_path}")
        st.sidebar.info(f"Create a CSV file named {video_base_name.split('.')[0]}.mp4temp_audio.wav.csv in the transcripts folder.")

    # Create a question form
    st.subheader("Ask a Question About the Video")
    with st.form(key="question_form"):
        question = st.text_area("Type your question about the video:")
        submit_button = st.form_submit_button(label="Submit Question")

    # Handle form submission
    if submit_button:
        if question.strip():
            if st.session_state.faiss_index is not None:
                with st.spinner("Searching for relevant parts of the video..."):
                    try:
                        # Perform semantic search
                        results = RAG.semantic_search(
                            question,
                            st.session_state.faiss_index,
                            st.session_state.metadata,
                            st.session_state.model_name,
                            top_k=3
                        )

                        # Display results
                        st.subheader(f"Results for: {question}")

                        for i, result in enumerate(results):
                            score = 1 - result['distance']/2  # Normalize to 0-1 score

                            # Create an expandable section for each result
                            with st.expander(f"Result {i+1} (Relevance: {score:.2f})"):
                                st.markdown(f"**Text:** {result['text']}")
                                st.markdown(f"**Timestamp:** {result['start']:.2f}s - {result['end']:.2f}s")

                                # Button to jump to timestamp
                                if st.button(f"Jump to {result['start']:.2f}s", key=f"jump_{i}"):
                                    st.session_state.current_video_time = result['start']
                                    st.rerun()  # Updated from experimental_rerun to rerun

                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
            else:
                st.warning("Please load the transcript data first by clicking the 'Load Transcript Data' button in the sidebar.")
        else:
            st.error("Please enter a question before submitting.")

if __name__ == "__main__":
    main()
