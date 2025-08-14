import RAG
import VideoProcess
import Transcription
from config import VIDEOS_FOLDER, WAV_FOLDER, TRANSCRIPTS_FOLDER

def main():
    for file in VIDEOS_FOLDER.glob("*.*"):
        if file.suffix.lower() not in [".mp4", ".avi", ".mov"]:
            continue

        name = file.stem
        print(f"\nProcessing video: {file.name}")

        # Convert video to WAV
        audio_file = VideoProcess.mp4_to_wav(file)

        # Initialize transcriber
        transcriber = Transcription.AudioTranscriber(model_name="openai/whisper-small")

        output_file = TRANSCRIPTS_FOLDER / f"{name}.csv"

        try:
            # Transcribe audio
            results = transcriber.transcribe_audio_file(str(audio_file), max_segment_duration=5.0)
            transcriber.save_transcription(results, str(output_file))

            # Optional: Preview
            df = transcriber.get_transcription_dataframe(results)
            print(df[['segment_id', 'start', 'end', 'word_count', 'words_per_second', 'text']].head())

            print(f"✅ Transcription completed! Saved to: {output_file}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    main()
