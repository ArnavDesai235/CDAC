import librosa
import numpy as np
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from typing import List, Dict
import os

class AudioTranscriber:
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Initialize the transcriber with Hugging Face Whisper model.

        Args:
            model_name: Hugging Face model name (default: "openai/whisper-tiny")
        """
        print(f"Loading Whisper model: {model_name}")

        # Load processor and model from Hugging Face
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        print(f"Model loaded successfully on device: {self.device}")

    def split_audio_into_segments(self, audio_path: str, max_duration: float = 7.0) -> List[Dict]:
        """
        Split audio file into segments of maximum duration.

        Args:
            audio_path: Path to the audio file
            max_duration: Maximum duration of each segment in seconds (default: 5.0)

        Returns:
            List of dictionaries containing segment info
        """
        # Load audio file at 16kHz (Whisper's expected sample rate)
        audio, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(audio) / sr

        segments = []
        start_time = 0.0

        while start_time < total_duration:
            end_time = min(start_time + max_duration, total_duration)

            # Convert time to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Extract audio segment
            audio_segment = audio[start_sample:end_sample]

            segments.append({
                'audio': audio_segment,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })

            start_time = end_time

        return segments

    def transcribe_segment(self, audio_segment: np.ndarray) -> str:
        """
        Transcribe a single audio segment using Hugging Face Whisper.

        Args:
            audio_segment: Audio data as numpy array

        Returns:
            Transcribed text
        """
        # Process audio with the processor
        input_features = self.processor(
            audio_segment,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Move to device
        input_features = input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # Decode the transcription
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()

    def transcribe_audio_file(self, audio_path: str, max_segment_duration: float = 5.0) -> List[Dict]:
        """
        Transcribe entire audio file with segmentation.

        Args:
            audio_path: Path to the audio file
            max_segment_duration: Maximum duration of each segment in seconds

        Returns:
            List of transcription results with timestamps
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Processing audio file: {audio_path}")

        # Split audio into segments
        segments = self.split_audio_into_segments(audio_path, max_segment_duration)
        print(f"Split into {len(segments)} segments")

        transcription_results = []

        for i, segment in enumerate(segments):
            print(f"Transcribing segment {i+1}/{len(segments)} ({segment['start']:.2f}s - {segment['end']:.2f}s)")

            # Transcribe segment
            text = self.transcribe_segment(segment['audio'])

            result = {
                'segment_id': i,
                'text': text,
                'start': round(segment['start'], 2),
                'end': round(segment['end'], 2),
                'duration': round(segment['duration'], 2)
            }

            transcription_results.append(result)
            print(f"  Text: {text}")

        return transcription_results

    def save_transcription(self, transcription_results: List[Dict], output_path: str):
        """
        Save transcription results to a file.

        Args:
            transcription_results: List of transcription dictionaries
            output_path: Path to save the transcription file
        """
        # Determine file format based on extension
        file_ext = os.path.splitext(output_path)[1].lower()

        if file_ext == '.json':
            # Save as JSON
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_results, f, indent=2, ensure_ascii=False)

        elif file_ext == '.csv':
            # Save as CSV using pandas
            df = pd.DataFrame(transcription_results)
            df.to_csv(output_path, index=False, encoding='utf-8')

        elif file_ext == '.txt':
            # Save as plain text with timestamps
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Audio Transcription with Timestamps\n")
                f.write("=" * 40 + "\n\n")

                for result in transcription_results:
                    f.write(f"[{result['start']:.2f}s - {result['end']:.2f}s] {result['text']}\n")

        else:
            # Default to SRT format
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(transcription_results):
                    start_time = self.seconds_to_srt_time(result['start'])
                    end_time = self.seconds_to_srt_time(result['end'])

                    f.write(f"{i+1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{result['text']}\n\n")

        print(f"Transcription saved to: {output_path}")

    def get_transcription_dataframe(self, transcription_results: List[Dict]) -> pd.DataFrame:
        """
        Convert transcription results to pandas DataFrame for analysis.

        Args:
            transcription_results: List of transcription dictionaries

        Returns:
            pandas DataFrame with transcription data
        """
        df = pd.DataFrame(transcription_results)

        # Add some useful derived columns
        df['word_count'] = df['text'].str.split().str.len().fillna(0)
        df['char_count'] = df['text'].str.len()
        df['words_per_second'] = (df['word_count'] / df['duration']).round(2)

        return df

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

