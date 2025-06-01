import cv2
from fer import FER
import numpy as np
import subprocess
import whisper
import os
from pydub import AudioSegment
import language_tool_python
from collections import defaultdict


# --- Audio Extraction ---
def extract_audio_from_video(video_file, output_audio_file="temp_audio.wav"):
    try:
        command = [
            'ffmpeg',
            '-i', video_file,
            '-vn',  # No video
            '-acodec', 'pcm_s16le', # Ensure compatibility with pydub (raw PCM)
            '-ac', '1',          # Mono audio (Whisper performs best with mono)
            '-ar', '16000',       # 16kHz sample rate (common for speech)
            output_audio_file
        ]
        subprocess.run(command, check=True, capture_output=True)
        return output_audio_file
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error extracting audio: {e}")
        return None

# --- Audio Transcription with Timestamps ---
def transcribe_audio_with_timestamps(audio_file):
    model = whisper.load_model("small")
    audio = AudioSegment.from_wav(audio_file)
    segment_length_ms = 600000
    segments = [audio[i:i + segment_length_ms] for i in range(0, len(audio), segment_length_ms)]
    full_transcript = []
    for i, segment in enumerate(segments):
        segment_file = f"temp_segment_{i}.wav"
        segment.export(segment_file, format="wav")
        result = model.transcribe(segment_file, task="translate", word_timestamps=True)
        full_transcript.extend(result["segments"])
        os.remove(segment_file)
    return full_transcript

# --- Basic Speaking Confidence Analysis (from timestamps) ---
def analyze_speaking_confidence(transcript_segments):
    total_words = 0
    total_duration = 0
    pause_threshold = 0.5  # Seconds
    significant_pause_count = 0

    for segment in transcript_segments:
        words = segment.get("words", [])
        if words:
            start_time = words[0]["start"]
            end_time = words[-1]["end"]
            segment_duration = end_time - start_time
            num_words = len(words)
            total_words += num_words
            total_duration += segment_duration

            for i in range(len(words) - 1):
                gap = words[i+1]["start"] - words[i]["end"]
                if gap > pause_threshold:
                    significant_pause_count += 1

    overall_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
    speaking_confidence_score = 70
    if overall_wpm < 100 or overall_wpm > 160:
        speaking_confidence_score -= 10
    if significant_pause_count > (total_words / 100 * 5):
        speaking_confidence_score -= 15
    speaking_confidence_score = max(0, min(100, speaking_confidence_score))

    return speaking_confidence_score

# --- Facial Emotion Confidence Analysis ---
def analyze_facial_confidence(video_path, window_size=10, confidence_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    detector = FER(mtcnn=True)
    confidence_scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        top_emotion = detector.top_emotion(frame)
        confidence_scores.append(top_emotion[1] if top_emotion else None)
    cap.release()

    valid_scores = np.array([score for score in confidence_scores if score is not None])
    smoothed_confidence = np.convolve(valid_scores, np.ones(window_size)/window_size, mode='same') if valid_scores.size > 0 else np.array([])
    high_confidence_percentage = np.mean(smoothed_confidence >= confidence_threshold) * 100 if smoothed_confidence.size > 0 else 0.0
    average_confidence = np.nanmean(smoothed_confidence) if smoothed_confidence.size > 0 else 0.0

    combined_facial_confidence = (0.6 * average_confidence * 100 + 0.4 * high_confidence_percentage) if smoothed_confidence.size > 0 else 0.0
    return combined_facial_confidence

# --- Grammar Checking ---
def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return matches

def calculate_grammar_score(full_transcript_text):
    grammar_errors = check_grammar(full_transcript_text)
    num_words = len(full_transcript_text.split()) if full_transcript_text else 0
    num_errors = len(grammar_errors)
    major_error_categories = ["GRAMMAR", "SENTENCE_STRUCTURE"]
    error_categories = defaultdict(int)
    for error in grammar_errors:
        error_categories[error.category] += 1
    num_major_errors = sum(error_categories[cat] for cat in major_error_categories)

    overall_error_rate = num_errors / num_words if num_words > 0 else 0
    major_error_rate = num_major_errors / num_words if num_words > 0 else 0

    estimated_overall_correctness = (1 - overall_error_rate) * 100 if num_words > 0 else 100.0
    estimated_major_correctness = (1 - major_error_rate) * 100 if num_words > 0 else 100.0

    grammar_score = (0.7 * estimated_major_correctness + 0.3 * estimated_overall_correctness) if num_words > 0 else 100.0
    return grammar_score

# --- Main Function (Simplified Output) ---
def analyze_video_for_scores(video_path):
    temp_audio_file = "temp_audio.wav"

    audio_file = extract_audio_from_video(video_path, temp_audio_file)

    if audio_file:
        transcript_segments = transcribe_audio_with_timestamps(audio_file)
        full_transcript_text = " ".join([word_info["word"] for segment in transcript_segments for word_info in segment.get("words", [])])
        speaking_confidence = analyze_speaking_confidence(transcript_segments)
        facial_confidence = analyze_facial_confidence(video_path)
        grammar_score = calculate_grammar_score(full_transcript_text)
        os.remove(temp_audio_file)
        return speaking_confidence, facial_confidence, grammar_score
    else:
        return 0.0, 0.0, 0.0

if __name__ == "__main__":
    video_folder = "answers_video"
    video_file_name = "myvideo1"
    video_path = os.path.join(video_folder, f"{video_file_name}.webm") # Assuming .webm extension

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        speaking_confidence, facial_confidence, grammar_score = analyze_video_for_scores(video_path)
        print(f"Speaking Confidence Score   : {speaking_confidence:.2f}")
        print(f"Facial Confidence Score     : {facial_confidence:.2f}")
        print(f"Grammar Confidence Score    : {grammar_score:.2f}")