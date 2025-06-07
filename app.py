from flask import Flask, request, redirect, url_for, render_template, session
import os
import subprocess
import cv2
from fer import FER
import numpy as np
import whisper
from pydub import AudioSegment
import language_tool_python
from collections import defaultdict
import io
from concurrent.futures import ThreadPoolExecutor # Import ThreadPoolExecutor

app = Flask(__name__)
app.secret_key = b'\x1c\x9a\x85\x01\x9b\x1d\xee\xa3\x16\x08\x9c\xa6\x8e\x19\x7d\x0f\x8f\xeb\x8f\x19\xfa\x99\xcd\x17'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ANSWERS_FOLDER = 'answers_video'
os.makedirs(ANSWERS_FOLDER, exist_ok=True)

# --- Global Model Loading (for efficiency and thread-safety where applicable) ---
# Load Whisper model once when the app starts
global_whisper_model = whisper.load_model("small")

# Load LanguageTool model once when the app starts
# Note: language_tool_python's LanguageTool can be relatively slow to initialize.
# While the check method is generally thread-safe, initializing it once is efficient.
global_language_tool = language_tool_python.LanguageTool('en-US')

# Initialize a ThreadPoolExecutor for concurrent video processing
# Adjust max_workers based on your system's CPU cores and expected load.
# A common heuristic is the number of CPU cores or slightly more for I/O-bound tasks.
executor = ThreadPoolExecutor(max_workers=4)

# --- Helper Functions ---
def extract_audio_from_video(video_file, output_audio_file):
    try:
        command = [
            'ffmpeg',
            '-i', video_file,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Ensure compatibility with pydub (raw PCM)
            '-ac', '1',  # Mono audio (Whisper performs best with mono)
            '-ar', '16000',  # 16kHz sample rate (common for speech)
            output_audio_file
        ]
        subprocess.run(command, check=True, capture_output=True)
        return output_audio_file
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio_with_timestamps(audio_file):
    # Use the globally loaded Whisper model
    model = global_whisper_model
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
                gap = words[i + 1]["start"] - words[i]["end"]
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

def analyze_facial_confidence(video_path, window_size=10, confidence_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    # Re-initialize FER detector per thread/call. While this incurs overhead,
    # it's generally safer for libraries that might not be fully thread-safe
    # or have internal state that's modified during processing.
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
    smoothed_confidence = np.convolve(valid_scores, np.ones(window_size) / window_size,
                                      mode='same') if valid_scores.size > 0 else np.array([])
    high_confidence_percentage = np.mean(smoothed_confidence >= confidence_threshold) * 100 if smoothed_confidence.size > 0 else 0.0
    average_confidence = np.nanmean(smoothed_confidence) if smoothed_confidence.size > 0 else 0.0

    combined_facial_confidence = (
        0.6 * average_confidence * 100 + 0.4 * high_confidence_percentage) if smoothed_confidence.size > 0 else 0.0
    return combined_facial_confidence

def check_grammar(text):
    # Use the globally loaded LanguageTool instance
    tool = global_language_tool
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

def run_resume_script(upload_folder):
    """
    Runs the appropriate resume processing script on the latest uploaded file.

    Args:
        upload_folder (str): The path to the uploads folder.

    Returns:
        tuple: A tuple containing:
            - bool: True if the script ran successfully, False otherwise.
            - str: The path to the output file ('resume_output.json') on success,
                    or the error message on failure.
    """
    try:
        files_in_uploads = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
        if not files_in_uploads:
            return False, "No files found in the uploads folder."

        latest_file = max(
            [os.path.join(upload_folder, f) for f in files_in_uploads],
            key=os.path.getmtime
        )

        if latest_file.endswith(".pdf"):
            command = ['python', 'pdf_to_json.py', latest_file]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode == 0:
                return True, "resume_output.json", latest_file  # Return the path of the processed file
            else:
                return False, process.stderr, latest_file
        elif latest_file.endswith(".csv"):
            return False, "CSV processing not yet implemented", latest_file
        else:
            return False, "Unsupported file type in the latest upload.", latest_file

    except FileNotFoundError:
        return False, f"Error: Uploads folder not found at {upload_folder}", None
    except Exception as e:
        return False, f"An unexpected error occurred: {e}", None

def generate_questions():
    process = subprocess.run(['python', 'generate_questions.py'], capture_output=True, text=True)
    if process.returncode == 0:
        return True, "generated_questions.txt"
    else:
        return False, process.stderr

# --- New helper function for processing a single video in a thread ---
def process_single_video(video_file_path):
    # Generate a unique temporary audio file name for each video processing task
    audio_file_path = f"temp_audio_{os.path.basename(video_file_path).replace('.', '_')}.wav"

    try:
        if extract_audio_from_video(video_file_path, audio_file_path):
            transcript_segments = transcribe_audio_with_timestamps(audio_file_path)
            speaking_confidence = analyze_speaking_confidence(transcript_segments)
            facial_confidence = analyze_facial_confidence(video_file_path)
            full_transcript_text = " ".join(
                [word_info["word"] for segment in transcript_segments for word_info in segment.get("words", [])])
            grammar_score = calculate_grammar_score(full_transcript_text)
            return speaking_confidence, facial_confidence, grammar_score
        else:
            print(f"Failed to extract audio from video file: {video_file_path}")
            return None
    except Exception as e:
        print(f"Error processing video {video_file_path}: {e}")
        return None
    finally:
        # Ensure temporary audio file is removed even if errors occur
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('combine.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return 'No file part'
    file = request.files['resume']
    if file.filename == '':
        return 'No selected file'
    if file and (file.filename.endswith('.pdf') or file.filename.endswith('.csv')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        success, output_path, uploaded_file_path = run_resume_script(
            app.config['UPLOAD_FOLDER'])  # Get the uploaded file path
        if success:
            questions_success, questions_path = generate_questions()
            if questions_success:
                # Remove the resume file after successful processing and before redirecting
                try:
                    os.remove(uploaded_file_path)
                    print(f"Removed processed resume file: {uploaded_file_path}")
                    # Extract candidate name from the first line of the questions file
                    try:
                        with open(questions_path, 'r') as f:
                            first_line = f.readline()
                        candidate_name = first_line.strip()
                        # Basic heuristics to find a name (adjust as needed)
                        if " " not in candidate_name or len(candidate_name) < 3:
                            candidate_name = "Candidate"
                        session['candidate_name'] = candidate_name
                    except Exception as e:
                        print(f"Error extracting candidate name from questions file: {e}")
                        candidate_name = "Candidate"
                        session['candidate_name'] = candidate_name

                    return redirect(url_for('interview_page'))
                except OSError as e:
                    print(f"Error removing resume file {uploaded_file_path}: {e}")
                    return redirect(url_for('interview_page'))  # Still redirect even if removal fails
            else:
                return f'Error generating questions: {questions_path}'
        else:
            return f'Error processing resume: {output_path}'
    return 'Invalid file type'

@app.route('/interview', methods=['GET'])
def interview_page():
    try:
        with open('generated_questions.txt', 'r') as f:
            lines = f.readlines()

        selected_lines = []
        ranges = [
            (5, 14),
            (19, 28),
            (33, 37)
        ]

        for start, end in ranges:
            # Adjust for 0-based indexing
            start -= 1
            end -= 1
            if start < len(lines) and end < len(lines):
                selected_lines.extend(lines[start:end + 1])

        # Clean up the lines (remove leading/trailing whitespace and newlines)
        questions = [line.strip() for line in selected_lines if line.strip()]

        return render_template('interview.html', questions=questions)

    except FileNotFoundError:
        return 'Error: generated_questions.txt not found. Please upload a resume first.'

@app.route('/save_video/<int:question_number>', methods=['POST'])
def save_video(question_number):
    if 'video_data' not in request.files:
        return 'No video data received', 400
    video_file = request.files['video_data']
    if video_file.filename == '':
        return 'No video file name', 400
    filename = f'answer{question_number}.webm'
    filepath = os.path.join(ANSWERS_FOLDER, filename)
    try:
        video_file.save(filepath)
        return 'Video saved successfully', 200
    except Exception as e:
        return f'Error saving video: {e}', 500

@app.route('/end_interview', methods=['POST'])
def end_interview():
    candidate_name = session.get('candidate_name', 'Unknown')
    return render_template('thank_you.html', candidate_name=candidate_name)

@app.route('/calculate_confidence', methods=['POST'])
def calculate_confidence():
    candidate_name = session.get('candidate_name', 'Unknown')
    
    video_files_to_process = [f for f in os.listdir(ANSWERS_FOLDER) if f.endswith('.webm')]
    if not video_files_to_process:
        return "No video files found in the answers folder."

    futures = []
    video_file_paths_to_delete = []

    for video_file_name in video_files_to_process:
        video_file_path = os.path.join(ANSWERS_FOLDER, video_file_name)
        video_file_paths_to_delete.append(video_file_path)
        # Submit each video processing task to the thread pool
        futures.append(executor.submit(process_single_video, video_file_path))

    speaking_confidences = []
    facial_confidences = []
    grammar_scores = []

    # Collect results as they complete
    for future in futures:
        result = future.result() # This will block until the task is complete
        if result:
            speaking_confidence, facial_confidence, grammar_score = result
            speaking_confidences.append(speaking_confidence)
            facial_confidences.append(facial_confidence)
            grammar_scores.append(grammar_score)
        else:
            print(f"Skipping a video due to processing failure.")


    # Delete video files after processing
    for video_file_path in video_file_paths_to_delete:
        try:
            os.remove(video_file_path)
            print(f"Deleted video file: {video_file_path}")
        except OSError as e:
            print(f"Error deleting video file {video_file_path}: {e}")

    # Calculate averages.
    avg_speaking_confidence = np.mean(speaking_confidences) if speaking_confidences else 0
    avg_facial_confidence = np.mean(facial_confidences) if facial_confidences else 0
    avg_grammar_score = np.mean(grammar_scores) if grammar_scores else 0
    
    return render_template('results.html', candidate_name=candidate_name,
                           speaking_confidence=f"{avg_speaking_confidence:.2f}",
                           facial_confidence=f"{avg_facial_confidence:.2f}",
                           grammar_score=f"{avg_grammar_score:.2f}")

if __name__ == '__main__':
    app.run(debug=True)