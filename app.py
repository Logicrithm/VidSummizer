"""
VidSummarize - Flask Backend Application
UNICODE FIXED: Proper handling for multilingual content (Hindi, Tamil, etc.)
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
import uuid
import json
import threading
import re
from datetime import datetime
from video_processor import VideoProcessor

# ========== CRITICAL UNICODE FIX ==========
# Force UTF-8 encoding for console output (fixes Windows scribbles)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Set environment for proper Unicode handling
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ========== TRANSCRIBER SELECTION ==========
USE_SMART_TRANSCRIBER = True

try:
    if USE_SMART_TRANSCRIBER:
        from smart_transcriber import transcribe_audio
        TRANSCRIBER_TYPE = "smart_v2"
        print("✅ Loaded Smart Transcriber v2")
    else:
        raise ImportError("Manual fallback")
except ImportError:
    try:
        from fast_transcriber import Transcriber
        TRANSCRIBER_TYPE = "enhanced"
        print("✅ Loaded Enhanced Transcriber")
    except ImportError:
        from transcriber_legacy import Transcriber
        TRANSCRIBER_TYPE = "legacy"
        print("⚠️ Using Legacy Transcriber")

# Summarizer import
try:
    from summarizer import Summarizer
except ImportError:
    from summarizer_legacy import Summarizer

from pdf_generator import PDFGenerator


# ========== FLASK APP CONFIGURATION ==========

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False  # CRITICAL: Allow Unicode in JSON responses

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

video_processor = VideoProcessor(
    upload_folder=app.config['UPLOAD_FOLDER'],
    output_folder=app.config['OUTPUT_FOLDER']
)

transcriber = None
summarizer = None
pdf_generator = None

def get_transcriber():
    global transcriber
    if transcriber is None:
        print("[App] Loading transcription model...")
        
        if TRANSCRIBER_TYPE == "smart_v2":
            transcriber = "smart_v2_ready"
            print("✅ Smart Transcriber v2 ready (models load on demand)")
        elif TRANSCRIBER_TYPE == "enhanced":
            transcriber = Transcriber(model_size='base', performance_mode='balanced')
            if hasattr(transcriber, 'backend'):
                print(f"✅ Using {transcriber.backend} backend")
                print(f"   Device: {transcriber.device}")
        else:
            transcriber = Transcriber(model_size='base')
            print("✅ Legacy transcriber loaded")
    
    return transcriber

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("[App] Loading BART summarization model...")
        summarizer = Summarizer(model_name='facebook/bart-large-cnn')
    return summarizer

def get_pdf_generator():
    global pdf_generator
    if pdf_generator is None:
        print("[App] Initializing PDF generator...")
        pdf_generator = PDFGenerator()
    return pdf_generator

jobs = {}


# ========== HELPER FUNCTIONS ==========

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_job_id():
    return str(uuid.uuid4())


def create_job(job_type, source):
    job_id = generate_job_id()
    jobs[job_id] = {
        'id': job_id,
        'type': job_type,
        'source': source,
        'stage': 'uploading',
        'progress': 0,
        'status': 'processing',
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'language': None,
        'requested_language': None,
        'duration': None,
        'error': None,
        'transcript': None,
        'summary': None,
        'audio_path': None,
        'video_path': None,
        'transcript_path': None,
        'summary_path': None,
        'transcript_pdf_path': None,
        'summary_pdf_path': None,
        'word_count': 0,
        'summary_word_count': 0,
        'transcriber_type': TRANSCRIBER_TYPE,
        'transcription_strategy': None,
        'transcription_time': None,
        'speed_ratio': None
    }
    return job_id


def normalize_language_code(language):
    if not language:
        return None
    normalized = language.strip()
    if not normalized or normalized.lower() in {"auto", "detect", "none"}:
        return None
    return normalized


def update_job(job_id, updates):
    if job_id in jobs:
        jobs[job_id].update(updates)
        jobs[job_id]['updated_at'] = datetime.now().isoformat()
        return True
    return False


def get_job(job_id):
    return jobs.get(job_id)


def transcribe_with_fallback(audio_path, language=None):
    """
    UNICODE SAFE: Unified transcription with proper text handling
    """
    language = normalize_language_code(language)
    if TRANSCRIBER_TYPE == "smart_v2":
        result = transcribe_audio(audio_path, language=language)
        
        return {
            'success': True,
            'text': result['text'],  # Already Unicode from Whisper
            'language': result.get('language', 'unknown'),
            'word_count': len(result['text'].split()),
            'segments': result.get('segments', []),
            'duration': result.get('profile', {}).get('duration', 0),
            'processing_time': result.get('total_time', 0),
            'speed_ratio': result.get('speed_ratio', 0),
            'strategy': result.get('strategy', 'unknown'),
            'backend': result.get('backend', 'faster_whisper')
        }
    
    elif TRANSCRIBER_TYPE == "enhanced":
        trans = get_transcriber()
        try:
            result = trans.transcribe_audio(audio_path, language=language)
        except TypeError:
            result = trans.transcribe_audio(audio_path)
        
        return {
            'success': result.get('success', False),
            'text': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'word_count': result.get('word_count', 0),
            'segments': result.get('segments', []),
            'duration': result.get('duration', 0),
            'processing_time': result.get('processing_time', 0),
            'speed_ratio': result.get('speed_ratio', 0),
            'strategy': 'enhanced',
            'backend': getattr(trans, 'backend', 'unknown')
        }
    
    else:
        trans = get_transcriber()
        try:
            result = trans.transcribe_audio(audio_path, language=language)
        except TypeError:
            result = trans.transcribe_audio(audio_path)
        
        return {
            'success': result.get('success', False),
            'text': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'word_count': result.get('word_count', 0),
            'segments': result.get('segments', []),
            'duration': 0,
            'processing_time': 0,
            'speed_ratio': 0,
            'strategy': 'legacy',
            'backend': 'whisper'
        }


def save_utf8(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def save_transcript(text, output_path):
    """
    UNICODE SAFE: Always use UTF-8 encoding
    """
    try:
        save_utf8(output_path, text)
        return True
    except Exception as e:
        print(f"[App] Error saving transcript: {e}")
        return False


def save_debug_checkpoint(stage, text, job_id):
    """
    Debug helper: Save text at each stage to verify Unicode handling
    """
    try:
        debug_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_debug_{stage}.txt")
        content = f"=== {stage.upper()} ===\n\n{text}"
        save_utf8(debug_path, content)
        print(f"[Debug] Saved {stage} checkpoint: {len(text)} chars")
    except Exception as e:
        print(f"[Debug] Failed to save {stage} checkpoint: {e}")


def clean_transcript_text(text):
    """
    Unicode-safe cleanup: normalize spaces only, keep all characters.
    """
    return re.sub(r'\s+', ' ', text).strip()


def process_video_job(job_id, filepath, language=None):
    """
    UNICODE SAFE: Video processing with proper text handling
    """
    try:
        print(f"[Job {job_id}] Starting video processing with {TRANSCRIBER_TYPE} transcriber...")
        
        update_job(job_id, {
            'stage': 'extracting_audio',
            'progress': 20
        })
        
        result = video_processor.extract_audio_from_video(filepath)
        
        if not result['success']:
            raise Exception(result['error'])
        
        audio_path = result['audio_path']
        
        update_job(job_id, {
            'stage': 'transcribing',
            'progress': 40,
            'audio_path': audio_path,
            'video_path': filepath,
            'duration': result.get('duration', 0)
        })
        
        print(f"[Job {job_id}] Audio extraction completed")
        
        # TRANSCRIPTION - Unicode comes from Whisper natively
        print(f"[Job {job_id}] Starting transcription ({TRANSCRIBER_TYPE})...")
        
        transcription_result = transcribe_with_fallback(audio_path, language=language)
        
        if not transcription_result['success']:
            raise Exception(transcription_result.get('error', 'Transcription failed'))
        
        transcript_text = transcription_result['text']
        detected_language = transcription_result.get('language', 'unknown')
        word_count = transcription_result.get('word_count', 0)
        
        # DEBUG: Save raw + cleaned checkpoints
        raw_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_raw.txt")
        save_utf8(raw_path, transcript_text)
        cleaned_text = clean_transcript_text(transcript_text)
        cleaned_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_cleaned.txt")
        save_utf8(cleaned_path, cleaned_text)
        
        if not transcript_text or len(transcript_text.strip()) < 10:
            raise Exception("Transcription produced no text. The audio may be empty or unclear.")
        
        if word_count < 5:
            raise Exception(f"Transcription too short ({word_count} words). Please check the audio quality.")
        
        # Save transcript with UTF-8
        transcript_filename = f"{job_id}_transcript.txt"
        transcript_path = os.path.join(app.config['OUTPUT_FOLDER'], transcript_filename)
        save_transcript(transcript_text, transcript_path)
        
        update_job(job_id, {
            'stage': 'summarizing',
            'progress': 60,
            'transcript': transcript_text,
            'transcript_path': transcript_path,
            'language': detected_language,
            'word_count': word_count,
            'transcription_strategy': transcription_result.get('strategy', 'unknown'),
            'transcription_time': transcription_result.get('processing_time', 0),
            'speed_ratio': transcription_result.get('speed_ratio', 0)
        })
        
        print(f"[Job {job_id}] Transcription completed")
        print(f"[Job {job_id}] Language: {detected_language}, Words: {word_count}")
        
        # SUMMARIZATION
        print(f"[Job {job_id}] Starting summarization...")
        
        summary_input = cleaned_text or transcript_text
        try:
            summ = get_summarizer()
            summary_result = summ.summarize_text(summary_input)
            
            if not summary_result['success']:
                error_msg = summary_result.get('error', 'Summarization failed')
                print(f"[Job {job_id}] Summarization error: {error_msg}")
                
                if 'warning' in summary_result:
                    summary_text = summary_result.get('summary', summary_input)
                    summary_word_count = len(summary_text.split())
                else:
                    raise Exception(error_msg)
            else:
                summary_text = summary_result['summary']
                summary_word_count = summary_result.get('word_count', len(summary_text.split()))
            
            # DEBUG: Save summary
            save_debug_checkpoint('summary', summary_text, job_id)
            
            if not summary_text or len(summary_text.strip()) < 5:
                print(f"[Job {job_id}] Summary too short, using transcript excerpt")
                words = summary_input.split()[:200]
                summary_text = ' '.join(words) + ('...' if len(summary_input.split()) > 200 else '')
                summary_word_count = len(summary_text.split())
            
        except Exception as summ_error:
            print(f"[Job {job_id}] Summarization failed: {str(summ_error)}")
            words = summary_input.split()[:200]
            summary_text = ' '.join(words) + ('...' if len(summary_input.split()) > 200 else '')
            summary_word_count = len(summary_text.split())
        
        # Save summary with UTF-8
        summary_filename = f"{job_id}_summary.txt"
        summary_path = os.path.join(app.config['OUTPUT_FOLDER'], summary_filename)
        
        save_utf8(summary_path, summary_text)
        
        update_job(job_id, {
            'stage': 'generating_output',
            'progress': 80,
            'summary': summary_text,
            'summary_path': summary_path,
            'summary_word_count': summary_word_count
        })
        
        print(f"[Job {job_id}] Summarization completed (Words: {summary_word_count})")
        
        # PDF GENERATION
        print(f"[Job {job_id}] Generating PDF files...")
        
        pdf_gen = get_pdf_generator()
        
        metadata = {
            'language': detected_language,
            'duration': result.get('duration', 0),
            'word_count': word_count,
            'date': datetime.now().strftime('%B %d, %Y')
        }
        
        transcript_pdf_filename = f"{job_id}_transcript.pdf"
        transcript_pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], transcript_pdf_filename)
        pdf_gen.generate_transcript_pdf(transcript_text, transcript_pdf_path, metadata)
        
        summary_pdf_filename = f"{job_id}_summary.pdf"
        summary_pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], summary_pdf_filename)
        pdf_gen.generate_summary_pdf(summary_text, summary_pdf_path, metadata)
        
        update_job(job_id, {
            'stage': 'completed',
            'progress': 100,
            'status': 'completed',
            'transcript_pdf_path': transcript_pdf_path,
            'summary_pdf_path': summary_pdf_path
        })
        
        print(f"[Job {job_id}] Processing completed successfully")
        
    except Exception as e:
        print(f"[Job {job_id}] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        update_job(job_id, {
            'stage': 'failed',
            'status': 'failed',
            'progress': 0,
            'error': str(e)
        })


def process_youtube_job(job_id, youtube_url, language=None):
    """
    UNICODE SAFE: YouTube processing with proper text handling
    """
    try:
        print(f"[Job {job_id}] Starting YouTube processing...")
        
        update_job(job_id, {
            'stage': 'uploading',
            'progress': 10
        })
        
        result = video_processor.download_youtube_video(youtube_url)
        
        if not result['success']:
            raise Exception(result['error'])
        
        audio_path = result['audio_path']
        
        update_job(job_id, {
            'stage': 'transcribing',
            'progress': 40,
            'audio_path': audio_path,
            'video_path': result.get('video_path'),
            'duration': result.get('duration', 0)
        })
        
        print(f"[Job {job_id}] YouTube download completed")
        
        transcription_result = transcribe_with_fallback(audio_path, language=language)
        
        if not transcription_result['success']:
            raise Exception(transcription_result.get('error', 'Transcription failed'))
        
        transcript_text = transcription_result['text']
        detected_language = transcription_result.get('language', 'unknown')
        word_count = transcription_result.get('word_count', 0)
        
        raw_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_raw.txt")
        save_utf8(raw_path, transcript_text)
        cleaned_text = clean_transcript_text(transcript_text)
        cleaned_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_cleaned.txt")
        save_utf8(cleaned_path, cleaned_text)
        
        if not transcript_text or len(transcript_text.strip()) < 10:
            raise Exception("Transcription produced no text.")
        
        transcript_filename = f"{job_id}_transcript.txt"
        transcript_path = os.path.join(app.config['OUTPUT_FOLDER'], transcript_filename)
        save_transcript(transcript_text, transcript_path)
        
        update_job(job_id, {
            'stage': 'summarizing',
            'progress': 60,
            'transcript': transcript_text,
            'transcript_path': transcript_path,
            'language': detected_language,
            'word_count': word_count,
            'transcription_strategy': transcription_result.get('strategy', 'unknown'),
            'transcription_time': transcription_result.get('processing_time', 0),
            'speed_ratio': transcription_result.get('speed_ratio', 0)
        })
        
        print(f"[Job {job_id}] Transcription completed")
        
        # Summarization (same as video processing)
        summary_input = cleaned_text or transcript_text
        try:
            summ = get_summarizer()
            summary_result = summ.summarize_text(summary_input)
            
            if summary_result['success']:
                summary_text = summary_result['summary']
                summary_word_count = summary_result.get('word_count', len(summary_text.split()))
            else:
                words = summary_input.split()[:200]
                summary_text = ' '.join(words) + '...'
                summary_word_count = len(summary_text.split())
        except:
            words = summary_input.split()[:200]
            summary_text = ' '.join(words) + '...'
            summary_word_count = len(summary_text.split())
        
        summary_filename = f"{job_id}_summary.txt"
        summary_path = os.path.join(app.config['OUTPUT_FOLDER'], summary_filename)
        
        save_utf8(summary_path, summary_text)
        
        update_job(job_id, {
            'stage': 'generating_output',
            'progress': 80,
            'summary': summary_text,
            'summary_path': summary_path,
            'summary_word_count': summary_word_count
        })
        
        # PDF generation
        pdf_gen = get_pdf_generator()
        
        metadata = {
            'language': detected_language,
            'duration': result.get('duration', 0),
            'word_count': word_count,
            'date': datetime.now().strftime('%B %d, %Y')
        }
        
        transcript_pdf_filename = f"{job_id}_transcript.pdf"
        transcript_pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], transcript_pdf_filename)
        pdf_gen.generate_transcript_pdf(transcript_text, transcript_pdf_path, metadata)
        
        summary_pdf_filename = f"{job_id}_summary.pdf"
        summary_pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], summary_pdf_filename)
        pdf_gen.generate_summary_pdf(summary_text, summary_pdf_path, metadata)
        
        update_job(job_id, {
            'stage': 'completed',
            'progress': 100,
            'status': 'completed',
            'transcript_pdf_path': transcript_pdf_path,
            'summary_pdf_path': summary_pdf_path
        })
        
        print(f"[Job {job_id}] YouTube processing completed")
        
    except Exception as e:
        print(f"[Job {job_id}] YouTube processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        update_job(job_id, {
            'stage': 'failed',
            'status': 'failed',
            'progress': 0,
            'error': str(e)
        })


# ========== PAGE ROUTES ==========

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/processing')
def processing():
    job_id = request.args.get('job_id')
    if not job_id:
        return "Job ID is required", 400
    return render_template('processing.html', job_id=job_id)


@app.route('/results')
def results():
    job_id = request.args.get('job_id')
    if not job_id:
        return "Job ID is required", 400
    return render_template('results.html', job_id=job_id)


# ========== API ROUTES ==========

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'VidSummarize API is running',
        'transcriber': TRANSCRIBER_TYPE,
        'unicode_support': True,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['video']
    language = normalize_language_code(request.form.get('language') or request.args.get('language'))
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)
        job_id = create_job('upload', filepath)
        update_job(job_id, {'requested_language': language})
        
        thread = threading.Thread(target=process_video_job, args=(job_id, filepath, language))
        thread.daemon = True
        thread.start()
        
        print(f"[API] File uploaded successfully: {job_id}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'File uploaded successfully, processing started'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/youtube', methods=['POST'])
def process_youtube():
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'YouTube URL is required'}), 400
    
    youtube_url = data['url']
    language = normalize_language_code(data.get('language'))
    
    if 'youtube.com' not in youtube_url and 'youtu.be' not in youtube_url:
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    
    try:
        job_id = create_job('youtube', youtube_url)
        update_job(job_id, {'requested_language': language})
        
        thread = threading.Thread(target=process_youtube_job, args=(job_id, youtube_url, language))
        thread.daemon = True
        thread.start()
        
        print(f"[API] YouTube URL submitted: {job_id}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'YouTube URL submitted, processing started'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify({
        'success': True,
        'status': job['status'],
        'stage': job['stage'],
        'progress': job.get('progress', 0),
        'message': f"Processing: {job['stage']}",
        'language': job.get('language'),
        'duration': job.get('duration'),
        'word_count': job.get('word_count', 0),
        'summary_word_count': job.get('summary_word_count', 0),
        'transcriber_type': job.get('transcriber_type'),
        'transcription_strategy': job.get('transcription_strategy'),
        'speed_ratio': job.get('speed_ratio'),
        'error': job.get('error')
    }), 200


@app.route('/api/download/transcript/<job_id>', methods=['GET'])
def download_transcript(job_id):
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] == 'failed':
        return jsonify({'error': f"Processing failed: {job.get('error', 'Unknown error')}"}), 400
    
    if job['stage'] != 'completed':
        return jsonify({'error': 'Processing not completed yet'}), 400
    
    format_type = request.args.get('format', 'txt')
    
    if format_type == 'txt':
        transcript_path = job.get('transcript_path')
        
        if transcript_path and os.path.exists(transcript_path):
            return send_file(
                transcript_path,
                as_attachment=True,
                download_name=f'transcript_{job_id}.txt',
                mimetype='text/plain; charset=utf-8'
            )
        else:
            return jsonify({'error': 'Transcript file not found'}), 404
    
    elif format_type == 'pdf':
        transcript_pdf_path = job.get('transcript_pdf_path')
        
        if not transcript_pdf_path or not os.path.exists(transcript_pdf_path):
            return jsonify({'error': 'PDF not available'}), 400
        
        return send_file(
            transcript_pdf_path,
            as_attachment=True,
            download_name=f'transcript_{job_id}.pdf',
            mimetype='application/pdf'
        )
    
    else:
        return jsonify({'error': 'Invalid format'}), 400


@app.route('/api/download/summary/<job_id>', methods=['GET'])
def download_summary(job_id):
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] == 'failed':
        return jsonify({'error': f"Processing failed: {job.get('error', 'Unknown error')}"}), 400
    
    if job['stage'] != 'completed':
        return jsonify({'error': 'Processing not completed yet'}), 400
    
    format_type = request.args.get('format', 'txt')
    
    if format_type == 'txt':
        summary_path = job.get('summary_path')
        
        if summary_path and os.path.exists(summary_path):
            return send_file(
                summary_path,
                as_attachment=True,
                download_name=f'summary_{job_id}.txt',
                mimetype='text/plain; charset=utf-8'
            )
        else:
            return jsonify({'error': 'Summary file not found'}), 404
    
    elif format_type == 'pdf':
        summary_pdf_path = job.get('summary_pdf_path')
        
        if not summary_pdf_path or not os.path.exists(summary_pdf_path):
            return jsonify({'error': 'PDF not available'}), 400
        
        return send_file(
            summary_pdf_path,
            as_attachment=True,
            download_name=f'summary_{job_id}.pdf',
            mimetype='application/pdf'
        )
    
    else:
        return jsonify({'error': 'Invalid format'}), 400


@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_processing(job_id):
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['stage'] == 'completed':
        return jsonify({'error': 'Job already completed'}), 400
    
    update_job(job_id, {
        'status': 'cancelled',
        'stage': 'cancelled',
        'error': 'Cancelled by user'
    })
    
    return jsonify({
        'success': True,
        'message': 'Job cancelled successfully'
    }), 200


# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    return render_template('home.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413


# ========== RUN APPLICATION ==========

if __name__ == '__main__':
    print("=" * 70)
    print("🎬 VidSummarize - Multilingual Video Processing System")
    print("=" * 70)
    print("✅ Unicode Support: ENABLED")
    print("✅ Encoding: UTF-8")
    print(f"✅ Console Encoding: {sys.stdout.encoding}")
    print("=" * 70)
    
    if TRANSCRIBER_TYPE == "smart_v2":
        print("✅ Using Smart Transcriber v2")
        print("   • Intelligent audio profiling")
        print("   • Multilingual support (Hindi, Tamil, Telugu, etc.)")
        print("   • Adaptive model selection (tiny/base/small)")
        print("   • Silence-based chunking")
        print("   • Parallel processing (2-4 workers)")
        print("   • 50-65% faster than previous version")
        print("   • Strategies: direct, chunked, streaming")
    elif TRANSCRIBER_TYPE == "enhanced":
        print("✅ Using Enhanced Transcriber")
        print("   • Faster-Whisper backend")
        print("   • Performance modes: fast/balanced/accurate")
        print("   • 2-3x faster than OpenAI Whisper")
    else:
        print("⚠️  Using Legacy Transcriber")
        print("   • Consider upgrading to Smart Transcriber v2")
    
    print("=" * 70)
    print("\n🚀 Starting Flask server...")
    print("   Access at: http://localhost:5000")
    print("   Upload videos or paste YouTube URLs")
    print("   Supports multilingual content")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
