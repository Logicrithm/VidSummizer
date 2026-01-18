"""
VidSummizer - Video Summarization Application
Updated with YouTube Caption Support for Online Version
Compatible with your orchestrator.py class-based approach
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store job status in memory
jobs = {}


# ============================================================================
# IMPORTS - Compatible with your project structure
# ============================================================================

# Import orchestrator (your class-based summarization pipeline)
try:
    from orchestrator import Orchestrator, summarize, summarize_with_stats
    ORCHESTRATOR_AVAILABLE = True
    print("✓ Loaded: Orchestrator from orchestrator.py")
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    print(f"⚠ WARNING: Could not import Orchestrator: {e}")

# Import video processor (for handling uploaded videos)
try:
    from video_processor import extract_audio
    VIDEO_PROCESSOR_AVAILABLE = True
    print("✓ Loaded: extract_audio from video_processor")
except ImportError:
    VIDEO_PROCESSOR_AVAILABLE = False
    print("⚠ WARNING: video_processor not available")

# Import transcriber (for local video transcription)
try:
    from transcriber import transcribe_audio
    TRANSCRIBER_AVAILABLE = True
    print("✓ Loaded: transcribe_audio from transcriber")
except ImportError:
    try:
        from smart_transcriber import SmartTranscriber
        transcriber = SmartTranscriber()
        transcribe_audio = transcriber.transcribe
        TRANSCRIBER_AVAILABLE = True
        print("✓ Loaded: SmartTranscriber from smart_transcriber")
    except ImportError:
        TRANSCRIBER_AVAILABLE = False
        print("⚠ WARNING: transcriber not available")

# Import PDF generator
try:
    from pdf_generator import generate_pdf
    PDF_GENERATOR_AVAILABLE = True
    print("✓ Loaded: generate_pdf from pdf_generator")
except ImportError:
    PDF_GENERATOR_AVAILABLE = False
    print("⚠ WARNING: pdf_generator not available")


# ============================================================================
# YOUTUBE CAPTION HELPERS
# ============================================================================

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_captions(url):
    """
    Fetch captions from YouTube video
    Returns: (captions_text, error_message)
    """
    video_id = extract_video_id(url)
    
    if not video_id:
        return None, "Invalid YouTube URL format"
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
        full_text = full_text.replace('\n', ' ').strip()
        return full_text, None
        
    except TranscriptsDisabled:
        return None, "This video has captions disabled by the creator"
    except NoTranscriptFound:
        return None, "No captions available for this video"
    except Exception as e:
        return None, f"Error fetching captions: {str(e)}"


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def create_simple_text_output(transcript, summary, output_path):
    """
    Fallback document generator if pdf_generator doesn't work
    Creates a formatted text file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VIDEO SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write("SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(summary + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("FULL TRANSCRIPT:\n")
            f.write("=" * 80 + "\n")
            f.write(transcript + "\n")
        return True
    except Exception as e:
        print(f"Error creating text document: {str(e)}")
        return False


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('home.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'VidSummizer API is running',
        'version': '2.0',
        'features': {
            'youtube_captions': True,
            'local_transcription': TRANSCRIBER_AVAILABLE,
            'video_processing': VIDEO_PROCESSOR_AVAILABLE,
            'summarization': ORCHESTRATOR_AVAILABLE,
            'pdf_generation': PDF_GENERATOR_AVAILABLE
        }
    })


@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    """Process YouTube video using captions (online version)"""
    youtube_url = request.form.get('youtube_url', '').strip()
    
    if not youtube_url:
        return render_template("error.html", 
            message="Please provide a valid YouTube URL",
            show_upload_option=True)
    
    if not extract_video_id(youtube_url):
        return render_template("error.html",
            message="Invalid YouTube URL format. Please use a valid YouTube video link.",
            show_upload_option=True)
    
    job_id = str(uuid.uuid4())
    
    # Try to get captions
    captions, error = get_youtube_captions(youtube_url)
    
    if captions:
        try:
            jobs[job_id] = {
                'status': 'processing',
                'stage': 'caption_extraction',
                'progress': 30
            }
            
            # Generate summary using your Orchestrator
            if ORCHESTRATOR_AVAILABLE:
                jobs[job_id].update({
                    'stage': 'summarization',
                    'progress': 60
                })
                
                # Use your orchestrator's summarize function
                # It returns just the summary text
                summary = summarize(captions, mode='balanced', target_words=300)
                
            else:
                # Fallback: simple summary (first 300 words)
                words = captions.split()
                summary = ' '.join(words[:300]) + "..." if len(words) > 300 else captions
            
            # Generate PDF/document
            jobs[job_id].update({
                'stage': 'pdf_generation',
                'progress': 80
            })
            
            pdf_filename = f"youtube_summary_{job_id}.pdf"
            pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
            
            if PDF_GENERATOR_AVAILABLE:
                try:
                    generate_pdf(
                        transcript=captions,
                        summary=summary,
                        output_path=pdf_path,
                        title="YouTube Video Summary"
                    )
                except Exception as e:
                    # If PDF generation fails, create text file
                    print(f"PDF generation failed: {str(e)}, creating text file instead")
                    pdf_filename = f"youtube_summary_{job_id}.txt"
                    pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
                    create_simple_text_output(captions, summary, pdf_path)
            else:
                # Fallback to text file
                pdf_filename = f"youtube_summary_{job_id}.txt"
                pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
                create_simple_text_output(captions, summary, pdf_path)
            
            # Mark as completed
            jobs[job_id].update({
                'status': 'completed',
                'stage': 'done',
                'progress': 100,
                'transcript': captions,
                'summary': summary,
                'pdf_path': pdf_path,
                'pdf_filename': pdf_filename
            })
            
            return redirect(url_for('results', job_id=job_id))
            
        except Exception as e:
            jobs[job_id] = {
                'status': 'failed',
                'error': str(e)
            }
            return render_template("error.html",
                message=f"Error processing captions: {str(e)}",
                show_upload_option=True)
    else:
        # No captions available
        return render_template("error.html",
            message=f"⚠️ {error}.<br><br>"
                   "This video cannot be processed in the online version. "
                   "Please download the video and use the 'Upload Video' option below.",
            show_upload_option=True)


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload and processing (local version)"""
    if 'video_file' not in request.files:
        return render_template("error.html",
            message="No file uploaded. Please select a video file.")
    
    file = request.files['video_file']
    
    if file.filename == '':
        return render_template("error.html",
            message="No file selected. Please choose a video file.")
    
    if not allowed_file(file.filename):
        return render_template("error.html",
            message=f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    
    # Check if required modules are available
    if not VIDEO_PROCESSOR_AVAILABLE or not TRANSCRIBER_AVAILABLE or not ORCHESTRATOR_AVAILABLE:
        return render_template("error.html",
            message="Video upload processing is not available in this deployment. "
                   "Missing required modules: video processor, transcriber, or orchestrator.")
    
    try:
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        unique_filename = f"{job_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        jobs[job_id] = {
            'status': 'processing',
            'stage': 'upload',
            'progress': 10,
            'filename': filename,
            'filepath': filepath
        }
        
        return redirect(url_for('processing', job_id=job_id))
        
    except Exception as e:
        return render_template("error.html",
            message=f"Error uploading file: {str(e)}")


@app.route('/processing/<job_id>')
def processing(job_id):
    """Processing status page - processes the video in background"""
    
    if job_id not in jobs:
        return render_template('error.html',
            message="Job not found. Please try uploading again.")
    
    # Get filepath from job
    filepath = jobs[job_id].get('filepath')
    
    if not filepath:
        return render_template('error.html',
            message="Invalid job configuration.")
    
    try:
        # Step 1: Extract audio from video
        jobs[job_id].update({
            'stage': 'extracting_audio',
            'progress': 20
        })
        
        audio_path = filepath.replace(os.path.splitext(filepath)[1], '.wav')
        extract_audio(filepath, audio_path)
        
        # Step 2: Transcribe audio
        jobs[job_id].update({
            'stage': 'transcribing',
            'progress': 50
        })
        
        transcript = transcribe_audio(audio_path)
        
        # Step 3: Summarize using your Orchestrator
        jobs[job_id].update({
            'stage': 'summarizing',
            'progress': 70
        })
        
        summary = summarize(transcript, mode='balanced', target_words=300)
        
        # Step 4: Generate PDF
        jobs[job_id].update({
            'stage': 'generating_pdf',
            'progress': 90
        })
        
        pdf_filename = f"video_summary_{job_id}.pdf"
        pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
        
        if PDF_GENERATOR_AVAILABLE:
            try:
                generate_pdf(
                    transcript=transcript,
                    summary=summary,
                    output_path=pdf_path,
                    title="Video Summary"
                )
            except Exception as e:
                # Fallback to text file
                pdf_filename = f"video_summary_{job_id}.txt"
                pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
                create_simple_text_output(transcript, summary, pdf_path)
        else:
            # Use text file
            pdf_filename = f"video_summary_{job_id}.txt"
            pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
            create_simple_text_output(transcript, summary, pdf_path)
        
        # Clean up temporary files
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        
        # Mark as completed
        jobs[job_id].update({
            'status': 'completed',
            'stage': 'done',
            'progress': 100,
            'transcript': transcript,
            'summary': summary,
            'pdf_path': pdf_path,
            'pdf_filename': pdf_filename
        })
        
        return redirect(url_for('results', job_id=job_id))
        
    except Exception as e:
        jobs[job_id] = {
            'status': 'failed',
            'error': str(e)
        }
        return render_template('error.html',
            message=f"Error processing video: {str(e)}")


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """API endpoint to check job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'progress': 100,
            'redirect_url': url_for('results', job_id=job_id)
        })
    elif job['status'] == 'failed':
        return jsonify({
            'status': 'failed',
            'error': job.get('error', 'Unknown error')
        })
    else:
        return jsonify({
            'status': 'processing',
            'stage': job.get('stage', 'unknown'),
            'progress': job.get('progress', 0)
        })


@app.route('/results/<job_id>')
def results(job_id):
    """Display results page"""
    if job_id not in jobs:
        return render_template("error.html",
            message="Job not found. Please try again.")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        if job['status'] == 'failed':
            return render_template("error.html",
                message=f"Processing failed: {job.get('error', 'Unknown error')}")
        else:
            return redirect(url_for('processing', job_id=job_id))
    
    return render_template('results.html',
        job_id=job_id,
        transcript=job['transcript'],
        summary=job['summary'],
        pdf_filename=job['pdf_filename'])


@app.route('/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    """Download generated files"""
    if job_id not in jobs:
        return "File not found", 404
    
    job = jobs[job_id]
    
    if file_type == 'pdf':
        return send_file(
            job['pdf_path'],
            as_attachment=True,
            download_name=job['pdf_filename']
        )
    elif file_type == 'transcript':
        # Generate transcript text file
        transcript_path = os.path.join(app.config['OUTPUT_FOLDER'], f'transcript_{job_id}.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(job['transcript'])
        return send_file(
            transcript_path,
            as_attachment=True,
            download_name=f'transcript_{job_id}.txt'
        )
    elif file_type == 'summary':
        # Generate summary text file
        summary_path = os.path.join(app.config['OUTPUT_FOLDER'], f'summary_{job_id}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(job['summary'])
        return send_file(
            summary_path,
            as_attachment=True,
            download_name=f'summary_{job_id}.txt'
        )
    else:
        return "Invalid file type", 400


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return render_template("error.html",
        message=f"File too large. Maximum file size is {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return render_template("error.html",
        message="Internal server error. Please try again later.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Get port from environment variable (for deployment platforms like Render)
    port = int(os.environ.get("PORT", 5000))
    
    # Print startup info
    print("\n" + "=" * 70)
    print("VidSummizer Starting...")
    print("=" * 70)
    print(f"YouTube Caption Support: ✓")
    print(f"Orchestrator: {'✓' if ORCHESTRATOR_AVAILABLE else '✗'}")
    print(f"Video Processor: {'✓' if VIDEO_PROCESSOR_AVAILABLE else '✗'}")
    print(f"Transcriber: {'✓' if TRANSCRIBER_AVAILABLE else '✗'}")
    print(f"PDF Generator: {'✓' if PDF_GENERATOR_AVAILABLE else '✗'}")
    print("=" * 70)
    print(f"Running on port {port}")
    print("=" * 70 + "\n")
    
    # Run Flask app
    app.run(host="0.0.0.0", port=port, debug=False)
