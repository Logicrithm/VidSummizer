"""
VidSummizer - Video Summarization Application
Updated with YouTube Caption Support for Online Version
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

# Import your existing modules
from orchestrator import process_video_workflow
from summarizer import summarize_text
from pdf_generator import generate_pdf
from transcriber import transcribe_audio

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

# Store job status in memory (consider using Redis for production)
jobs = {}


# ============================================================================
# YOUTUBE CAPTION HELPERS
# ============================================================================

def extract_video_id(url):
    """
    Extract YouTube video ID from various URL formats
    Supports: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/embed/ID
    """
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
        # Try to get transcript (auto-generated or manual)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all caption segments into single text
        full_text = " ".join([entry["text"] for entry in transcript_list])
        
        # Clean up text
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


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'VidSummizer API is running',
        'version': '2.0',
        'features': {
            'youtube_captions': True,
            'local_transcription': True
        }
    })


@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    """
    Process YouTube video using captions (online) or full transcription (local)
    """
    youtube_url = request.form.get('youtube_url', '').strip()
    
    if not youtube_url:
        return render_template("error.html", 
            message="Please provide a valid YouTube URL",
            show_upload_option=True)
    
    # Validate YouTube URL format
    if not extract_video_id(youtube_url):
        return render_template("error.html",
            message="Invalid YouTube URL format. Please use a valid YouTube video link.",
            show_upload_option=True)
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Try to get captions first (for online version)
    captions, error = get_youtube_captions(youtube_url)
    
    if captions:
        # SUCCESS: We have captions, use fast path
        try:
            # Update job status
            jobs[job_id] = {
                'status': 'processing',
                'stage': 'caption_extraction',
                'progress': 30
            }
            
            # Generate summary using your existing summarizer
            jobs[job_id].update({
                'stage': 'summarization',
                'progress': 60
            })
            
            summary = summarize_text(captions)
            
            # Generate PDF using your existing PDF generator
            jobs[job_id].update({
                'stage': 'pdf_generation',
                'progress': 80
            })
            
            pdf_filename = f"youtube_summary_{job_id}.pdf"
            pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
            
            generate_pdf(
                transcript=captions,
                summary=summary,
                output_path=pdf_path,
                title="YouTube Video Summary"
            )
            
            # Update job status to completed
            jobs[job_id].update({
                'status': 'completed',
                'stage': 'done',
                'progress': 100,
                'transcript': captions,
                'summary': summary,
                'pdf_path': pdf_path,
                'pdf_filename': pdf_filename
            })
            
            # Redirect to results page
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
        # NO CAPTIONS: Show error for online version
        return render_template("error.html",
            message=f"⚠️ {error}.<br><br>"
                   "This video cannot be processed in the online version. "
                   "Please download the video and use the 'Upload Video' option below.",
            show_upload_option=True)


@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handle video file upload and processing
    Uses full transcription pipeline (local version)
    """
    # Check if file was uploaded
    if 'video_file' not in request.files:
        return render_template("error.html",
            message="No file uploaded. Please select a video file.")
    
    file = request.files['video_file']
    
    # Check if filename is empty
    if file.filename == '':
        return render_template("error.html",
            message="No file selected. Please choose a video file.")
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return render_template("error.html",
            message=f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{job_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'processing',
            'stage': 'upload',
            'progress': 10,
            'filename': filename
        }
        
        # Redirect to processing page
        return redirect(url_for('processing', job_id=job_id, filepath=filepath))
        
    except Exception as e:
        return render_template("error.html",
            message=f"Error uploading file: {str(e)}")


@app.route('/processing/<job_id>')
def processing(job_id):
    """Processing status page"""
    filepath = request.args.get('filepath')
    
    # Start processing in background (in production, use Celery or similar)
    # For now, process synchronously
    try:
        # Update status
        jobs[job_id].update({
            'stage': 'extracting_audio',
            'progress': 20
        })
        
        # Process video using your orchestrator
        result = process_video_workflow(filepath)
        
        jobs[job_id].update({
            'stage': 'transcribing',
            'progress': 50
        })
        
        jobs[job_id].update({
            'stage': 'summarizing',
            'progress': 70
        })
        
        jobs[job_id].update({
            'stage': 'generating_pdf',
            'progress': 90
        })
        
        # Mark as completed
        jobs[job_id].update({
            'status': 'completed',
            'stage': 'done',
            'progress': 100,
            'transcript': result['transcript'],
            'summary': result['summary'],
            'pdf_path': result['pdf_path'],
            'pdf_filename': result['pdf_filename']
        })
        
    except Exception as e:
        jobs[job_id] = {
            'status': 'failed',
            'error': str(e)
        }
    
    return render_template('processing.html', job_id=job_id)


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
    
    # Run Flask app
    # debug=False for production, host="0.0.0.0" to accept external connections
    app.run(host="0.0.0.0", port=port, debug=False)
