"""
VidSummarize - Video Processing Module
Handles video download, audio extraction, and file processing
"""

import os
import subprocess
import yt_dlp
import ffmpeg
from datetime import datetime


class VideoProcessor:
    """Handles all video and audio processing operations"""
    
    def __init__(self, upload_folder='uploads', output_folder='outputs'):
        self.upload_folder = upload_folder
        self.output_folder = output_folder
        
        # Create folders if they don't exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path for output audio file (optional)
            
        Returns:
            dict: Result with audio_path and metadata
        """
        try:
            print(f"[VideoProcessor] Extracting audio from: {video_path}")
            
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(self.output_folder, f"{base_name}_audio.wav")
            
            # Extract audio using ffmpeg-python
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Get video metadata
            metadata = self._get_video_metadata(video_path)
            
            print(f"[VideoProcessor] Audio extracted successfully: {output_path}")
            
            return {
                'success': True,
                'audio_path': output_path,
                'duration': metadata.get('duration', 0),
                'size': os.path.getsize(output_path)
            }
            
        except ffmpeg.Error as e:
            error_message = str(e)
            if isinstance(getattr(e, 'stderr', None), str) and e.stderr:
                error_message = e.stderr
            print(f"[VideoProcessor] FFmpeg error: {error_message}")
            return {
                'success': False,
                'error': f"Audio extraction failed: {error_message}"
            }
        except Exception as e:
            print(f"[VideoProcessor] Error: {str(e)}")
            return {
                'success': False,
                'error': f"Audio extraction failed: {str(e)}"
            }
    
    def download_youtube_video(self, youtube_url, output_filename=None):
        """
        Download YouTube video and extract audio
        
        Args:
            youtube_url (str): YouTube video URL
            output_filename (str): Custom filename (optional)
            
        Returns:
            dict: Result with audio_path and metadata
        """
        try:
            print(f"[VideoProcessor] Downloading YouTube video: {youtube_url}")
            
            # Generate filename
            if output_filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"youtube_{timestamp}"
            
            video_path = os.path.join(self.upload_folder, f"{output_filename}.mp4")
            audio_path = os.path.join(self.output_folder, f"{output_filename}_audio.wav")
            
            # yt-dlp options
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': video_path,
                'cookiefile': r"C:\Users\KIIT\Downloads\www.youtube.com_cookies.txt",
                'quiet': False,
                'no_warnings': False,
            }
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                video_title = info.get('title', 'Unknown')
                video_duration = info.get('duration', 0)
            
            print(f"[VideoProcessor] Video downloaded: {video_title}")
            
            # Extract audio from downloaded video
            audio_result = self.extract_audio_from_video(video_path, audio_path)
            
            if audio_result['success']:
                audio_result['video_path'] = video_path
                audio_result['video_title'] = video_title
                audio_result['duration'] = video_duration
                return audio_result
            else:
                return audio_result
                
        except yt_dlp.utils.DownloadError as e:
            print(f"[VideoProcessor] YouTube download error: {str(e)}")
            return {
                'success': False,
                'error': f"YouTube download failed: {str(e)}"
            }
        except Exception as e:
            print(f"[VideoProcessor] Error: {str(e)}")
            return {
                'success': False,
                'error': f"YouTube processing failed: {str(e)}"
            }
    
    def _get_video_metadata(self, video_path):
        """
        Get video metadata using ffprobe
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video metadata
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            duration = float(probe['format']['duration'])
            
            metadata = {
                'duration': duration,
                'width': video_info.get('width'),
                'height': video_info.get('height'),
                'fps': eval(video_info.get('r_frame_rate', '0/1')),
                'video_codec': video_info.get('codec_name'),
                'audio_codec': audio_info.get('codec_name') if audio_info else None,
                'size': int(probe['format']['size'])
            }
            
            return metadata
            
        except Exception as e:
            print(f"[VideoProcessor] Metadata extraction error: {str(e)}")
            return {'duration': 0, 'size': 0}
    
    def get_audio_duration(self, audio_path):
        """
        Get duration of audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            float: Duration in seconds
        """
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            print(f"[VideoProcessor] Duration check error: {str(e)}")
            return 0
    
    def cleanup_file(self, file_path):
        """
        Delete a file safely
        
        Args:
            file_path (str): Path to file to delete
            
        Returns:
            bool: Success status
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[VideoProcessor] Cleaned up: {file_path}")
                return True
            return False
        except Exception as e:
            print(f"[VideoProcessor] Cleanup error: {str(e)}")
            return False


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("VidSummarize - Video Processor Test")
    print("=" * 60)
    
    processor = VideoProcessor()
    
    # Test 1: Check if FFmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print("✅ FFmpeg is installed")
        print(result.stdout.split('\n')[0])
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH")
    
    print("\n" + "=" * 60)
    print("Video Processor module loaded successfully!")
    print("=" * 60)
