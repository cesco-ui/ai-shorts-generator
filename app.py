#!/usr/bin/env python3
"""
🚀 Production AI Shorts Generator API for Render
Fixed: OpenAI API v1.0+, Health endpoints, Google Drive downloads
"""

import os
import json
import tempfile
import shutil
import re
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# Disable Flask's automatic .env loading
os.environ['FLASK_SKIP_DOTENV'] = '1'

from flask import Flask, request, jsonify, send_file, redirect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

class ProductionShortsGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"📁 Working directory: {self.temp_dir}")
        
        # Import heavy libraries only when needed
        try:
            import yt_dlp
            from openai import OpenAI
            from moviepy.editor import VideoFileClip
            import requests
            
            self.yt_dlp = yt_dlp
            self.VideoFileClip = VideoFileClip
            self.requests = requests
            
            # Initialize OpenAI client (NEW API v1.0+)
            if self.api_key:
                self.openai_client = OpenAI(api_key=self.api_key)
            else:
                self.openai_client = None
                
        except ImportError as e:
            logger.error(f"❌ Missing required package: {e}")
            raise
    
    def extract_google_drive_id(self, url):
        """Extract file ID from Google Drive URL"""
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/open\?id=([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def download_youtube_video(self, url):
        """Download YouTube video using yt-dlp"""
        try:
            logger.info(f"📥 Downloading YouTube: {url}")
            
            output_path = os.path.join(self.temp_dir, "%(title)s.%(ext)s")
            
            ydl_opts = {
                'format': 'best[height<=720]/best',  # Max 720p for bandwidth
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                # Add user agent to avoid bot detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            }
            
            with self.yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Find the downloaded file
                for file in os.listdir(self.temp_dir):
                    if file.endswith(('.mp4', '.webm', '.mkv')):
                        video_path = os.path.join(self.temp_dir, file)
                        file_size = os.path.getsize(video_path) / (1024*1024)  # MB
                        logger.info(f"✅ Downloaded: {file_size:.1f} MB")
                        
                        return {
                            "success": True,
                            "path": video_path,
                            "title": info.get('title', 'Unknown'),
                            "duration": info.get('duration', 0),
                            "size_mb": file_size
                        }
                        
                return {"success": False, "error": "No video file found after download"}
                
        except Exception as e:
            logger.error(f"❌ YouTube download failed: {e}")
            return {"success": False, "error": str(e)}
    
    def download_google_drive_video(self, url):
        """Download Google Drive video with better handling"""
        try:
            file_id = self.extract_google_drive_id(url)
            if not file_id:
                return {"success": False, "error": "Could not extract Google Drive file ID"}
            
            logger.info(f"📥 Downloading Google Drive: {file_id}")
            
            # Use direct download URL with better headers
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # First request to get download confirmation if needed
            session = self.requests.Session()
            response = session.get(download_url, headers=headers, stream=True)
            
            # Check if we need to confirm download for large files
            if 'confirm=' in response.text and 'download_warning' in response.text:
                # Extract confirmation token
                confirm_match = re.search(r'confirm=([^&]+)', response.text)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(download_url, headers=headers, stream=True)
            
            # Check if we got a valid response
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}: Failed to download from Google Drive"}
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                return {"success": False, "error": "Got HTML page instead of video file. Check if file is publicly accessible."}
            
            # Save the file
            video_path = os.path.join(self.temp_dir, f"gdrive_{file_id}.mp4")
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Check file size
            file_size = os.path.getsize(video_path) / (1024*1024)  # MB
            logger.info(f"✅ Downloaded: {file_size:.1f} MB")
            
            if file_size < 0.1:  # Less than 100KB is suspicious
                return {"success": False, "error": "Downloaded file is too small, might be an error page"}
            
            return {
                "success": True,
                "path": video_path,
                "title": f"Google Drive Video {file_id}",
                "duration": 0,  # We'll get this from video analysis
                "size_mb": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ Google Drive download failed: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_with_ai(self, video_path):
        """Analyze video with AI and generate title/hook - FIXED for OpenAI v1.0+"""
        try:
            logger.info("🤖 AI analyzing video and generating title/hook...")
            
            if not self.openai_client:
                # Fallback values when no OpenAI key
                return {
                    "start_time": 10,
                    "end_time": 70,
                    "duration": 60,
                    "title": "AI-Generated Short Clip",
                    "hook": "Check out this amazing moment!",
                    "analysis": "AI analysis unavailable - no API key provided",
                    "confidence": 0.8
                }
            
            # Get basic video info
            try:
                with self.VideoFileClip(video_path) as video:
                    total_duration = video.duration
            except:
                total_duration = 300  # Fallback to 5 minutes
            
            # Create AI prompt
            prompt = f"""
            Analyze this {total_duration:.0f}-second video for the best 60-second clip for social media.
            
            Respond with ONLY a JSON object:
            {{
                "start_time": <seconds>,
                "end_time": <seconds>, 
                "title": "<engaging title max 60 chars>",
                "hook": "<compelling hook max 100 chars>",
                "analysis": "<why this segment is engaging>",
                "confidence": <0.0-1.0>
            }}
            
            Make the clip 45-75 seconds long. Choose the most engaging part.
            """
            
            # FIXED: Use new OpenAI API syntax
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert video editor who creates viral social media clips."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Parse response
            ai_response = response.choices[0].message.content.strip()
            
            # Clean JSON response
            ai_response = ai_response.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(ai_response)
                
                # Validate and adjust times
                start_time = max(0, min(result.get('start_time', 10), total_duration - 60))
                end_time = min(start_time + 60, total_duration, result.get('end_time', start_time + 60))
                
                return {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "title": result.get('title', 'AI-Generated Clip')[:60],
                    "hook": result.get('hook', 'Amazing content!')[:100],
                    "analysis": result.get('analysis', 'AI-selected highlight'),
                    "confidence": result.get('confidence', 0.85)
                }
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("AI returned invalid JSON, using fallback")
                start_time = max(0, min(30, total_duration - 60))
                return {
                    "start_time": start_time,
                    "end_time": start_time + 60,
                    "duration": 60,
                    "title": "AI-Generated Short Clip",
                    "hook": "Don't miss this amazing moment!",
                    "analysis": "AI-selected highlight segment",
                    "confidence": 0.75
                }
                
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            # Return fallback values
            return {
                "start_time": 10,
                "end_time": 70,
                "duration": 60,
                "title": "Short Clip",
                "hook": "Check this out!",
                "analysis": f"AI analysis failed: {str(e)}",
                "confidence": 0.5
            }
    
    def create_clip(self, video_path, start_time, end_time):
        """Create video clip using MoviePy"""
        try:
            logger.info(f"✂️ Creating clip: {start_time}s to {end_time}s")
            
            with self.VideoFileClip(video_path) as video:
                # Create clip
                clip = video.subclip(start_time, end_time)
                
                # Output path
                clip_path = os.path.join(self.temp_dir, "ai_short.mp4")
                
                # Write video file
                clip.write_videofile(
                    clip_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(self.temp_dir, 'temp-audio.m4a'),
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                # Check if file was created successfully
                if os.path.exists(clip_path):
                    file_size = os.path.getsize(clip_path) / (1024*1024)  # MB
                    logger.info(f"✅ Clip created: {file_size:.3f} MB")
                    return {"success": True, "path": clip_path, "size_mb": file_size}
                else:
                    return {"success": False, "error": "Clip file was not created"}
                    
        except Exception as e:
            logger.error(f"❌ Clip creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def process_video(self, url, video_type="auto"):
        """Main processing function"""
        try:
            logger.info(f"🎬 Processing {video_type}: {url}")
            
            # Normalize video_type values for backward compatibility
            if video_type == "drive":
                video_type = "google_drive"
            
            # Auto-detect video type
            if video_type == 'auto':
                if "youtube.com" in url or "youtu.be" in url:
                    video_type = "youtube"
                elif "drive.google.com" in url:
                    video_type = "google_drive"
                else:
                    return {"error": "Unsupported video URL. Please use YouTube or Google Drive links."}
            
            # Download video
            if video_type == "youtube":
                download_result = self.download_youtube_video(url)
            elif video_type in ["google_drive", "drive"]:
                download_result = self.download_google_drive_video(url)
            else:
                return {"error": f"Unsupported video type: {video_type}"}
            
            if not download_result["success"]:
                return {"error": f"Download failed: {download_result['error']}"}
            
            video_path = download_result["path"]
            
            # AI analysis
            ai_result = self.analyze_with_ai(video_path)
            
            # Create clip
            clip_result = self.create_clip(video_path, ai_result["start_time"], ai_result["end_time"])
            
            if not clip_result["success"]:
                return {"error": f"Clip creation failed: {clip_result['error']}"}
            
            # Generate download URL
            download_url = f"/api/download/ai_short.mp4"
            
            return {
                "success": True,
                "original_video": {
                    "url": url,
                    "title": download_result.get("title", "Unknown"),
                    "source": video_type,
                    "duration": download_result.get("duration", 0),
                    "size_mb": download_result.get("size_mb", 0)
                },
                "ai_analysis": {
                    "start_time": ai_result["start_time"],
                    "end_time": ai_result["end_time"],
                    "duration": ai_result["duration"],
                    "analysis": ai_result["analysis"],
                    "confidence": ai_result["confidence"]
                },
                "generated_content": {
                    "title": ai_result["title"],
                    "hook": ai_result["hook"]
                },
                "clip": {
                    "filename": "ai_short.mp4",
                    "size_mb": clip_result["size_mb"],
                    "duration": ai_result["duration"]
                },
                "download_url": download_url,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

# Initialize generator
shorts_generator = None

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "service": "AI Shorts Generator",
        "version": "3.1.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "generate": "/api/generate-short",
            "download": "/api/download/<filename>"
        },
        "supported_sources": ["youtube", "google_drive", "drive"],
        "features": [
            "AI-powered clip selection",
            "Auto-generated titles and hooks", 
            "Smart video editing",
            "Multiple video sources"
        ]
    })

@app.route('/health')
def health_alias():
    """Health check endpoint alias - FIXED"""
    return redirect('/api/health')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Shorts Generator", 
        "version": "3.1.0",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(OPENAI_API_KEY)
    })

@app.route('/api/generate-short', methods=['POST'])
def generate_short():
    """Generate short video clip from long video"""
    global shorts_generator
    
    try:
        logger.info("\n🚀 New shorts generation request received!")
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        video_url = data.get('video_url')
        video_type = data.get('video_type', 'auto')
        
        if not video_url:
            return jsonify({"error": "video_url is required"}), 400
        
        logger.info(f"📹 Processing: {video_url}")
        
        # Initialize generator if needed
        if not shorts_generator:
            shorts_generator = ProductionShortsGenerator(OPENAI_API_KEY)
        
        # Process video
        result = shorts_generator.process_video(video_url, video_type)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        logger.info("✅ Processing complete!")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up after each request
        if shorts_generator:
            shorts_generator.cleanup()
            shorts_generator = None

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated video file"""
    try:
        logger.info(f"📤 Serving file: {filename}")
        
        # Security: only allow specific filenames
        allowed_files = ['ai_short.mp4']
        if filename not in allowed_files:
            return jsonify({"error": "File not found"}), 404
        
        # Find the most recent temp directory with this file
        temp_base = tempfile.gettempdir()
        for item in os.listdir(temp_base):
            item_path = os.path.join(temp_base, item)
            if os.path.isdir(item_path) and item.startswith('tmp'):
                file_path = os.path.join(item_path, filename)
                if os.path.exists(file_path):
                    return send_file(file_path, as_attachment=True, download_name=filename)
        
        return jsonify({"error": "File not found"}), 404
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Production AI Shorts Generator...")
    print("📍 Health check: http://localhost:5000/api/health")  
    print("🎬 Generate short: POST http://localhost:5000/api/generate-short")
    print("💡 Fixed: OpenAI API v1.0+, health endpoints, downloads")
    print("🔥 Ready for production deployment!")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
