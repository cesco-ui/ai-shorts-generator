#!/usr/bin/env python3
"""
🚀 Production AI Shorts Generator API for Render
Real video processing with yt-dlp, MoviePy, and OpenAI GPT-4
Now includes AI-generated titles and hooks!
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

from flask import Flask, request, jsonify, send_file

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
            import openai
            from moviepy.editor import VideoFileClip
            import requests
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            
            self.yt_dlp = yt_dlp
            self.openai = openai
            self.VideoFileClip = VideoFileClip
            self.requests = requests
            
            # Set OpenAI API key
            if self.api_key:
                openai.api_key = self.api_key
                
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
            }
            
            with self.yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Find the downloaded file
                for file in os.listdir(self.temp_dir):
                    if file.endswith(('.mp4', '.webm', '.mkv')):
                        video_path = os.path.join(self.temp_dir, file)
                        file_size = os.path.getsize(video_path) / (1024*1024)
                        logger.info(f"✅ Downloaded: {file} ({file_size:.1f} MB)")
                        
                        return {
                            "path": video_path,
                            "title": info.get('title', 'Unknown'),
                            "duration": info.get('duration', 0),
                            "file_size_mb": round(file_size, 2),
                            "description": info.get('description', '')[:500] + "..." if info.get('description', '') else ""
                        }
            
            return {"error": "No video file found after download"}
            
        except Exception as e:
            logger.error(f"❌ YouTube download failed: {e}")
            return {"error": f"YouTube download failed: {str(e)}"}
    
    def download_google_drive_video(self, url):
        """Download video from Google Drive"""
        try:
            file_id = self.extract_google_drive_id(url)
            if not file_id:
                return {"error": "Could not extract Google Drive file ID"}
            
            logger.info(f"📥 Downloading Google Drive: {file_id}")
            
            # Use direct download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = self.requests.get(download_url, stream=True)
            if response.status_code != 200:
                return {"error": f"Failed to download: HTTP {response.status_code}"}
            
            # Save to temp file
            video_path = os.path.join(self.temp_dir, f"gdrive_{file_id}.mp4")
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(video_path) / (1024*1024)
            logger.info(f"✅ Downloaded: {file_size:.1f} MB")
            
            return {
                "path": video_path,
                "title": f"Google Drive Video {file_id}",
                "duration": 0,  # Will be detected by MoviePy
                "file_size_mb": round(file_size, 2),
                "description": "Google Drive video content"
            }
            
        except Exception as e:
            logger.error(f"❌ Google Drive download failed: {e}")
            return {"error": f"Google Drive download failed: {str(e)}"}
    
    def analyze_video_with_ai(self, video_path, video_info):
        """Use GPT-4 to analyze video, find best clip, and generate title/hook"""
        try:
            logger.info("🤖 AI analyzing video and generating title/hook...")
            
            # Get video duration using MoviePy
            try:
                with self.VideoFileClip(video_path) as clip:
                    duration = clip.duration
            except:
                duration = video_info.get('duration', 300)  # Fallback
            
            # Create enhanced analysis prompt
            prompt = f"""
            Analyze this video and create the perfect YouTube Short package.
            
            Video Details:
            - Title: {video_info.get('title', 'Unknown')}
            - Duration: {duration} seconds
            - File size: {video_info.get('file_size_mb', 0)} MB
            - Description: {video_info.get('description', 'No description')}
            
            Please provide:
            1. Best 60-second segment (start and end times)
            2. Catchy YouTube Short title (8-15 words max)
            3. Engaging hook/description for the clip
            4. Brief reason for segment selection
            
            Requirements:
            - Segment must be exactly 60 seconds
            - Start time must be at least 10 seconds from beginning
            - End time must be at least 10 seconds before the end
            - Title should be clickbait but accurate
            - Hook should create curiosity and engagement
            - Choose the most viral-worthy segment
            
            Respond in JSON format:
            {{
                "start_time": 30.0,
                "end_time": 90.0,
                "generated_title": "This Secret Trick Will Blow Your Mind!",
                "generated_hook": "You won't believe what happens next... this simple technique changed everything!",
                "reason": "High energy section with key revelation",
                "confidence": 0.85
            }}
            """
            
            # Call OpenAI API
            if not self.api_key or self.api_key == 'demo-key-replace-in-render':
                # Fallback analysis for demo/testing
                mid_point = max(30, duration / 2)
                start_time = max(10, mid_point - 30)
                end_time = min(duration - 10, start_time + 60)
                
                return {
                    "start_time": start_time,
                    "end_time": end_time,
                    "generated_title": f"Amazing {video_info.get('title', 'Video')[:20]}... Clip!",
                    "generated_hook": "This incredible moment will leave you speechless! Don't miss this viral-worthy segment.",
                    "reason": "Middle section selected (demo mode)",
                    "confidence": 0.75,
                    "transcript_sample": "Demo analysis - no OpenAI key provided"
                }
            
            response = self.openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert viral content creator and video editor specializing in YouTube Shorts that get millions of views."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.8
            )
            
            # Parse AI response
            ai_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_start = ai_text.find('{')
                json_end = ai_text.rfind('}') + 1
                json_str = ai_text[json_start:json_end]
                
                analysis = json.loads(json_str)
                
                # Validate the analysis
                start_time = float(analysis.get('start_time', 30))
                end_time = float(analysis.get('end_time', 90))
                
                # Ensure valid timing
                if end_time - start_time > 65:  # Allow some flexibility
                    end_time = start_time + 60
                elif end_time - start_time < 55:
                    end_time = start_time + 60
                
                # Ensure within video bounds
                start_time = max(5, min(start_time, duration - 65))
                end_time = min(duration - 5, start_time + 60)
                
                result = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "generated_title": analysis.get('generated_title', f"Amazing {video_info.get('title', 'Video')[:20]}..."),
                    "generated_hook": analysis.get('generated_hook', "This incredible moment will leave you speechless!"),
                    "reason": analysis.get('reason', 'AI selected engaging segment'),
                    "confidence": analysis.get('confidence', 0.8),
                    "transcript_sample": ai_text[:100] + "..." if len(ai_text) > 100 else ai_text
                }
                
                logger.info(f"🎯 Found highlight: {start_time}s to {end_time}s")
                logger.info(f"📝 Generated title: {result['generated_title']}")
                return result
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"⚠️ AI response parsing failed: {e}")
                # Fallback to middle section with generated content
                mid_point = duration / 2
                start_time = max(10, mid_point - 30)
                end_time = min(duration - 10, start_time + 60)
                
                return {
                    "start_time": start_time,
                    "end_time": end_time,
                    "generated_title": f"Best Moments from {video_info.get('title', 'This Video')[:25]}...",
                    "generated_hook": "The most engaging part of this video - you have to see this!",
                    "reason": "Fallback: middle section selected",
                    "confidence": 0.6,
                    "transcript_sample": "AI analysis failed, using fallback with generated content"
                }
                
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            # Fallback analysis with generated content
            duration = video_info.get('duration', 300)
            start_time = max(10, duration * 0.3)  # Start at 30% through video
            end_time = min(duration - 10, start_time + 60)
            
            return {
                "start_time": start_time,
                "end_time": end_time,
                "generated_title": f"Incredible {video_info.get('title', 'Video')[:30]}... Moments",
                "generated_hook": "You won't believe what happens in this clip - prepare to be amazed!",
                "reason": "Fallback analysis due to AI error",
                "confidence": 0.5,
                "transcript_sample": f"Error: {str(e)}"
            }
    
    def create_short_clip(self, video_path, start_time, end_time):
        """Create the actual short clip using MoviePy"""
        try:
            logger.info(f"✂️ Creating clip: {start_time}s to {end_time}s")
            
            output_path = os.path.join(self.temp_dir, "ai_short.mp4")
            
            with self.VideoFileClip(video_path) as video:
                # Create the clip
                clip = video.subclip(start_time, end_time)
                
                # Resize for YouTube Shorts (9:16 aspect ratio)
                # Get current dimensions
                w, h = clip.size
                
                # Calculate new dimensions for 9:16
                target_ratio = 9/16
                current_ratio = w/h
                
                if current_ratio > target_ratio:
                    # Video is too wide, crop width
                    new_width = int(h * target_ratio)
                    x_center = w // 2
                    x1 = x_center - new_width // 2
                    x2 = x_center + new_width // 2
                    clip = clip.crop(x1=x1, x2=x2)
                else:
                    # Video is too tall, crop height
                    new_height = int(w / target_ratio)
                    y_center = h // 2
                    y1 = y_center - new_height // 2
                    y2 = y_center + new_height // 2
                    clip = clip.crop(y1=y1, y2=y2)
                
                # Resize to standard Short resolution
                clip = clip.resize(height=1920)
                
                # Write the final clip
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(self.temp_dir, 'temp_audio.m4a'),
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)
                logger.info(f"✅ Clip created: {file_size:.1f} MB")
                
                return {
                    "path": output_path,
                    "file_size_mb": round(file_size, 2),
                    "duration": end_time - start_time
                }
            else:
                return {"error": "Failed to create clip file"}
                
        except Exception as e:
            logger.error(f"❌ Clip creation failed: {e}")
            return {"error": f"Clip creation failed: {str(e)}"}
    
    def process_video(self, url, video_type):
        """Complete video processing pipeline"""
        try:
            logger.info(f"🎬 Processing {video_type}: {url}")
            
            # Step 1: Download video (accept both "drive" and "google_drive")
            if video_type == "youtube":
                download_result = self.download_youtube_video(url)
            elif video_type in ["google_drive", "drive"]:
                download_result = self.download_google_drive_video(url)
            else:
                return {"error": f"Unsupported video type: {video_type}. Supported types: youtube, google_drive, drive"}
            
            if "error" in download_result:
                return download_result
            
            video_path = download_result["path"]
            
            # Step 2: AI analysis with title/hook generation
            analysis = self.analyze_video_with_ai(video_path, download_result)
            
            # Step 3: Create short clip
            clip_result = self.create_short_clip(
                video_path, 
                analysis["start_time"], 
                analysis["end_time"]
            )
            
            if "error" in clip_result:
                return clip_result
            
            # Step 4: Return complete result with generated content
            return {
                "success": True,
                "original_video": {
                    "url": url,
                    "title": download_result["title"],
                    "source": video_type,
                    "duration": download_result["duration"],
                    "file_size_mb": download_result["file_size_mb"]
                },
                "ai_analysis": analysis,
                "generated_content": {
                    "title": analysis["generated_title"],
                    "hook": analysis["generated_hook"],
                    "confidence": analysis["confidence"]
                },
                "short_clip": {
                    "path": clip_result["path"],
                    "duration": clip_result["duration"],
                    "file_size_mb": clip_result["file_size_mb"],
                    "format": "mp4"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

# Global generator instance
generator = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ai-shorts-api-production",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "features": [
            "YouTube download (yt-dlp)",
            "Google Drive download", 
            "AI analysis (GPT-4)",
            "AI title generation",
            "AI hook generation",
            "Video editing (MoviePy)",
            "9:16 aspect ratio conversion"
        ],
        "message": "🚀 Production AI Shorts API with Title/Hook Generation ready!"
    })

@app.route('/api/generate-short', methods=['POST'])
def generate_short():
    """Main endpoint to generate AI shorts with titles and hooks"""
    global generator
    
    try:
        logger.info("\n🚀 New shorts generation request received!")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        video_url = data.get('video_url')
        video_type = data.get('video_type', 'auto')
        
        if not video_url:
            return jsonify({"error": "video_url is required"}), 400
        
        # Auto-detect video type and normalize values
        if video_type == 'auto':
            if "youtube.com" in video_url or "youtu.be" in video_url:
                video_type = "youtube"
            elif "drive.google.com" in video_url:
                video_type = "google_drive"
            else:
                return jsonify({"error": "Unsupported video URL. Please use YouTube or Google Drive links."}), 400

        # Normalize video_type values for backward compatibility
        if video_type == "drive":
            video_type = "google_drive"
        
        logger.info(f"📹 Processing: {video_url}")
        
        # Initialize generator
        generator = ProductionShortsGenerator(OPENAI_API_KEY)
        
        # Process the video
        result = generator.process_video(video_url, video_type)
        
        if "error" in result:
            return jsonify(result), 400
        
        # Get the hostname for download URL
        host = request.host
        scheme = 'https' if request.is_secure else 'http'
        
        # Prepare final response with generated content
        clip_filename = os.path.basename(result['short_clip']['path'])
        
        response_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "original_video": result["original_video"],
            "ai_analysis": result["ai_analysis"],
            "generated_content": result["generated_content"],
            "short_clip": {
                "duration": result["short_clip"]["duration"],
                "file_size_mb": result["short_clip"]["file_size_mb"],
                "format": result["short_clip"]["format"]
            },
            "download_url": f"{scheme}://{host}/api/download/{clip_filename}",
            "ai_clip_filename": clip_filename,
            "ai_clip_title": result["generated_content"]["title"],
            "ai_clip_hook": result["generated_content"]["hook"]
        }
        
        logger.info("✅ Processing complete!")
        logger.info(f"📝 Title: {result['generated_content']['title']}")
        logger.info(f"🪝 Hook: {result['generated_content']['hook']}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download the generated clip"""
    try:
        if not generator:
            return jsonify({"error": "No active generator session"}), 404
            
        file_path = os.path.join(generator.temp_dir, filename)
        if os.path.exists(file_path):
            logger.info(f"📤 Serving file: {filename}")
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        "message": "🎬 AI Shorts API - Production with Title/Hook Generation",
        "status": "running",
        "version": "3.0.0",
        "endpoints": {
            "health": "/api/health",
            "generate": "/api/generate-short",
            "download": "/api/download/<filename>"
        },
        "supported_sources": [
            "YouTube (youtube.com, youtu.be)",
            "Google Drive (drive.google.com)"
        ],
        "supported_video_types": [
            "auto (automatic detection)",
            "youtube",
            "google_drive", 
            "drive (alias for google_drive)"
        ],
        "new_features": [
            "AI-generated titles",
            "AI-generated hooks",
            "Enhanced viral content analysis"
        ]
    })

# Cleanup on shutdown
@app.teardown_appcontext
def cleanup_generator(error):
    """Clean up generator on request end"""
    global generator
    if generator:
        generator.cleanup()
        generator = None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
