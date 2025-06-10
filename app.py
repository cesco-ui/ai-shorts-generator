#!/usr/bin/env python3
"""
🔌 Production AI Shorts API for Render
"""

import os
import json
import tempfile
import shutil
from datetime import datetime

# Disable Flask's automatic .env loading
os.environ['FLASK_SKIP_DOTENV'] = '1'

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Your API key directly embedded
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'demo-key-replace-in-render')
class ShortsGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.temp_dir = tempfile.mkdtemp()
        print(f"📁 Working directory: {self.temp_dir}")
    
    def create_mock_video(self):
        """Create a small mock video file for testing"""
        try:
            video_path = os.path.join(self.temp_dir, "video.mp4")
            with open(video_path, 'wb') as f:
                f.write(b'\x00\x00\x00\x20ftypmp42')
                f.write(b'\x00' * 1000)  # 1KB of dummy data
            
            print(f"📹 Created mock video: {video_path} ({os.path.getsize(video_path)} bytes)")
            return video_path
            
        except Exception as e:
            print(f"❌ Mock video creation failed: {e}")
            return None
    
    def process_video(self, url, video_type):
        """Mock the complete video processing workflow"""
        try:
            print(f"🎬 Processing {video_type}: {url}")
            
            # Step 1: Create mock video
            video_path = self.create_mock_video()
            if not video_path:
                return {"error": "Failed to create mock video"}
            
            # Step 2: Mock AI analysis
            print("🤖 AI analyzing video...")
            analysis = {
                "highlight_start": 30.0,
                "highlight_end": 90.0,
                "duration": 60.0,
                "reason": "High engagement segment detected",
                "confidence": 0.89,
                "transcript_sample": "This is the most interesting part..."
            }
            print(f"🎯 Found highlight: {analysis['highlight_start']}s to {analysis['highlight_end']}s")
            
            # Step 3: Create the "short" clip
            print("✂️ Creating short clip...")
            clip_path = os.path.join(self.temp_dir, "ai_short.mp4")
            shutil.copy2(video_path, clip_path)
            
            if not os.path.exists(clip_path):
                return {"error": "Failed to create clip"}
            
            file_size = os.path.getsize(clip_path) / (1024*1024)  # MB
            print(f"✅ Clip created: {file_size:.3f} MB")
            
            return {
                "success": True,
                "original_video": {
                    "url": url,
                    "title": f"Mock Video ({video_type})",
                    "source": video_type,
                    "duration": 300
                },
                "ai_analysis": analysis,
                "short_clip": {
                    "path": clip_path,
                    "duration": analysis["duration"],
                    "file_size_mb": round(file_size, 3),
                    "format": "mp4"
                }
            }
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

# Global generator instance
generator = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ai-shorts-api",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "message": "🎉 AI Shorts API ready!"
    })

@app.route('/api/generate-short', methods=['POST'])
def generate_short():
    """Main endpoint to generate AI shorts"""
    try:
        print("\n🚀 New shorts generation request!")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        video_url = data.get('video_url')
        video_type = data.get('video_type', 'auto')
        
        if not video_url:
            return jsonify({"error": "video_url is required"}), 400
        
        # Auto-detect video type
        if video_type == 'auto':
            if "youtube.com" in video_url or "youtu.be" in video_url:
                video_type = "youtube"
            elif "drive.google.com" in video_url:
                video_type = "google_drive"
            else:
                video_type = "unknown"
        
        print(f"📹 Processing {video_type}: {video_url}")
        
        # Initialize generator
        global generator
        generator = ShortsGenerator(OPENAI_API_KEY)
        
        # Process the video
        result = generator.process_video(video_url, video_type)
        
        if "error" in result:
            return jsonify(result), 400
        
        # Get the hostname for download URL
        host = request.host
        scheme = 'https' if request.is_secure else 'http'
        
        # Prepare final response
        response_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "original_video": result["original_video"],
            "ai_analysis": result["ai_analysis"],
            "short_clip": {
                "duration": result["short_clip"]["duration"],
                "file_size_mb": result["short_clip"]["file_size_mb"],
                "format": result["short_clip"]["format"]
            },
            "download_url": f"{scheme}://{host}/api/download/{os.path.basename(result['short_clip']['path'])}"
        }
        
        print("✅ Processing complete!")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download the generated clip"""
    try:
        if not generator:
            return jsonify({"error": "No active generator session"}), 404
            
        file_path = os.path.join(generator.temp_dir, filename)
        if os.path.exists(file_path):
            print(f"📤 Serving file: {filename}")
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        "message": "🎬 AI Shorts API",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "generate": "/api/generate-short",
            "download": "/api/download/<filename>"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
