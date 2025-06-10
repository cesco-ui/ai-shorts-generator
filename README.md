# AI Shorts Generator API

Production API for generating AI-powered short clips from videos.

## Endpoints

- `GET /api/health` - Health check
- `POST /api/generate-short` - Generate short clip
- `GET /api/download/<filename>` - Download generated clip

## Usage

```bash
curl -X POST https://your-app.onrender.com/api/generate-short \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://youtube.com/watch?v=..."}'
```

Built for n8n integration.
