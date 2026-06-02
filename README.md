# videoseokeyword_final

## Overview
`videoseokeyword_final` is a Python web application that extracts SEO‑relevant keywords from YouTube video transcripts. Users can input a video URL, view the transcript, and see the top keywords that can boost search visibility. The project demonstrates a clean Flask architecture with reusable templates.

## Features
- **Video URL input** – Simple form to submit a YouTube link.  
- **Automatic transcript retrieval** – Fetches subtitles via the YouTube API (or `youtube-transcript-api`).  
- **Keyword extraction** – Uses NLP techniques (spaCy, RAKE, etc.) to surface the most relevant terms.  
- **Responsive UI** – Clean, Bootstrap‑styled pages for input, processing, and results.  
- **Error handling** – Graceful messages for invalid URLs, missing transcripts, or API issues.

## Tech Stack
| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, Flask |
| **NLP** | spaCy, RAKE (or similar) |
| **API** | YouTube Data API (requires `YOUR_OWN_API_KEY`) |
| **Frontend** | HTML5, CSS3, Bootstrap 5 (templates in `templates/`) |
| **Environment** | `venv` / `requirements.txt` |

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/videoseokeyword_final.git
   cd videoseokeyword_final
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root (or export directly) with:
   ```env
   YOUTUBE_API_KEY=YOUR_OWN_API_KEY
   FLASK_APP=app.py
   FLASK_ENV=development   # optional, for debug mode
   ```

5. **Download NLP models (if using spaCy)**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage
### Running the app
```bash
flask run
```
The application will be available at `http://127.0.0.1:5000/`.

### Workflow
1. Open the home page (`/`) – rendered by `templates/index.html`.  
2. Paste a YouTube video URL and submit.  
3. The server fetches the transcript, extracts keywords, and redirects to `/results`.  
4. Results are displayed using `templates/results.html`, showing the transcript snippet and the top keywords.

### File Overview
| File | Purpose |
|------|---------|
| `app.py` | Main Flask application – routes, API calls, keyword logic. |
| `templates/base.html` | Base layout with common CSS/JS includes. |
| `templates/index.html` | Home page with the video URL form. |
| `templates/results.html` | Results page displaying transcript and keywords. |

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.