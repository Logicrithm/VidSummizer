# 🎬 VidSummarize - Video to Summary in Seconds!

Turn any video or YouTube link into a quick summary automatically! Perfect for studying, research, or just saving time.

## 🌟 What Does This Do?

- ⬆️ Upload videos (mp4, avi, mov, etc.) or paste a YouTube link
- 🎤 Automatically converts speech to text (transcription)
- 📝 Creates smart summaries from the transcribed text
- 📄 Generates PDF reports with both transcript and summary
- ⚡ Super fast - even for long videos!

## 🛠️ Before You Start (Things You Need)

### 1. **Python** (the programming language)
   - Download from [python.org](https://www.python.org/downloads/)
   - Version 3.8 or newer
   - **Important:** Check "Add Python to PATH" during installation on Windows

### 2. **FFmpeg** (video processing tool)
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - This helps convert videos to audio
   - **On Windows:** Add the folder to your system PATH
   - **On Mac:** Run `brew install ffmpeg`
   - **On Linux:** Run `apt-get install ffmpeg`

### 3. **Git** (optional, for downloading this project)
   - Download from [git-scm.com](https://git-scm.com/download)

---

## ⚙️ Setup (Step by Step)

### Step 1️⃣ - Download or Clone This Project

**Option A - Using Git (Easier):**
```bash
git clone https://github.com/yourusername/VidSummarize.git
cd VidSummarize
```

**Option B - Manual Download:**
1. Click the green "Code" button on GitHub
2. Click "Download ZIP"
3. Extract the zip file
4. Open the folder in your terminal/command prompt

### Step 2️⃣ - Create a Virtual Environment (Recommended!)

Think of this as a separate space just for this project - keeps things clean.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (you need to do this every time you work on the project)
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line now.

### Step 3️⃣ - Install All Required Programs

```bash
pip install -r requirements.txt
```

This downloads and installs everything the project needs (Flask, Whisper AI, etc.)

**⏱️ This might take 5-10 minutes - be patient!**

### Step 4️⃣ - Check If FFmpeg Works

```bash
ffmpeg -version
```

If you see version information, you're good! If not, make sure FFmpeg is in your PATH.

---

## 🚀 Running the Application

### Step 1️⃣ - Activate Virtual Environment (Again!)

```bash
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 2️⃣ - Start the Server

```bash
python app.py
```

You should see something like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 3️⃣ - Open in Your Browser

Click here → http://127.0.0.1:5000

Or copy-paste that URL into your browser address bar.

### Step 4️⃣ - Use the Application!

1. **Upload a video** (drag & drop or click to browse)
2. **OR paste a YouTube link**
3. **Click "Process"**
4. **Wait** while it processes (shows progress)
5. **Download** your transcript and summary as PDF

---

## 📱 What You Can Do

### Upload Videos
- MP4, AVI, MOV, WMV, MKV files
- Up to 500MB file size
- Local video files from your computer

### YouTube Links
- Paste any YouTube URL
- Works with most videos (but respect copyright!)

### Output Files
- **Transcript** - All the spoken words from the video
- **Summary** - Short version of what was said
- **PDF** - Pretty formatted report with both

---

## ❓ Common Problems & Fixes

### Problem: "Python is not recognized"
**Fix:** Python isn't in your PATH. Uninstall and reinstall Python, making sure to check "Add to PATH"

### Problem: "FFmpeg is not recognized"  
**Fix:** FFmpeg isn't in your PATH. Download it again and add it manually to system PATH

### Problem: "pip install fails"
**Fix:** Make sure your virtual environment is activated (see `(venv)` at start of terminal)

### Problem: "Port 5000 is already in use"
**Fix:** Something else is using that port. Run this instead:
```bash
python app.py --port 5001
```

### Problem: Video upload fails
**Fix:** Make sure the file is less than 500MB and is a supported format (mp4, avi, mov, etc.)

### Problem: Processing takes forever
**Fix:** 
- Large videos take time - this is normal
- YouTube videos need downloading first
- Check your internet connection

---

## 🎓 How It Works (Simple Explanation)

1. **Video Upload** → You give it a video file
2. **Audio Extract** → System pulls out just the audio
3. **Transcription** → AI listens and converts speech to text
4. **Summarization** → AI reads the text and makes it shorter
5. **PDF Generation** → Creates a nice document with everything
6. **Download** → You get your files!

---

## 📁 Folder Structure

```
VidSummarize/
├── uploads/           ← Videos you upload go here
├── outputs/           ← Generated files saved here
├── templates/         ← Website pages (HTML)
├── static/
│   ├── css/          ← Styling (colors, fonts)
│   └── js/           ← Website interactivity (JavaScript)
├── app.py            ← Main program (start here!)
├── requirements.txt   ← List of programs needed
└── README.md         ← This file!
```

---

## 🚨 Tips for Success

✅ **Always activate virtual environment** before running  
✅ **First run is slow** - it downloads AI models (this is normal!)  
✅ **Keep videos under 500MB** for faster processing  
✅ **Use modern browsers** (Chrome, Firefox, Safari, Edge)  
✅ **Check your internet** if YouTube links aren't working  

---

## 🤝 Contributing

Found a bug? Have an idea? Want to improve this?

1. Fork this repository
2. Create a branch (`git checkout -b feature/amazing-idea`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-idea`)
5. Open a Pull Request

---

## ⚖️ License

This project is open source and available under the MIT License.

---

## 🆘 Need Help?

- Check the "Common Problems" section above
- Read the code comments in `app.py` and other files
- Open an Issue on GitHub describing your problem
- Ask in the Discussions tab

---

## 🎉 You're All Set!

That's it! You now have VidSummarize running. Enjoy converting videos to summaries!

**Happy summarizing!** 🚀

```bash
python app.py
```

### 3. Open in Browser

```
http://localhost:5000
```

## 🧪 Testing the Application

### Test 1: Home Page
- Open `http://localhost:5000`
- You should see the VidSummarize landing page
- Try switching between "Upload Video" and "Paste YouTube Link" tabs

### Test 2: File Upload (Mock)
1. Click "Upload Video"
2. Select any video file (MP4, AVI, MOV, etc.)
3. Click "Start Processing"
4. You'll be redirected to the processing page
5. Watch the mock status progression
6. After a few status checks, you'll be redirected to results

### Test 3: YouTube URL (Mock)
1. Click "Paste YouTube Link"
2. Enter any YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
3. Click "Process"
4. Same flow as file upload

### Test 4: API Health Check
```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "VidSummarize API is running",
  "timestamp": "2025-01-08T..."
}
```

### Test 5: Download Files (Mock)
- On the results page, click any download button
- You'll get a mock TXT file with placeholder content

## ⚠️ Current Limitations (Will Fix in Next Steps)

❌ **No actual video processing** - Files are uploaded but not processed  
❌ **No audio extraction** - No FFmpeg integration yet  
❌ **No transcription** - No Whisper integration yet  
❌ **No summarization** - No BART/T5 integration yet  
❌ **No PDF generation** - Only TXT files for now  
❌ **Mock status progression** - Status updates are simulated  
❌ **In-memory storage** - Jobs are lost when server restarts  

## ✅ What Works

✅ Complete frontend UI/UX  
✅ File upload (saves files)  
✅ YouTube URL validation  
✅ Route handling  
✅ Error handling  
✅ Mock status API  
✅ Mock file downloads  

## 🔜 Next Steps

### Step 2: File Upload Handling & Audio Extraction
- FFmpeg integration
- Audio extraction from video
- YouTube video downloading (yt-dlp)
- Proper file management

### Step 3: Whisper Transcription
- OpenAI Whisper integration
- Multi-language transcription
- Timestamp generation

### Step 4: BART/T5 Summarization
- HuggingFace Transformers integration
- Summary generation
- Text processing

### Step 5: File Generation
- PDF generation with ReportLab
- Professional document formatting
- Chapter segmentation (optional)

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution:** Install dependencies with `pip install -r requirements.txt`

### Issue: "Permission denied" when uploading files
**Solution:** Make sure the `uploads/` folder has write permissions

### Issue: Port 5000 already in use
**Solution:** Change port in `app.py` (line at the end: `app.run(port=5000)`)

### Issue: Frontend not loading styles
**Solution:** Make sure all CSS files are in `static/css/` folder

## 📝 Notes

- This is **Step 1** of the backend implementation
- All video processing is **mocked** for now
- Status progression is **simulated** to test the frontend
- Real processing will be added in subsequent steps
- The app uses **in-memory storage** (jobs dictionary) - data is lost on restart

## 🎓 Learning Points

### What You Can Learn from This Step:
1. Flask route handling and organization
2. File upload with Flask
3. RESTful API design
4. Error handling in Flask
5. JSON response formatting
6. Frontend-backend integration

## 💡 Tips

- Keep the server running while testing
- Check the terminal for logs and errors
- Use browser DevTools (F12) to see API requests
- The mock status progression happens automatically every time you check status
- Job IDs are UUIDs for uniqueness

## 🆘 Need Help?

If you encounter any issues:
1. Check if all dependencies are installed
2. Make sure you're in the virtual environment
3. Check the terminal for error messages
4. Verify all frontend files are in the correct folders
5. Try restarting the Flask server

---

**Ready to proceed to Step 2? Let me know!** 🚀