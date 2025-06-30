#!/bin/bash

# JGB Complete Scraping Launcher
# ==============================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          JGB Complete Collection Scraper                 ║"
echo "║                                                          ║"
echo "║  This will:                                              ║"
echo "║  • Scrape all JGB shows from sugarmegs.org             ║"
echo "║  • Download audio files                                  ║"
echo "║  • Generate deep learning embeddings                     ║"
echo "║  • Enable similarity-based remixing                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Check GPU
echo "Checking GPU status..."
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Install missing dependencies if needed
echo "Checking dependencies..."
pip install -q librosa soundfile torch torchaudio scikit-learn beautifulsoup4 h5py

# Run the scraper
echo ""
echo "Starting scraping process..."
echo "This may take several hours depending on your connection."
echo ""

# Run with nohup for long-running process
nohup python3 run_full_scraping.py > scraping_output.log 2>&1 &
SCRAPER_PID=$!

echo "Scraping started in background (PID: $SCRAPER_PID)"
echo "Monitor progress with: tail -f scraping_output.log"
echo ""
echo "To stop scraping: kill $SCRAPER_PID"