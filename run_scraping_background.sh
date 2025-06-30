#!/bin/bash

# Run JGB scraping in background with proper logging
echo "Starting JGB complete collection scraping..."
echo "This will run in the background and process all 2009 shows found."
echo ""

# Create logs directory
mkdir -p logs

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run scraping with output redirection
nohup python3 run_full_scraping.py > "logs/scraping_$TIMESTAMP.log" 2>&1 &
PID=$!

echo "✓ Scraping started with PID: $PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/scraping_$TIMESTAMP.log"
echo ""
echo "Check status with:"
echo "  ps -p $PID"
echo ""
echo "Stop scraping with:"
echo "  kill $PID"
echo ""
echo "The scraper will:"
echo "  • Download audio files from 2009 JGB shows"
echo "  • Extract 250+ audio features per track"
echo "  • Generate deep learning embeddings"
echo "  • Save everything to jgb_complete_collection/"
echo ""
echo "Estimated time: Several hours depending on connection speed"