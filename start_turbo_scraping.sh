#!/bin/bash

echo "🚀 STARTING TURBO SCRAPER - MAXIMUM SPEED MODE"
echo "=============================================="

# Check current scraper
echo "Current scraper status:"
ps aux | grep enhanced_scraper | grep -v grep

# Kill the slow scraper
echo -e "\nStopping slow scraper..."
pkill -f "enhanced_scraper_runner.py"
sleep 2

# Show current progress
echo -e "\nCurrent progress:"
echo "Shows scraped: $(ls jgb_complete_collection/metadata/*.json 2>/dev/null | wc -l)"
echo "Audio size: $(du -sh jgb_complete_collection/audio 2>/dev/null | cut -f1)"

# Start TURBO scraper
echo -e "\n🏎️ Starting TURBO scraper with 32 parallel workers..."
nohup python3 -u turbo_scraper.py > turbo_scraper.log 2>&1 &
TURBO_PID=$!

echo "✓ Turbo scraper started (PID: $TURBO_PID)"

# Create monitoring script
cat > monitor_turbo.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "🏎️  TURBO SCRAPER STATUS"
    echo "========================"
    echo "Time: $(date)"
    echo ""
    
    # Check if running
    if pgrep -f "turbo_scraper.py" > /dev/null; then
        echo "Status: ✅ RUNNING"
        
        # Show latest progress
        echo -e "\nLatest progress:"
        tail -5 turbo_scraper.log | grep -E "\[.*\]" || tail -5 turbo_scraper.log
        
        # Count shows
        TURBO_COUNT=$(ls jgb_turbo_collection/metadata/*.json 2>/dev/null | wc -l)
        OLD_COUNT=$(ls jgb_complete_collection/metadata/*.json 2>/dev/null | wc -l)
        TOTAL_COUNT=$((TURBO_COUNT + OLD_COUNT))
        
        echo -e "\nTotal shows downloaded: $TOTAL_COUNT / 2080"
        echo "Turbo downloads: $TURBO_COUNT"
        echo "Previous downloads: $OLD_COUNT"
        
        # Show download rate
        if [ -f turbo_scraper.log ]; then
            RATE=$(tail -20 turbo_scraper.log | grep -oP 'Rate: \K[0-9.]+' | tail -1)
            if [ ! -z "$RATE" ]; then
                echo -e "\nCurrent rate: $RATE shows/minute"
                REMAINING=$((2080 - TOTAL_COUNT))
                if [ "$RATE" != "0" ]; then
                    ETA=$(echo "scale=0; $REMAINING / $RATE / 60" | bc 2>/dev/null || echo "calculating")
                    echo "Estimated time remaining: $ETA hours"
                fi
            fi
        fi
        
        # Show disk usage
        echo -e "\nDisk usage:"
        du -sh jgb_turbo_collection/audio 2>/dev/null || echo "Calculating..."
        
    else
        echo "Status: ❌ NOT RUNNING"
        echo ""
        echo "Last 10 log entries:"
        tail -10 turbo_scraper.log
    fi
    
    echo -e "\nPress Ctrl+C to exit monitor"
    sleep 5
done
EOF

chmod +x monitor_turbo.sh

echo -e "\n📊 Starting monitor..."
./monitor_turbo.sh