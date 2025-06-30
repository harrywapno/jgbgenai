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
