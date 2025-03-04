#!/bin/bash

if [ ! -d "logs" ]; then
  mkdir "logs"
fi
if [ ! -d "videos" ]; then
  mkdir "videos"
fi
if ls videos/*.mp4 1> /dev/null 2>&1; then
  rm videos/*.mp4
fi

#rm videos/*.mp4
run_time=$(date +%Y%m%d%H%M%S)
download_log="download_$run_time.log"
success=0
total=0
skip_headers=1
while IFS=, read -r col1 col2
do
    if ((skip_headers))
    then
        ((skip_headers--))
    else
        echo "$col2"
        ((total++))
	echo "$col2" >> logs/"$download_log"
	python -m snapinsta "$col2" "$col1" >> logs/"$download_log" 2>> logs/"$download_log" 
        
	if [ $? -eq 0 ]; then
	    ((success ++))
        else
	    echo "...failed with exit code $?"
        fi
    fi
done < data/short_climb_links.csv

echo "Successfully downloaded $success of $total video links."
