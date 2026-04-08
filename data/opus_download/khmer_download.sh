if command -v wget >/dev/null 2>&1; then
    wget https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/km.txt.gz -O km.txt.gz
    wget https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/mono/km.txt.gz -O km-para.txt.gz 
    wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/km.txt.gz -O km-opensubtitles.txt.gz
else
    curl https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/km.txt.gz --output km.txt.gz
    curl https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/mono/km.txt.gz --output km-para.txt.gz
    curl https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/km.txt.gz --output km-opensubtitles.txt.gz
fi

if command -v gunzip >/dev/null 2>&1; then
    gunzip km.txt.gz
    gunzip km-para.txt.gz
    gunzip km-opensubtitles.txt.gz
else
    tar -xvzf km.txt.gz
    tar -xvzf km-para.txt.gz
    tar -xvzf km-opensubtitles.txt.gz
fi

cat km.txt km-para.txt km-opensubtitles.txt > km-combined.txt
sort -u km-combined.txt -o km-combined-dedup.txt

echo "Done"