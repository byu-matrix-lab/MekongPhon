wget https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/km.txt.gz -O km.txt.gz
gunzip km.txt.gz
wget https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/mono/km.txt.gz -O km-para.txt.gz
gunzip km-para.txt.gz
wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/km.txt.gz -O km-opensubtitles.txt.gz
gunzip km-opensubtitles.txt.gz

cat km.txt km-para.txt km-opensubtitles.txt > km-combined.txt
sort -u km-combined.txt -o km-combined-dedup.txt

echo "Done"