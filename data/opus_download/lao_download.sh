wget https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/lo.txt.gz -O lo.txt.gz
gunzip lo.txt.gz
wget https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/mono/lo.txt.gz -O lo-para.txt.gz
gunzip lo-para.txt.gz

cat lo.txt lo-para.txt > lo-combined.txt
sort -u lo-combined.txt -o lo-combined-dedup.txt

echo "Done"