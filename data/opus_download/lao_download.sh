if command -v wget >/dev/null 2>&1; then
    wget https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/lo.txt.gz -O lo.txt.gz
    wget https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/mono/lo.txt.gz -O lo-para.txt.gz 
else
    curl https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/lo.txt.gz --output lo.txt.gz
    curl https://object.pouta.csc.fi/OPUS-ParaCrawl-Bonus/v9/mono/lo.txt.gz --output lo-para.txt.gz
fi

if command -v gunzip >/dev/null 2>&1; then
    gunzip lo.txt.gz
    gunzip lo-para.txt.gz
else
    tar -xvzf lo.txt.gz
    tar -xvzf lo-para.txt.gz
fi


cat lo.txt lo-para.txt > lo-combined.txt
sort -u lo-combined.txt -o lo-combined-dedup.txt

echo "Done"