pandoc --number-sections --listings \
    -H auto_linebreak_listings.tex \
    --variable papersize=a4 \
    --variable urlcolor=cyan \
    -V lang=en-US \
    --highlight-style pygments \
    -s Compare_KDTree_implementations.md \
    -o Compare_KDTree_implementations.pdf \
    --template eisvogel --pdf-engine=xelatex

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer \
  -dNOPAUSE -dQUIET -dBATCH -dDetectDuplicateImages \
  -sOutputFile=Compare_KDTree_implementations_lr.pdf Compare_KDTree_implementations.pdf
