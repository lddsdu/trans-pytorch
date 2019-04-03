TOOLPATH=tools
PYTHONPATH="/home/jack/anaconda3/bin/python"

proess_file_prefix=process


$PYTHONPATH ../${TOOLPATH}/${proess_file_prefix}_cn.py --chinese_corpus ../data/neu2017/NEU_cn.txt --target ../data/cn.json --vocab_file ../data/cn_vocab.txt
$PYTHONPATH ../${TOOLPATH}/${proess_file_prefix}_en.py --english_corpus ../data/neu2017/NEU_en.txt --target ../data/en.json --vocab_file ../data/en_vocab.txt