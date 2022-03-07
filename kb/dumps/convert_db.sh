SUFFIX="wiki-latest-pages-articles.xml.bz2"

lans=("bn" "de" "en" "es" "fa" "hi" "ko" "nl" "ru" "tr" "zh")

for i in "${lans[@]}"
do
   wikipedia2vec build-dump-db "${i}${SUFFIX}" "${i}.out"
done

