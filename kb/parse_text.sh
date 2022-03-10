
lans=("bn" "de" "es" "en" "fa" "hi" "ko" "nl" "ru" "tr" "zh")
for i in "${lans[@]}"
do
   python -u parse_text.py --lan "${i}" &> log/${i}_parse_text.log &
done

