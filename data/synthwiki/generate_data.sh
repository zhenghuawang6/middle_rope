for lengeth in 2000 5000 7000;
do
for ip in "first" "last" "halfway";
do

python ./process_data.py \
--junk_size $lengeth \
--insert_place $ip \
--model_name ../../download/Phi-3-mini-128k-instruct \

done
done