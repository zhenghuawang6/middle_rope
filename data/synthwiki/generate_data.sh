for lengeth in 2000 5000 7000 10000 15000;
do
for ip in "first" "last" "halfway";
do

python ./process_data.py \
--junk_size $lengeth \
--insert_place $ip \
--model_name ../../download/vicuna-7b-v1.5-16k \

done
done