for lengeth in 3800;
do

python ./process_data.py \
--junk_size $lengeth \
--model_name lmsys/vicuna-7b-v1.5 \
 
done