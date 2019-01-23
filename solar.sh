
time python main.py --data ./data/solar.npz --save ./save/solar.pk --log ./logs/solar.log --exps 3 --patience 10 \
    --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear \
    --multi 1 --horizon 3 --highway_window 12 --window 36 --skip 144 --ps 3 

