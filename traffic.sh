
time python main.py --data ./data/traffic.npz --save ./save/traffic.pk --log ./logs/traffic.log --exps 3 --patience 10 \
    --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun linear \
    --multi 1 --horizon 3 --highway_window 12 --window 48 --skip 24 --ps 3 

