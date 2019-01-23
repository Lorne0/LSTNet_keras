## Keras version of LSTNet

### Environment
* python 3.6.0
* tensorflow 1.12.0
* Keras 2.2.0

### Usage
```
unzip data.zip
mkdir save/ logs/
./er.sh
```

### Multi-input
The original version is a little redundant since it should put the huge tensor into the model as the input.<br />
However, if the time interval is small, like 5 or 10 mins, the input may be too huge for memory and lacking of efficiency during training.<br />
Therefore, I wrote a version called **LSTNet_multi_inputs** which deconstructs the input as (1) short-term time series, like (t-3, t-2, t-1, t) and (2) long-term skip time series, like (t-2xskip, t-skip, t). <br />
The result is as good as the original one, but much faster. <br />


