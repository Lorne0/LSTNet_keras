## Keras version of LSTNet


#### Experiment Result

| rrse/corr | solar         | traffic       | ele           | er            |
|-----------|---------------|---------------|---------------|---------------|
| paper     | 0.1843/0.9843 | 0.4777/0.8721 | 0.0864/0.9283 | 0.0226/0.9735 |
| keras     | 0.2196/0.9763 | 0.4924/0.8679 | *0.2756/0.9271 | 0.0319/0.9433 |

*not sure why the rrse is so poor, maybe the way using gradient clipping

