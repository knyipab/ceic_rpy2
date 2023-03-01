# ceic_rpy2
A wrapper of CEIC R for Python. This is for those who do not have access to CEIC Python subscription. 

## Pre-requisites
* R installed and R pacakge `ceic` installed
* Python pacakges in `requirement.txt` including `rpy2`

## How to use
### import
```
import ceic_rpy2 as ceic

ceic.login("<username>", "<password>")
ceic.get_series(41091101)


```