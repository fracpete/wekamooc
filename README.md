WekaMOOC
========

Contains code examples for the MOOC series *Data Mining with Weka*:

https://weka.waikato.ac.nz/

You can use the `WEKAMOOC_DATA` environment variable to point the
scripts to the correct data directory, rather than changing the
directory in each of the scripts manually. 

On Linux or Mac OSX, you can do this in the terminal as follows:

* just for the current command:

  `WEKAMOOC_DATA=/home/me/somewhere/data python class-1.3.py`

  `WEKAMOOC_DATA=/home/me/somewhere/data python class-1.4.py`

* for the current shell session:

  `export WEKAMOOC_DATA=/home/me/somewhere/data`

  `python class-1.3.py`

  `python class-1.4.py`

