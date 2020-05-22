# SMA Project B7 - Manuel DRAZYK, Zakhar TYMCHENKO

#### Repository
- data - folder with dataset [add link and description]
- algorithms - folder with api and implementation of algorithms
- - Majority Voting
- - Dawid and Skene (DS)

### Dependencies and installation (Linux & macOS)

- Ubuntu 16 or higher
- macOS 10.14.6 or higher
- python3 + numpy

### Running

- Run the analytics on the dataset (stats, plots):
```bash
    $ python3 analytics.py
```

- Run the aggregation (MV+DS):
```bash
    $ python3 main.py
```

- Change DS implementation:
- - Comment line 12 and uncomment line 13 to switch from numpy version to legacy
- - Comment line 12 and uncomment line 14 to switch from numpy version to standard
- See report for details how they are different
