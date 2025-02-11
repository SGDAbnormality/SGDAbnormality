
PYTHON="python3"
BASEFOLDER=.

#$PYTHON --version

$PYTHON $BASEFOLDER/generate_metrics.py -m "INCEPTION" -b 250 -w 0.0
$PYTHON $BASEFOLDER/generate_metrics.py -m "INCEPTION" -b 500 -w 0.0
$PYTHON $BASEFOLDER/generate_metrics.py -m "INCEPTION" -b 5000 -w 0.0

$PYTHON $BASEFOLDER/generate_metrics.py -m "INCEPTION_DISCARD" -b 250 -w 0.001
$PYTHON $BASEFOLDER/generate_metrics.py -m "INCEPTION_DISCARD" -b 250 -w 0.005
$PYTHON $BASEFOLDER/generate_metrics.py -m "INCEPTION_DISCARD" -b 250 -w 0.01
