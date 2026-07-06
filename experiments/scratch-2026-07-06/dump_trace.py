import sys, traceback
from bdb import Bdb
import os
import signal

pid = int(sys.argv[1])
os.kill(pid, signal.SIGUSR1) # Requires custom signal handler... wait, python does not dump trace on SIGUSR1 by default.
