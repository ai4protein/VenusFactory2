"""Separate, bounded CPU worker; no external model provider calls."""
import sys
import time
from backend.web_v2.model_platform import Platform, read

if __name__ == '__main__':
    service = Platform()
    # Let the API publish the PID before the worker updates the job record.
    for _ in range(100):
        if read(service.path('jobs', sys.argv[1])).get('pid'):
            break
        time.sleep(.05)
    service.execute(sys.argv[1])
