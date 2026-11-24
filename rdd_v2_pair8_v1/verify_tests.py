"""Run CPU implementation tests under the same strict backend flags."""
import json
from pathlib import Path
import sys
import unittest

import strict_runtime

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    strict_runtime.configure(json.loads((root/'protocol.json').read_text()))
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
