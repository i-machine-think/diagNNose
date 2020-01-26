"""
Removes `package` from the headers, and deletes unnecessary rst files
that are created by sphinx-apidoc.
"""

import glob
import os

files = glob.glob('source/diagnnose.*')

for fn in files:
    with open(fn) as f:
        content = list(f)
    content[0] = content[0][:-9] + '\n'
    content[1] = content[1][:-9] + '\n'

    with open(fn, 'w') as f:
        f.write(''.join(content))

os.remove("source/diagnnose.rst")
os.remove("source/diagnnose.typedefs.rst")
os.remove("source/modules.rst")
