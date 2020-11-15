"""
Removes `package` from the headers, and deletes unnecessary rst files
that are created by sphinx-apidoc.
"""

import glob
import os

os.remove("source/diagnnose.rst")
os.remove("source/diagnnose.typedefs.rst")
os.remove("source/modules.rst")


# document __getitem__ in ActivationReader
# check in .rst file if line 22 aligns to the setup of ActivationReader

# It is stated here that this behaviour should also be possible to be set from the docstring itself,
# but I didn't get it working: https://stackoverflow.com/a/22087376/3511979
with open("source/diagnnose.activations.rst", "r") as f:
    content = list(f)
    content.insert(22, "   :special-members: __getitem__,__len__")

with open("source/diagnnose.activations.rst", "w") as f:
    f.write("".join(content))


files = glob.glob('source/diagnnose.*')

for fn in files:
    with open(fn) as f:
        content = list(f)

    content[0] = content[0][:-9].split(".")[1] + '\n'
    content[1] = content[1][:-9] + '\n'

    if "Submodules" in content[8]:
        del content[8:11]

    new_content = []

    for idx, line in enumerate(content):
        if "automodule" in line:
            snake_name = line.split(" ")[-1].split(".")[-1]
            if snake_name != content[0]:
                capital_name = " ".join([x.capitalize() for x in snake_name.strip().split("_")])
                new_content.append(capital_name + "\n")
                new_content.append("~" * len(capital_name) + "\n")
        new_content.append(line)

    with open(fn, 'w') as f:
        f.write(''.join(new_content))
