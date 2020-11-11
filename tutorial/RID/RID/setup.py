from cx_Freeze import setup, Executable

import distutils
import opcode
import os

# opcode is not a virtualenv module, so we can use it to find the stdlib; this is the same
# trick used by distutils itself it installs itself into the virtualenv
distutils_path = os.path.join(os.path.dirname(opcode.__file__), 'distutils')
 
buildOptions = dict(include_files=[(distutils_path, 'distutils'), 'deep_sort/', 'tools/', 'core/'], packages=['matplotlib', 'tqdm', 'cv2', 'tensorflow', 'easydict', 'pil', 'lxml'], excludes = ["tkinter"])
 
exe = [Executable('main.py')]
 
setup(
    name='testName',
    version='0.0.1',
    author='ahn',
    description = 'description',
    options = dict(build_exe = buildOptions),
    executables = exe
)