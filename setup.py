import io
import os
from pathlib import Path
from setuptools import find_packages, setup

NAME = 'invoice_extractor'
DESCRIPTION = 'Invoice Extraction Model with OCR and NER'
URL = 'https://github.com/mikhaelkirenius/invoice-extractor'
EMAIL = 'mikhaelkireniusranata@gmail.com'
AUTHOR = 'Mikhael Kirenius Ranata'
REQUIRES_PYTHON = '>=3.8.0'

pwd = os.path.abspath(os.path.dirname(__file__))

def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

try:
    with open('VERSION') as f:
        version = f.read().strip()
except FileNotFoundError:
    version = '1.0.0'

setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        'invoice_extractor': [
            'models/**/*',
            'config/**/*',
        ]
    },
    install_requires=list_reqs(),
    extras_require={
        'dev': ['pytest', 'black', 'flake8'],
        'api': ['fastapi', 'uvicorn'],
        'ui': ['streamlit'],
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'invoice-extract=src.api.main:main',
            'invoice-ui=src.ui.app:main',
        ],
    },
)