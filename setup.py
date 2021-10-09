import sys
import os
from setuptools import setup, find_packages

setup(
    name='luxai2021',
    version='0.1.0',
    author='Cerebro AI',
    author_email='-',
    packages=find_packages(exclude=['tests*']),
    url='https://github.com/cerebro-ai/lux-ai-2021',
    license='MIT',
    description='Code for Lux AI 2021 Kaggle competition using RL with networks',
    long_description=open('README.md').read(),
    install_requires=[
            "gym",
            "numpy",
            "stable_baselines3"
    ]
)

if sys.version_info < (3,7) or sys.version_info >= (3,8):
    os.system("")
    class style():
        YELLOW = '\033[93m'
    version = str(sys.version_info.major) + "." + str(sys.version_info.minor)
    message = f'/!\ Warning, python{version} detected, you will need to use python3.7 to submit to kaggle.'
    message = style.YELLOW + message
    print(message)