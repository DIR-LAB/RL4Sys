# setup.py

from setuptools import setup, find_packages

setup(
    name='RL4Sys',
    version='0.1.0',
    author='DIRLab',
    author_email='ddai@uncc.edu',
    description='A system-oriented reinforcement learning framework using Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DIR-LAB/RL4Sys',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyzmq',
        'pickle',
        # Add other dependencies required by your package
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    # Add some relevant keywords
    keywords='reinforcement learning, RL, system integration',
)
