from setuptools import setup, find_packages

setup(
    name='test_algorithms_morphing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'analyse_datasets_accuracies=test_algorithms_morphing.final:analyse_datasets_accuracies',
        ],
    },
    author='Francisco Pimentel Serra',
    author_email='franciscopimentelserra@gmail.com',
    description='A package for testing algorithm morphing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kiko-serra/test_algorithms_morphing',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
