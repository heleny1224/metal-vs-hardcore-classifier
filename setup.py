from setuptools import setup, find_packages

setup(
    name='metal-vs-hardcore-classifier',
    version='1.0.0',
    description='CNN-GRU Model for Metal vs Hardcore Music Classification',
    author='Helen Yang',
    author_email='yakaimp@gmail.com',
    url='https://github.com/heleny1224/metal-vs-hardcore-classifier',
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    install_requires=[
        'torch',
        'torchaudio',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'librosa',
    ],
    
    python_requires='>=3.10',
)