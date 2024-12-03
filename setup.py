from setuptools import setup, find_packages

setup(
    name="crypto_trading_bot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'python-binance>=1.0.17',
        'pandas>=1.5.3',
        'numpy>=1.24.2',
        'scikit-learn>=1.2.2',
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0.1',
        'ta>=0.10.2',
        'matplotlib>=3.7.1',
        'tqdm>=4.65.0'
    ]
)
