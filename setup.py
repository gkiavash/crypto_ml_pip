from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name="cryto_ml",
    url="https://github.com/gkiavash/crypto_ml_pip",
    author="Kiavash Ghamsari",
    author_email="gkiavash@gmail.com",
    # Needed to actually package something
    packages=["crypto_ml"],
    # Needed for dependencies
    install_requires=["numpy", "pandas", "ta", "pytest"],
    version="0.1",
    description="few util methods for crypto datasets",
)
