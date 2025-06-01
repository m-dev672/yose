from setuptools import setup, find_packages

setup(
    name="yose",
    version="0.1.0",
    description="",
    author="Koutarou Mori",
    author_email="m.koutarou2004@example.com",
    url="https://github.com/m-dev672/yose",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scikit-learn",
        "gensim",
        "mecab-python3",
        "POT",
        "jax",
        "ott-jax",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    include_package_data=True,
)