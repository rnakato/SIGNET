import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="signet",
    version="0.3.10", # for test
#    version="0.1.0", # for PyPI
    license="GPL3.0",
    install_requires=[
        "numpy>=1.14.2",
        "pandas>=0.22.0",
        "leidenalg>=0.8.3",
        "eeisp>=0.5.0",
        "matplotlib",
        "seaborn",
        "networkx",
        "igraph"
    ],
    author="Ryuichiro Nakato",
    author_email="rnakato@iqb.u-tokyo.ac.jp",
    description="SIGNET: Estimation and community detection of a signed network.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nakatolab/SIGNET",
    keywords="SIGNET signed network community detection",
    scripts=['signet/signet'
             ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
