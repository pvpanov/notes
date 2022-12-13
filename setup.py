# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib
import notes

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="notes",
    version=notes.__version__,
    description="collection of personal notes and code chunks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pvpanov/notes",
    author=notes.__author__,
    author_email=notes.__email__,
    classifiers=[
        "Development Status :: 2 - Fishing for ideas",
        "Intended Audience :: Statisticians/Data scientists/Quants",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="notes",
    package_dir={},
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3, <4",
    install_requires=[],
    extras_require={},
    package_data={"notes": []},
    data_files=[],
    entry_points={"console_scripts": ["notes=notes:main"]},
    project_urls={
        "Bug Reports": "https://github.com/pvpanov/notes/issues",
        "Say Thanks!": "https://saythanks.io/to/pvpanov93",
        "Source": "https://github.com/pvpanov/notes/",
    },
)