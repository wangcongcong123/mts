from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="mts",
    version="0.0.1",
    author="Congcong Wang",
    author_email="wangcongcongcc@gmail.com",
    description="Pytorch MLP implementation for classification and regression with Sklearn-like datasets.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="TBA",
    download_url="TBA",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
    ],  # to add more before being published
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="machine learning basics"
)
# pip install -e .
