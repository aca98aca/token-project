from setuptools import setup, find_packages

setup(
    name="token_sim",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular and extensible framework for simulating token economics and network dynamics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/token_sim",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 