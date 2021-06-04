import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amdtracer", 
    version="0.0.1",
    author="Michael Melesse",
    author_email="micmelesse@gmail.com",
    description="A tool kit for anaylzing Machine Learning Models from AMD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/micmelesse/amdtracer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)