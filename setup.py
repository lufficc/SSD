from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch-ssd",
    version="1.2.0",
    packages=find_packages(exclude=['ext']),
    install_requires=[
        "torch~=1.0",
        "torchvision~=0.3",
        "opencv-python~=4.0",
        "yacs==0.1.6",
        "Vizer~=0.1.4",
    ],
    author="Congcong Li",
    author_email="luffy.lcc@gmail.com",
    description="High quality, fast, modular reference implementation of SSD in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lufficc/SSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.6",
    include_package_data=True,
)
