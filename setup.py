from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_desc = f.read()

setup(
    name="sbss",
    version="0.0.1",
    description="Similarity-Based Stratified Splitting Algorithm",
    package_dir={"": "sbss"},
    packages=find_packages(where="sbss"),
    long_description=long_desc,
    long_description_content_type='text/x-rst',
    url="https://github.com/timothyckl/similarity-stratified-split",
    author="timothyckl",
    author_email="timothy.ckl@outlook.com",
    # license="",
    # classifiers=[
    #     "License :: ...",
    #     "Programming Language :: Python :: 3.10",
    #     "Operating System :: OS Independent"
    # ],
    install_requires=["numpy==1.23.5"],
    extra_require={
        "dev": ["pytest==7.4.3"]
    },
    python_require=">=3.10"
)
