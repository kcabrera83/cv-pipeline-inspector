from setuptools import setup, find_packages

setup(
    name="cv-pipeline-inspector",
    version="1.0.0",
    author="Ing. Kelvin Cabrera",
    author_email="kelvin@example.com",
    description="Computer Vision system for pipeline defect detection using image analysis",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
