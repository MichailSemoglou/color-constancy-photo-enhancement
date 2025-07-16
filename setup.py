from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="color-constancy-enhancement",
    version="1.0.0",
    author="Michail Semoglou",
    author_email="m.semoglou@tongji.edu.cn",
    description="Photo enhancement using color constancy algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichailSemoglou/color-constancy-photo-enhancement",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "color-constancy-enhance=color_constancy_enhancer:main",
        ],
    },
    keywords="color constancy, image processing, computer vision, photo enhancement",
    project_urls={
        "Bug Reports": "https://github.com/MichailSemoglou/color-constancy-photo-enhancement/issues",
        "Source": "https://github.com/MichailSemoglou/color-constancy-photo-enhancement",
    },
)
