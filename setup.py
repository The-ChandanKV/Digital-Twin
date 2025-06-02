from setuptools import setup, find_packages

setup(
    name="codebrain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "gitpython>=3.1.31",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.1",
        "astroid>=2.15.0",
        "pylint>=2.17.0",
        "requests>=2.31.0",
        "websockets>=11.0.3"
    ],
    entry_points={
        'console_scripts': [
            'codebrain-collect=codebrain.data_collection.main:main',
            'codebrain-train=codebrain.model.trainer:main',
            'codebrain-api=codebrain.api.app:main'
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A digital twin of your coding style and patterns",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/codebrain",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
) 