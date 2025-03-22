from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pycontext",
    version="0.1.0",
    author="Subhadip Mitra",
    author_email="contact@subhadipmitra.com",
    description="A framework for building autonomous multi-agent systems with standardized context management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bassrehab/pycontext",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "redis>=4.0.0",
        "numpy>=1.20.0",
        "aiohttp>=3.8.0",
        "prometheus-client>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
        ],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "huggingface": ["transformers>=4.30.0", "torch>=2.0.0"],
    },
)
