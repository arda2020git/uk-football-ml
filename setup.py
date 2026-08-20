from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent
REQUIREMENTS = ROOT / "requirements.txt"


def read_requirements():
    return [
        line.strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="uk-football-ml",
    version="0.1.0",
    description="Machine learning experiments for predicting football match outcomes",
    packages=[],
    python_requires=">=3.11",
    install_requires=read_requirements(),
)