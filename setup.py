import os
from setuptools import setup, find_packages


def parse_requirements(file):
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            if "/" not in line:
                required_packages.append(line.strip())
    return required_packages


def get_version():
    for line in open(os.path.join(os.path.dirname(__file__), "eogrow", "__init__.py")):
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
    return version


setup(
    name="eo-grow",
    python_requires=">=3.8",
    version=get_version(),
    description="Earth observation framework for scaled-up processing in Python",
    url="https://github.com/sentinel-hub/eo-grow",
    author="Sinergise EO research team",
    author_email="eoresearch@sinergise.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    extras_require={"DEV": parse_requirements("requirements-dev.txt")},
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "eogrow=eogrow.cli:EOGrowCli.main",
            "eogrow-ray=eogrow.cli:EOGrowCli.ray",
            "eogrow-template=eogrow.cli:EOGrowCli.make_template",
            "eogrow-validate=eogrow.cli:EOGrowCli.validate_config",
            "eogrow-test=eogrow.cli:EOGrowCli.run_test_pipeline",
        ]
    },
)
