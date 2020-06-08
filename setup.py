from setuptools import find_packages, setup

setup(
    name="kmeans_poc",
    packages=find_packages(),
    version="0.1.0",
    description="kmeans_poc",
    author="Your name (or your organization/company/team)",
    license="MIT",
    entry_points={
        "console_scripts": ["generate_notebook=generate_notebook:generate_notebook"],
    },
)
