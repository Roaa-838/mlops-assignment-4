# MLOps Assignment 4: CI/CD Pipeline

This repository uses GitHub Actions for Continuous Integration. On every push to a non-main branch, the pipeline automatically checks out the code, installs dependencies, lints the Python scripts, runs a PyTorch dry test, and uploads this README as an artifact.