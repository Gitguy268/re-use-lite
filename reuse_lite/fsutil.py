"""Filesystem helpers for the exFAT work drive."""
import os
import shutil


def rmtree(path):
    """shutil.rmtree that tolerates macOS removing '._' AppleDouble files behind our back on exFAT."""
    def onexc(func, p, exc):
        if not isinstance(exc, FileNotFoundError):
            raise exc
    if os.path.isdir(path):
        shutil.rmtree(path, onexc=onexc)
