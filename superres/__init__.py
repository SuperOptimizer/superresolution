"""Vesuvius x1 super-resolution (restoration).

Deblur / denoise / signal-recovery on Herculaneum scroll volumes *at the native
voxel grid* -- no grid expansion. The model inverts the acquisition MTF rolloff
and noise, filling attenuated frequency bands up to grid Nyquist with structure
that is data-defined rather than hallucinated.

See the README for the design rationale and the H100 deployment notes.
"""

__version__ = "0.1.0"
